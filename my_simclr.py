import argparse
import os
import torch
import numpy as np
from sklearn.cluster import KMeans

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
import copy

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--cudaid', default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '%s'%(args.cudaid)

#meta_info
meta_info = copy.deepcopy(args.__dict__)
p = create_config(args.config_env, args.config_exp, meta_info)
meta_info['dataset'] = p['dataset']
meta_info['probability'] = None
meta_info['pred'] = None
meta_info['mode'] = 'pretext'

@torch.no_grad()
def mine_global_clusters(dataloader, model, num_clusters, save_path):
    """
    挖掘全局无监督簇：利用SimCLR预训练的特征，通过K-Means聚类将数据集划分为K个视觉上相似的簇
    
    Args:
        dataloader: 数据加载器（不包含数据增强的base dataloader）
        model: SimCLR预训练的模型
        num_clusters: 聚类数量K
        save_path: 保存聚类结果的路径
    
    Returns:
        cluster_assignments: 每个样本的簇归属 (N,) numpy array
    """
    print(colored('Mining global unsupervised clusters with K-Means...', 'blue'))
    model.eval()
    
    # 提取所有样本的特征
    features_list = []
    targets_list = []
    
    print('Extracting features from all samples using SimCLR encoder...')
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target']
        
        # 使用SimCLR的完整模型提取特征（经过projection head）
        features = model(images)
        features_list.append(features.cpu())
        targets_list.append(targets)
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Processed {batch_idx + 1}/{len(dataloader)} batches')
    
    # 合并所有特征
    features = torch.cat(features_list, dim=0).numpy()  # (N, feature_dim)
    targets = torch.cat(targets_list, dim=0).numpy()    # (N,)
    
    print(f'Feature shape: {features.shape}')
    print(f'Performing K-Means clustering with K={num_clusters}...')
    
    # 执行K-Means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10, max_iter=300, verbose=1)
    cluster_assignments = kmeans.fit_predict(features)
    
    print(f'Clustering completed. Cluster distribution:')
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f'  Cluster {cluster_id}: {count} samples')
    
    # 保存聚类结果
    print(f'Saving cluster assignments to {save_path}...')
    np.save(save_path, cluster_assignments)
    
    # 同时保存聚类中心
    cluster_centers_path = save_path.replace('.npy', '_centers.npy')
    np.save(cluster_centers_path, kmeans.cluster_centers_)
    print(f'Cluster centers saved to {cluster_centers_path}')
    
    return cluster_assignments

def main():

    # Retrieve config file
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True, to_noisy_dataset=p['to_noisy_dataset'],
                                        split='train+unlabeled', meta_info=meta_info) # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms) 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_noisy_dataset=p['to_noisy_dataset'], split='train', meta_info=meta_info) # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu', weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
    
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' %(top1)) 
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine global clusters using K-Means on backbone features
    print(colored('Mining global unsupervised clusters ...', 'blue'))
    cluster_save_path = os.path.join(p['pretext_dir'], 'global_clusters.npy')
    num_clusters = p['num_classes']
    print(f'Clustering into {num_clusters} clusters...')
    cluster_assignments = mine_global_clusters(base_dataloader, model, num_clusters, cluster_save_path)
    print(colored(f'Global clusters saved to {cluster_save_path}', 'green'))

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
