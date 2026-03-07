from __future__ import print_function
import os
# 设置OpenBLAS线程数限制（必须在导入numpy之前）
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import copy
import pdb
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, get_scan_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model, get_criterion
from utils.evaluate_utils import scanmix_big_test, scanmix_test
from utils.train_utils import scanmix_train, scanmix_big_train, scanmix_big_eval_train, scanmix_big_warmup, scanmix_scan, PrototypeManager, extract_features

parser = argparse.ArgumentParser(description='DivideMix')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--inference', default=None, type=str)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--cudaid', default=0, type=int)
parser.add_argument('--dividemix_only', action='store_true')
parser.add_argument('--lr_sl', type=float, default=None)
parser.add_argument('--use_oracle', action='store_true', help='Use oracle mode (ground truth labels for prototype)')
parser.add_argument('--lambda_proto_oracle', type=float, default=0.0, help='Lambda for prototype loss in oracle mode')

parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

# 单GPU设置
device = torch.device('cuda:{}'.format(args.cudaid))

# 创建配置（在设置随机种子之前）
meta_info = copy.deepcopy(args.__dict__)
p = create_config(args.config_env, args.config_exp, meta_info)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#meta_info
meta_info['dataset'] = p['dataset']
meta_info['probability'] = None
meta_info['pred'] = None

checkpoint_dir = p.get('scanmix_dir', 'results/{}/scanmix/'.format(p['train_db_name']))
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.join(checkpoint_dir, 'savedDicts')).mkdir(parents=True, exist_ok=True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


@torch.no_grad()
def eval_train_2dgmm(net, eval_loader, prototype_manager, device, tau=0.1, use_oracle=False, true_labels_tensor=None, predicted_noise_rate=None):
    """
    使用2D-GMM进行样本筛选（基于交叉熵损失和原型距离损失）
    """
    print('\n=== 2D-GMM Sample Selection ===')
    if use_oracle:
        print('[Oracle Mode] Using GROUND TRUTH labels for prototype distance calculation')
    net.eval()
    
    # 存储损失和预测
    losses_cls = torch.zeros(len(eval_loader.dataset))
    losses_proto = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset), dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print('Computing losses...')
    for batch_idx, batch in enumerate(eval_loader):
        # 处理不同的数据格式
        if isinstance(batch, dict):
            inputs = batch['image'].to(device)
            targets = batch['target'].to(device)
            index = batch['meta']['index']
        elif isinstance(batch, (tuple, list)):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            index = batch[2]
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')
        
        # 1. 计算分类损失 l_cls
        outputs = net(inputs, forward_pass='dm')
        _, predicted = torch.max(outputs, 1)
        loss_cls = criterion(outputs, targets)
        
        # 2. 计算原型距离损失 l_proto
        features = net(inputs, forward_pass='backbone')
        
        # 归一化
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(prototype_manager.prototypes, dim=1)
        
        # 计算相似度
        similarities = torch.mm(features_norm, prototypes_norm.t())
        
        if use_oracle and true_labels_tensor is not None:
            labels_for_proto = torch.tensor([true_labels_tensor[idx].item() for idx in index], device=device)
        else:
            labels_for_proto = predicted
        
        # 计算原型距离损失
        proto_sim = similarities[torch.arange(len(labels_for_proto)), labels_for_proto]
        loss_proto = -torch.log(torch.exp(proto_sim / tau) / torch.exp(similarities / tau).sum(dim=1))
        
        # 存储
        for b in range(inputs.size(0)):
            losses_cls[index[b]] = loss_cls[b].item()
            losses_proto[index[b]] = loss_proto[b].item()
            pl[index[b]] = predicted[b].item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'  Processed {batch_idx + 1}/{len(eval_loader)} batches')
    
    # 归一化损失到[0, 1]
    losses_cls = (losses_cls - losses_cls.min()) / (losses_cls.max() - losses_cls.min())
    losses_proto = (losses_proto - losses_proto.min()) / (losses_proto.max() - losses_proto.min())
    
    print(f'Loss ranges - cls: [{losses_cls.min():.3f}, {losses_cls.max():.3f}], '
          f'proto: [{losses_proto.min():.3f}, {losses_proto.max():.3f}]')
    
    # 拟合2D-GMM
    l_cls_norm = losses_cls.reshape(-1, 1).numpy()
    l_proto_norm = losses_proto.reshape(-1, 1).numpy()
    input_loss = np.concatenate([l_cls_norm, l_proto_norm], axis=1)
    
    print(f'Fitting 2D-GMM with input shape: {input_loss.shape}')
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    
    prob = gmm.predict_proba(input_loss)
    
    # 判定干净分量
    means = gmm.means_
    norms = np.linalg.norm(means, axis=1)
    weights = gmm.weights_
    
    print(f'GMM means: {means}, weights: {weights}, L2 norms: {norms}')
    
    clean_component = norms.argmin()
    noisy_component = 1 - clean_component
    
    # 合理性检查
    mean_diff = np.abs(means[0] - means[1]).sum()
    if mean_diff < 0.1:
        print(f'  ⚠️ GMM means too similar (diff={mean_diff:.4f}), using CE loss only')
        clean_component = 0 if means[0, 0] < means[1, 0] else 1
        noisy_component = 1 - clean_component
    
    if weights[clean_component] > 0.95:
        norm_ratio = norms[clean_component] / (norms[noisy_component] + 1e-8)
        if norm_ratio > 0.7:
            print(f'  ⚠️ Clean weight too high ({weights[clean_component]:.2%}), swapping components')
            clean_component = noisy_component
            noisy_component = 1 - clean_component
    
    print(f'Clean component: {clean_component} (L2={norms[clean_component]:.4f}, weight={weights[clean_component]:.2%})')
    
    prob = prob[:, clean_component]
    prob = torch.from_numpy(prob).float()
    
    clean_ratio = (prob > 0.5).sum().item() / len(prob)
    print(f'Initial clean ratio: {clean_ratio:.2%}')
    
    print('='*50 + '\n')
    
    return prob, pl


def create_model():
    model = get_model(p, p['scan_model'])
    model = model.to(device)
    return model


# ========== 创建日志目录和文件 ==========
log_dir = os.path.join(checkpoint_dir, 'log')
Path(log_dir).mkdir(parents=True, exist_ok=True)

test_log = open(os.path.join(checkpoint_dir, 'acc.txt'), 'a')
stats_log = open(os.path.join(checkpoint_dir, 'stats.txt'), 'a')
fix_log = open(os.path.join(checkpoint_dir, 'fix.txt'), 'a')

# 详细训练日志
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_log = open(os.path.join(log_dir, f'training_{current_time}.log'), 'w')
train_log.write('='*80 + '\n')
train_log.write(f'Training Log - Started at {current_time}\n')
train_log.write(f'Dataset: {p["dataset"]}\n')
train_log.write(f'Config: {args.config_exp}\n')
train_log.write('='*80 + '\n\n')
train_log.flush()

def log_info(message):
    """记录信息到日志文件和控制台"""
    print(message)
    train_log.write(message + '\n')
    train_log.flush() 

def get_loader(p, mode, meta_info):
    if mode == 'test':
        meta_info['mode'] = 'test'
        val_transformations = get_val_transformations(p)
        val_dataset = get_val_dataset(p, val_transformations, meta_info=meta_info)
        val_dataloader = get_val_dataloader(p, val_dataset)
        return val_dataloader
    
    elif mode == 'train':
        meta_info['mode'] = 'labeled'
        train_transformations = get_train_transformations(p)
        labeled_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        labeled_dataloader = get_train_dataloader(p, labeled_dataset)
        meta_info['mode'] = 'unlabeled'
        unlabeled_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        unlabeled_dataloader = get_train_dataloader(p, unlabeled_dataset)
        return labeled_dataloader, unlabeled_dataloader

    elif mode == 'eval_train':
        meta_info['mode'] = 'all'
        eval_transformations = get_val_transformations(p)
        eval_dataset = get_train_dataset(p, eval_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        eval_dataloader = get_val_dataloader(p, eval_dataset)
        return eval_dataloader
    
    elif mode == 'warmup':
        meta_info['mode'] = 'all'
        warmup_transformations = get_train_transformations(p)
        warmup_dataset = get_train_dataset(p, warmup_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        warmup_dataloader = get_train_dataloader(p, warmup_dataset, explicit_batch_size=p['batch_size']*2)
        return warmup_dataloader

    elif mode == 'neighbors':
        meta_info['mode'] = 'neighbor'
        train_transformations = get_scan_transformations(p)
        neighbor_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_neighbors_dataset=True, to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        neighbor_dataloader = get_train_dataloader(p, neighbor_dataset, explicit_batch_size=p['batch_size_scan'])
        return neighbor_dataloader
    
    elif mode == 'align':
        # 用于提取特征重新计算KNN（无shuffle，无增强）
        meta_info['mode'] = 'all'
        align_transformations = get_val_transformations(p)  # 使用验证时的变换（无增强）
        align_dataset = get_train_dataset(p, align_transformations, 
                                        split='train', to_noisy_dataset=p['to_noisy_dataset'], meta_info=meta_info)
        align_dataloader = get_val_dataloader(p, align_dataset)  # 无shuffle
        return align_dataloader
    
    else:
        raise NotImplementedError

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
criterion_dm, criterion_sl = get_criterion(p)
conf_penalty = NegEntropy()

def main():
    
    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    
    # 创建clone模型用于co-teaching（避免训练过程中参考网络同时更新）
    net1_clone = create_model()
    net2_clone = create_model()

    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)
    
    # 初始化原型管理器
    # 动态获取特征维度
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # webvision图像尺寸
    with torch.no_grad():
        dummy_feat = net1(dummy_input, forward_pass='backbone')
        feature_dim = dummy_feat.shape[1]
    print(f'\n[INFO] Feature dimension detected: {feature_dim}')
    
    prototype_manager1 = None
    prototype_manager2 = None
    train_neighbor_indices_static = None  # 静态邻居库
    train_neighbor_indices_dynamic = None  # 动态邻居库
    
    # 初始化噪声率跟踪变量
    predicted_noise_rate = None
    initial_noise_rate = None
    noise_rate_history_max = None
    noise_rate_history_min = None
    is_high_noise_scenario = None

    if args.load_state_dict is not None:
        print('Loading saved state dict from {}'.format(args.load_state_dict))
        checkpoint = torch.load(args.load_state_dict, weights_only=False)
        net1.load_state_dict(checkpoint['net1_state_dict'])
        net2.load_state_dict(checkpoint['net2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        start_epoch = checkpoint['epoch']+1
        
        # 恢复噪声率跟踪变量（如果存在）
        if 'predicted_noise_rate' in checkpoint:
            predicted_noise_rate = checkpoint['predicted_noise_rate']
            initial_noise_rate = checkpoint['initial_noise_rate']
            noise_rate_history_max = checkpoint['noise_rate_history_max']
            noise_rate_history_min = checkpoint['noise_rate_history_min']
            is_high_noise_scenario = checkpoint['is_high_noise_scenario']
            print(f'Restored noise rate tracking: predicted={predicted_noise_rate:.4f}, '
                  f'initial={initial_noise_rate:.4f}, strategy={"HIGH_MAX" if is_high_noise_scenario else "LOW_MIN"}')
        else:
            # 旧版本checkpoint，尝试从fix.txt日志文件恢复历史噪声率
            print('Old checkpoint format detected, attempting to recover noise rate history...')
            
            history_recovered = False
            fix_log_path = os.path.join(checkpoint_dir, 'fix.txt')
            if os.path.exists(fix_log_path):
                try:
                    print(f'Parsing {fix_log_path} for historical noise rates...')
                    with open(fix_log_path, 'r') as f:
                        lines = f.readlines()
                    
                    # 查找所有噪声率记录（格式: "Epoch:X [1D-GMM] Current: 0.xxxx, Used: 0.xxxx"）
                    noise_rates = []
                    for line in lines:
                        if '[1D-GMM]' in line and 'Current:' in line:
                            try:
                                parts = line.split(',')
                                current = float(parts[0].split('Current:')[1].strip())
                                used = float(parts[1].split('Used:')[1].strip())
                                noise_rates.append((current, used))
                            except:
                                continue
                    
                    if noise_rates:
                        # 使用最后一次记录的值
                        last_current, last_used = noise_rates[-1]
                        predicted_noise_rate = last_used
                        
                        # 从所有历史记录中计算max和min
                        all_currents = [nr[0] for nr in noise_rates]
                        noise_rate_history_max = max(all_currents)
                        noise_rate_history_min = min(all_currents)
                        
                        # 推断初始噪声率（使用历史最小值作为近似）
                        initial_noise_rate = noise_rate_history_min
                        is_high_noise_scenario = (initial_noise_rate >= 0.5)
                        
                        history_recovered = True
                        print(f'✓ Recovered from log: predicted={predicted_noise_rate:.4f}, '
                              f'MAX={noise_rate_history_max:.4f}, MIN={noise_rate_history_min:.4f}')
                        print(f'  Found {len(noise_rates)} historical records')
                        print(f'  Strategy: {"HIGH_MAX" if is_high_noise_scenario else "LOW_MIN"}')
                except Exception as e:
                    print(f'Failed to parse log file: {e}')
            
            if not history_recovered:
                print('Could not recover noise rate history, will re-estimate at next eval')
        
        # 恢复原型管理器（如果存在）
        if 'prototype_manager1' in checkpoint and checkpoint['prototype_manager1'] is not None:
            print('Restoring prototype managers...')
            pm1_state = checkpoint['prototype_manager1']
            pm2_state = checkpoint['prototype_manager2']
            
            # 获取数据集大小
            eval_loader = get_loader(p, 'eval_train', meta_info)
            dataset_size = len(eval_loader.dataset)
            
            prototype_manager1 = PrototypeManager(
                num_classes=pm1_state['num_classes'],
                feature_dim=pm1_state['feature_dim'],
                device=device,
                alpha=pm1_state['alpha'],
                queue_size=64,
                dataset_size=dataset_size
            )
            prototype_manager1.prototypes = pm1_state['prototypes'].to(device)
            prototype_manager1.update_count = pm1_state['update_count']
            
            prototype_manager2 = PrototypeManager(
                num_classes=pm2_state['num_classes'],
                feature_dim=pm2_state['feature_dim'],
                device=device,
                alpha=pm2_state['alpha'],
                queue_size=64,
                dataset_size=dataset_size
            )
            prototype_manager2.prototypes = pm2_state['prototypes'].to(device)
            prototype_manager2.update_count = pm2_state['update_count']
            
            print(f'Prototype managers restored. Net1 updates: {prototype_manager1.update_count}, '
                  f'Net2 updates: {prototype_manager2.update_count}')
        else:
            print('No prototype managers found in checkpoint (training before warmup or old checkpoint format)')
        
        # 恢复lr_sl（如果从warmup后的checkpoint恢复）
        if args.lr_sl is None and start_epoch > p['warmup']:
            args.lr_sl = p.get('lr_sl', 0.0001)
            print(f'[Checkpoint Resume] Using lr_sl = {args.lr_sl}')
            fix_log.write('Checkpoint Resume: lr_sl = %.8f\n'%(args.lr_sl))
            fix_log.flush()
        
        # 恢复邻居库（如果存在）
        neighbor_path = os.path.join(p['pretext_dir'], 'topk-train-neighbors.npy')
        if os.path.exists(neighbor_path):
            train_neighbor_indices_static = torch.from_numpy(np.load(neighbor_path)).long()
            train_neighbor_indices_dynamic = train_neighbor_indices_static.clone()
            print(f'Loaded K-neighbors from {neighbor_path}, shape: {train_neighbor_indices_static.shape}')
        
        # 同步clone模型
        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())
        
        # test current state
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(start_epoch-1,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.flush()
    else:
        start_epoch = 0
        # 初始化clone模型
        net1_clone.load_state_dict(net1.state_dict())
        net2_clone.load_state_dict(net2.state_dict())

    all_loss = [[],[]] # save the history of losses from two networks
    
    # 初始化训练过程中使用的变量
    noise_level = "Unknown"
    lambda_proto = 0.0
    neighbor_fusion_alpha = 1.0
    
    # 初始化原型管理器和邻居索引（如果未从checkpoint恢复）
    if 'prototype_manager1' not in dir() or prototype_manager1 is None:
        prototype_manager1 = None
    if 'prototype_manager2' not in dir() or prototype_manager2 is None:
        prototype_manager2 = None
    if 'train_neighbor_indices_static' not in dir() or train_neighbor_indices_static is None:
        train_neighbor_indices_static = None
        train_neighbor_indices_dynamic = None

    for epoch in range(start_epoch, p['num_epochs']+1):   
        lr=p['lr']
        if epoch >= (p['num_epochs']/2):
            lr /= 10      
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr        
        test_loader = get_loader(p, 'test', meta_info)
        eval_loader = get_loader(p, 'eval_train', meta_info)  
        
        if epoch<p['warmup']:       
            warmup_trainloader = get_loader(p, 'warmup', meta_info)
            print('Warmup Net1')
            scanmix_big_warmup(p,epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device)    
            print('\nWarmup Net2')
            warmup_trainloader = get_loader(p, 'warmup', meta_info)
            scanmix_big_warmup(p,epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device)

            if epoch == p['warmup']-1:
                output1 = {}
                output2 = {}
                scanmix_big_eval_train(p,args,net1,epoch, eval_loader, CE, device, output1)
                eval_loader = get_loader(p, 'eval_train', meta_info)
                scanmix_big_eval_train(p,args,net2,epoch, eval_loader, CE, device, output2)
                prob1, prob2 = output1['prob'], output2['prob']
                pred1 = (prob1 > p['p_threshold'])      
                pred2 = (prob2 > p['p_threshold'])
                noise1 = len((~pred1).nonzero()[0])/len(pred1)
                noise2 = len((~pred2).nonzero()[0])/len(pred2)
                predicted_noise_rate = (noise1 + noise2) / 2
                
                # 记录初始噪声率
                initial_noise_rate = predicted_noise_rate
                noise_rate_history_max = predicted_noise_rate
                noise_rate_history_min = predicted_noise_rate
                is_high_noise_scenario = (initial_noise_rate >= 0.5)
                
                log_info('\n[DIAGNOSIS] PREDICTED NOISE RATE: {:.4f} (Net1: {:.4f}, Net2: {:.4f})'.format(predicted_noise_rate, noise1, noise2))
                fix_log.write('Epoch:%d PREDICTED NOISE RATE: %.4f\n'%(epoch, predicted_noise_rate))
                
                # 从配置文件读取lr_sl
                if args.lr_sl is None:
                    args.lr_sl = p.get('lr_sl', 0.0001)
                log_info(f'[DIAGNOSIS] Using lr_sl = {args.lr_sl}')
                
                # === 初始化原型（双视角聚类）===
                print('\n' + '='*60)
                print('Initializing Prototypes with Dual-View Clustering')
                print('='*60)
                
                # 加载SimCLR聚类结果
                simclr_cluster_path = os.path.join(p['pretext_dir'], 'global_clusters.npy')
                if os.path.exists(simclr_cluster_path):
                    simclr_clusters = torch.from_numpy(np.load(simclr_cluster_path)).long()
                    print(f'Loaded SimCLR clusters from {simclr_cluster_path}')
                else:
                    print(f'WARNING: SimCLR clusters not found at {simclr_cluster_path}')
                    print('Using random clusters as fallback')
                    simclr_clusters = torch.randint(0, p['num_classes'], (len(eval_loader.dataset),))
                
                # 加载K邻居信息
                neighbor_path = os.path.join(p['pretext_dir'], 'topk-train-neighbors.npy')
                if os.path.exists(neighbor_path):
                    train_neighbor_indices_static = torch.from_numpy(np.load(neighbor_path)).long()
                    print(f'Loaded STATIC K-neighbors from {neighbor_path}, shape: {train_neighbor_indices_static.shape}')
                    train_neighbor_indices_dynamic = train_neighbor_indices_static.clone()
                else:
                    print(f'WARNING: K-neighbors not found at {neighbor_path}')
                    train_neighbor_indices_static = None
                    train_neighbor_indices_dynamic = None
                
                # 获取数据集大小
                dataset_size = len(eval_loader.dataset)
                
                # 提取特征并初始化原型 - Net1
                print('\n[Net1] Extracting features and initializing prototypes...')
                eval_loader = get_loader(p, 'eval_train', meta_info)
                features1, noisy_targets1, predictions1, probs1 = extract_features(net1, eval_loader, device)
                
                # 根据噪声率自适应设置alpha
                prototype_alpha_low = p.get('prototype_alpha_low', 0.9)
                prototype_alpha_medium = p.get('prototype_alpha_medium', 0.95)
                prototype_alpha_high = p.get('prototype_alpha_high', 0.99)
                prototype_queue_size = p.get('prototype_queue_size', 64)
                
                if predicted_noise_rate <= 0.4:
                    prototype_alpha = prototype_alpha_low
                elif predicted_noise_rate <= 0.7:
                    prototype_alpha = prototype_alpha_medium
                else:
                    prototype_alpha = prototype_alpha_high
                print(f'  Using adaptive momentum alpha={prototype_alpha:.2f} for noise_rate={predicted_noise_rate:.1%}')
                
                prototype_manager1 = PrototypeManager(p['num_classes'], features1.shape[1], device, alpha=prototype_alpha, queue_size=prototype_queue_size, dataset_size=dataset_size)
                prototype_manager1.initialize_prototypes(
                    features1.to(device), 
                    predictions1.to(device),
                    simclr_clusters.to(device), 
                    pred_probs=probs1.to(device)
                )
                
                # 提取特征并初始化原型 - Net2
                print('\n[Net2] Extracting features and initializing prototypes...')
                eval_loader = get_loader(p, 'eval_train', meta_info)
                features2, noisy_targets2, predictions2, probs2 = extract_features(net2, eval_loader, device)
                
                prototype_manager2 = PrototypeManager(p['num_classes'], features2.shape[1], device, alpha=prototype_alpha, queue_size=prototype_queue_size, dataset_size=dataset_size)
                prototype_manager2.initialize_prototypes(
                    features2.to(device), 
                    predictions2.to(device), 
                    simclr_clusters.to(device), 
                    pred_probs=probs2.to(device)
                )
                
                print('\n' + '='*60)
                print('Prototype Initialization Completed!')
                print('='*60 + '\n')
                fix_log.flush()
    
        else:         
            print('\n' + '='*60)
            print(f'Epoch {epoch}: E-Step - Sample Selection with 2D-GMM')
            print('='*60)
            
            # 使用2D-GMM进行样本筛选
            eval_loader = get_loader(p, 'eval_train', meta_info)
            prob1, pl_1 = eval_train_2dgmm(net1, eval_loader, prototype_manager1, device, predicted_noise_rate=predicted_noise_rate)
            eval_loader = get_loader(p, 'eval_train', meta_info)
            prob2, pl_2 = eval_train_2dgmm(net2, eval_loader, prototype_manager2, device, predicted_noise_rate=predicted_noise_rate)
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])
            
            clean_ratio1 = pred1.sum() / len(pred1)
            clean_ratio2 = pred2.sum() / len(pred2)
            log_info(f'[DIAGNOSIS] Clean sample ratio - Net1: {clean_ratio1:.4f}, Net2: {clean_ratio2:.4f}')
            fix_log.write('Epoch:%d Clean ratio - Net1: %.4f, Net2: %.4f\n'%(epoch, clean_ratio1, clean_ratio2))
            fix_log.flush()

            print('[DM] Train Net1')
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            # 使用原版的scanmix_big_train函数（WebVision专用，不计算unlabeled loss）
            scanmix_big_train(p, epoch, net1, net2_clone, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device)
            
            print('\n[DM] Train Net2')
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            # 使用原版的scanmix_big_train函数
            scanmix_big_train(p, epoch, net2, net1_clone, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device)
            
            # 更新clone模型（DM训练完成后）
            net1_clone.load_state_dict(net1.state_dict())
            net2_clone.load_state_dict(net2.state_dict())
            
            if not args.dividemix_only:
                if args.lr_sl is None:
                    args.lr_sl = p.get('lr_sl', 0.0001)
                
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = args.lr_sl    
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = args.lr_sl  

                # SCAN邻居融合参数
                neighbor_fusion_alpha = p.get('neighbor_fusion_alpha', 0.0)

                print('\n[SL] Train Net1')
                meta_info['predicted_labels'] = pl_2   
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                scanmix_scan(neighbor_dataloader, net1, criterion_sl, optimizer1, epoch, device,
                           neighbor_indices_static=train_neighbor_indices_static,
                           neighbor_indices_dynamic=train_neighbor_indices_dynamic,
                           neighbor_fusion_alpha=neighbor_fusion_alpha)
                
                print('\n[SL] Train Net2')
                meta_info['predicted_labels'] = pl_1  
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                scanmix_scan(neighbor_dataloader, net2, criterion_sl, optimizer2, epoch, device,
                           neighbor_indices_static=train_neighbor_indices_static,
                           neighbor_indices_dynamic=train_neighbor_indices_dynamic,
                           neighbor_fusion_alpha=neighbor_fusion_alpha)

        # 测试
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(epoch, net1, net2, test_loader, device=device)
        
        # Epoch总结
        log_info(f'\n{"="*80}')
        log_info(f'EPOCH {epoch} SUMMARY')
        log_info(f'{"="*80}')
        log_info(f'Test Accuracy: {acc:.2f}%')
        
        if epoch >= p['warmup'] and predicted_noise_rate is not None:
            log_info(f'Predicted Noise Rate: {predicted_noise_rate:.1%}')
        
        log_info(f'{"="*80}\n')
        
        print('\nEpoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush()

        if (epoch+1) % 5 == 0:
            # 准备保存原型管理器状态
            pm1_state = None
            pm2_state = None
            try:
                if prototype_manager1 is not None:
                    pm1_state = {
                        'prototypes': prototype_manager1.prototypes.cpu(),
                        'num_classes': prototype_manager1.num_classes,
                        'feature_dim': prototype_manager1.feature_dim,
                        'alpha': prototype_manager1.alpha,
                        'update_count': prototype_manager1.update_count
                    }
                if prototype_manager2 is not None:
                    pm2_state = {
                        'prototypes': prototype_manager2.prototypes.cpu(),
                        'num_classes': prototype_manager2.num_classes,
                        'feature_dim': prototype_manager2.feature_dim,
                        'alpha': prototype_manager2.alpha,
                        'update_count': prototype_manager2.update_count
                    }
            except:
                pass  # 如果原型管理器还未初始化，不保存
                
            torch.save({
                        'net1_state_dict': net1.state_dict(),
                        'net2_state_dict': net2.state_dict(),
                        'epoch': epoch,
                        'optimizer1': optimizer1.state_dict(),
                        'optimizer2': optimizer2.state_dict(),
                        'predicted_noise_rate': predicted_noise_rate,
                        'initial_noise_rate': initial_noise_rate,
                        'noise_rate_history_max': noise_rate_history_max,
                        'noise_rate_history_min': noise_rate_history_min,
                        'is_high_noise_scenario': is_high_noise_scenario,
                        'prototype_manager1': pm1_state,
                        'prototype_manager2': pm2_state,
                        }, os.path.join(checkpoint_dir, 'savedDicts/checkpoint.json'))

if __name__ == "__main__":
    main()
