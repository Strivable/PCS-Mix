"""
重新生成neighbor文件（使用已有的MoCo模型权重）
用于解决neighbor文件与当前数据集配置不匹配的问题
"""
import argparse
import os
import torch
import numpy as np
from termcolor import colored
from utils.config import create_config
from utils.common_config import get_val_transformations, get_train_dataset, get_val_dataset, get_val_dataloader, get_model
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
import copy


def main():
    parser = argparse.ArgumentParser(description='Regenerate neighbor files using existing MoCo model')
    parser.add_argument('--config_env', help='Location of path config file')
    parser.add_argument('--config_exp', help='Location of experiments config file')
    parser.add_argument('--cudaid', default=0)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s'%(args.cudaid)
    
    # Setup
    meta_info = copy.deepcopy(args.__dict__)
    p = create_config(args.config_env, args.config_exp, meta_info)
    meta_info['dataset'] = p['dataset']
    meta_info['probability'] = None
    meta_info['pred'] = None
    meta_info['mode'] = 'all'
    
    print(colored('='*80, 'cyan'))
    print(colored('Regenerating neighbor files for current configuration', 'cyan'))
    print(colored('='*80, 'cyan'))
    print(f"Dataset: {p['train_db_name']}")
    print(f"Num classes: {p['num_classes']}")
    print(f"Model path: {p['pretext_model']}")
    print(colored('='*80, 'cyan'))
    
    # 检查模型文件是否存在
    if not os.path.exists(p['pretext_model']):
        raise FileNotFoundError(f"Pretrained model not found: {p['pretext_model']}\nPlease train MoCo first.")
    
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    # 准备数据集
    print(colored('\nPreparing datasets...', 'blue'))
    eval_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, eval_transforms, split='train', meta_info=meta_info)
    
    val_meta_info = meta_info.copy()
    val_meta_info['mode'] = 'test'
    val_dataset = get_val_dataset(p, eval_transforms, meta_info=val_meta_info)
    
    train_loader = get_val_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 加载模型
    print(colored('\nLoading pretrained model...', 'blue'))
    model = get_model(p, p['pretext_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    
    # 移除projection head（如果存在）
    if hasattr(model.module, 'contrastive_head'):
        print(colored('Removing projection head...', 'blue'))
        model.module.contrastive_head = torch.nn.Identity()
    
    # 获取backbone维度
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_output = model(sample_batch['image'].cuda())
        backbone_dim = sample_output.shape[1]
    print(f"Backbone feature dimension: {backbone_dim}")
    
    # 创建memory banks
    print(colored('\nCreating memory banks...', 'blue'))
    memory_bank_train = MemoryBank(len(train_dataset), backbone_dim, p['num_classes'], p.get('temperature', 0.1))
    memory_bank_train.cuda()
    memory_bank_val = MemoryBank(len(val_dataset), backbone_dim, p['num_classes'], p.get('temperature', 0.1))
    memory_bank_val.cuda()
    
    # 挖掘训练集邻居
    topk = 50
    print(colored(f'\nMining Top-{topk} nearest neighbors for training set...', 'blue'))
    fill_memory_bank(train_loader, model, memory_bank_train)
    indices, acc = memory_bank_train.mine_nearest_neighbors(topk)
    print(f'Accuracy of top-{topk} nearest neighbors on train set: {100*acc:.2f}%')
    
    # 保存训练集邻居
    np.save(p['topk_neighbors_train_path'], indices)
    print(colored(f'✓ Train neighbors saved: {p["topk_neighbors_train_path"]}', 'green'))
    print(f'  Shape: {indices.shape}')
    
    # 挖掘验证集邻居
    topk_val = 5
    print(colored(f'\nMining Top-{topk_val} nearest neighbors for validation set...', 'blue'))
    fill_memory_bank(val_loader, model, memory_bank_val)
    indices_val, acc_val = memory_bank_val.mine_nearest_neighbors(topk_val)
    print(f'Accuracy of top-{topk_val} nearest neighbors on val set: {100*acc_val:.2f}%')
    
    # 保存验证集邻居
    np.save(p['topk_neighbors_val_path'], indices_val)
    print(colored(f'✓ Val neighbors saved: {p["topk_neighbors_val_path"]}', 'green'))
    print(f'  Shape: {indices_val.shape}')
    
    print(colored('\n' + '='*80, 'green'))
    print(colored('✓ Neighbor regeneration completed successfully!', 'green'))
    print(colored('='*80 + '\n', 'green'))
    print(colored('You can now run SCAN training:', 'cyan'))
    print(colored(f'python my_scan.py --config_env {args.config_env} --config_exp {args.config_exp}', 'cyan'))


if __name__ == '__main__':
    main()
