from __future__ import print_function
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
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, get_scan_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model, get_criterion
from utils.evaluate_utils import scanmix_test
from utils.train_utils import scanmix_train, scanmix_eval_train, scanmix_warmup, scanmix_scan, PrototypeManager, extract_features

parser = argparse.ArgumentParser(description='DivideMix')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--inference', default=None, type=str)
parser.add_argument('--load_state_dict', default=None, type=str)
parser.add_argument('--cudaid', default=0)
parser.add_argument('--dividemix_only', action='store_true')
parser.add_argument('--lr_sl', type=float, default=None)
parser.add_argument('--use_oracle', action='store_true', help='Use oracle mode (ground truth labels for prototype)')
parser.add_argument('--lambda_proto_oracle', type=float, default=0.0, help='Lambda for prototype loss in oracle mode (default 0.0 for testing 2D-GMM only)')

parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

device = device = torch.device('cuda:{}'.format(args.cudaid))

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#meta_info
meta_info = copy.deepcopy(args.__dict__)
p = create_config(args.config_env, args.config_exp, meta_info)
meta_info['dataset'] = p['dataset']
meta_info['noise_file'] = '{}/{:.2f}'.format(p['noise_dir'], args.r)
if args.noise_mode == 'asym':
    meta_info['noise_file'] += '_asym'
elif 'semantic' in args.noise_mode:
    meta_info['noise_file'] += '_{}'.format(args.noise_mode)
meta_info['noise_file'] += '.json'
meta_info['probability'] = None
meta_info['pred'] = None
meta_info['noise_rate'] = args.r

Path(os.path.join(p['scanmix_dir'], 'savedDicts')).mkdir(parents=True, exist_ok=True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))




@torch.no_grad()
def eval_train_2dgmm(net, eval_loader, prototype_manager, device, tau=0.1, use_oracle=False, true_labels_tensor=None, predicted_noise_rate=None):
    """
    使用2D-GMM进行样本筛选（基于交叉熵损失和原型距离损失）
    
    Args:
        net: 网络模型
        eval_loader: 评估数据加载器
        prototype_manager: 原型管理器
        device: 设备
        tau: 温度参数（默认0.1）
        use_oracle: 是否使用Oracle模式（使用真实标签计算原型距离）
        true_labels_tensor: 真实标签张量（Oracle模式下使用）
        predicted_noise_rate: 预测的噪声率（用于限制筛选范围，±10%偏差）
    
    Returns:
        prob: (N,) 每个样本被判定为干净样本的概率
        pl: (N,) 预测标签
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
            # 字典格式：{'image': ..., 'target': ..., 'meta': {'index': ...}}
            inputs = batch['image'].to(device)
            targets = batch['target'].to(device)
            index = batch['meta']['index']
        elif isinstance(batch, (tuple, list)):
            # 元组格式：(img, target, index)
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
        
        # 计算相似度（相似度越高，距离越小）
        similarities = torch.mm(features_norm, prototypes_norm.t())  # (batch, num_classes)
        
        # ========== Oracle模式：使用真实标签计算原型距离 ==========
        if use_oracle and true_labels_tensor is not None:
            # 使用真实标签对应的原型
            labels_for_proto = torch.tensor([true_labels_tensor[idx].item() for idx in index], device=device)
        else:
            # 标准模式：使用预测标签
            labels_for_proto = predicted
        
        # 计算原型距离损失：取与对应类别原型的负相似度
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
    
    # Oracle模式诊断：显示原型距离损失的改善
    if use_oracle and true_labels_tensor is not None:
        print(f'[Oracle Diagnostic] Mean proto loss: {losses_proto.mean():.4f} (lower is better with perfect prototypes)')
        
        # 关键诊断：检查原型距离是否能区分干净/噪声样本
        eval_dataset = eval_loader.dataset
        if hasattr(eval_dataset, 'noise_labels'):
            noise_labels_np = torch.tensor(eval_dataset.noise_labels)
            true_labels_np = true_labels_tensor.cpu()
            is_clean = (noise_labels_np == true_labels_np)
            
            proto_loss_clean = losses_proto[is_clean].mean()
            proto_loss_noisy = losses_proto[~is_clean].mean()
            cls_loss_clean = losses_cls[is_clean].mean()
            cls_loss_noisy = losses_cls[~is_clean].mean()
            
            print(f'[Oracle Diagnostic] Loss comparison:')
            print(f'  Clean samples: cls_loss={cls_loss_clean:.4f}, proto_loss={proto_loss_clean:.4f}')
            print(f'  Noisy samples: cls_loss={cls_loss_noisy:.4f}, proto_loss={proto_loss_noisy:.4f}')
            print(f'  Separation: proto_delta={proto_loss_noisy - proto_loss_clean:.4f} (larger is better)')
            
            if proto_loss_noisy <= proto_loss_clean:
                print(f'  ⚠️  WARNING: Noisy samples have LOWER proto loss than clean!')
                print(f'       This means perfect prototypes CANNOT distinguish clean/noisy samples!')
                print(f'       Problem: Pretrained features may not align with label semantics')
    
    # 为保证GMM拟合效果，将损失分别归一化到[0, 1]
    l_cls_norm = losses_cls.reshape(-1, 1).numpy()
    l_proto_norm = losses_proto.reshape(-1, 1).numpy()
    
    # Min-Max归一化确保在同一尺度
    input_loss = np.concatenate([l_cls_norm, l_proto_norm], axis=1)  # (N, 2)
    
    print(f'Fitting 2D-GMM with input shape: {input_loss.shape}')
    
    # 拟合2D-GMM
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    
    # 预测概率
    prob = gmm.predict_proba(input_loss)
    
    # 计算哪个分量对应干净样本（通过L2范数判定）
    # 干净样本应该在两个损失维度上都较小
    means = gmm.means_  # (2, 2)
    norms = np.linalg.norm(means, axis=1)  # (2,)
    clean_component = norms.argmin()
    
    print(f'GMM means: {means}')
    print(f'GMM covariances: {gmm.covariances_}')
    print(f'Clean component (smaller L2 norm): {clean_component}')
    
    prob = prob[:, clean_component]
    prob = torch.from_numpy(prob).float()
    
    print(f'Sample selection completed. Prob range: [{prob.min():.3f}, {prob.max():.3f}]')
    
    # ========== 基于预测噪声率的样本数量限制 ==========
    if predicted_noise_rate is not None:
        total_samples = len(prob)
        expected_clean_ratio = 1.0 - predicted_noise_rate
        
        # 根据噪声率自适应调整容差：高噪声场景使用更严格的容差
        if predicted_noise_rate >= 0.7:
            tolerance = 0.05  # 极高噪声：±5%
        elif predicted_noise_rate >= 0.4:
            tolerance = 0.08  # 高噪声：±8%
        else:
            tolerance = 0.1   # 中低噪声：±10%
        
        min_clean_ratio = max(0.0, expected_clean_ratio - tolerance)
        max_clean_ratio = min(1.0, expected_clean_ratio + tolerance)
        
        min_clean_count = int(total_samples * min_clean_ratio)
        max_clean_count = int(total_samples * max_clean_ratio)
        
        # 使用阈值0.5筛选出的干净样本数
        clean_mask = prob > 0.5
        current_clean_count = clean_mask.sum().item()
        
        print(f'\n[Sample Count Control]')
        print(f'  Predicted noise rate: {predicted_noise_rate:.1%}')
        print(f'  Expected clean ratio: {expected_clean_ratio:.1%}')
        print(f'  Allowed range: [{min_clean_ratio:.1%}, {max_clean_ratio:.1%}] (±{tolerance:.0%} tolerance)')
        print(f'  Allowed clean samples: [{min_clean_count}, {max_clean_count}] out of {total_samples}')
        print(f'  Current clean samples: {current_clean_count} ({current_clean_count/total_samples:.1%})')
        
        if current_clean_count > max_clean_count:
            # 样本过多，按后验概率排序，只取top max_clean_count
            print(f'  ⚠️  Too many clean samples! Applying upper limit...')
            sorted_prob, sorted_indices = torch.sort(prob, descending=True)
            
            # 将超出上限的样本概率降低到阈值以下
            new_prob = prob.clone()
            for i in range(max_clean_count, total_samples):
                idx = sorted_indices[i]
                new_prob[idx] = min(new_prob[idx], 0.49)  # 设置为略低于阈值
            
            adjusted_count = (new_prob > 0.5).sum().item()
            print(f'  Adjusted to {adjusted_count} clean samples (top-{max_clean_count} by probability)')
            prob = new_prob
            
        elif current_clean_count < min_clean_count:
            # 样本过少，放宽阈值，取概率最高的min_clean_count个
            print(f'  ⚠️  Too few clean samples! Applying lower limit...')
            sorted_prob, sorted_indices = torch.sort(prob, descending=True)
            
            # 将前min_clean_count个样本概率提升到阈值以上
            new_prob = prob.clone()
            for i in range(min_clean_count):
                idx = sorted_indices[i]
                if new_prob[idx] <= 0.5:
                    new_prob[idx] = 0.51  # 设置为略高于阈值
            
            adjusted_count = (new_prob > 0.5).sum().item()
            print(f'  Adjusted to {adjusted_count} clean samples (top-{min_clean_count} by probability)')
            prob = new_prob
        else:
            print(f'  ✓ Within acceptable range, no adjustment needed')
    
    # ========== Oracle模式额外诊断：检查划分准确性 ==========
    if use_oracle and true_labels_tensor is not None:
        # 计算实际的噪声标签
        eval_dataset = eval_loader.dataset
        print(f'[DEBUG] eval_dataset type: {type(eval_dataset)}')
        print(f'[DEBUG] hasattr noise_labels: {hasattr(eval_dataset, "noise_labels")}')
        
        if hasattr(eval_dataset, 'noise_labels'):
            noise_labels = torch.tensor(eval_dataset.noise_labels)
            true_labels = true_labels_tensor.cpu()
            
            # 识别真正的干净样本和噪声样本
            is_clean_actual = (noise_labels == true_labels)  # 真实干净样本
            is_noisy_actual = ~is_clean_actual  # 真实噪声样本
            
            # 使用阈值划分（通常0.5）
            threshold = 0.5
            is_clean_predicted = prob > threshold  # 预测为干净
            
            # 计算准确性指标
            true_positives = (is_clean_predicted & is_clean_actual).sum().item()  # 正确识别干净样本
            false_positives = (is_clean_predicted & is_noisy_actual).sum().item()  # 噪声被误判为干净
            true_negatives = (~is_clean_predicted & is_noisy_actual).sum().item()  # 正确识别噪声
            false_negatives = (~is_clean_predicted & is_clean_actual).sum().item()  # 干净被误判为噪声
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f'\n[Oracle Quality Check]')
            print(f'  Actual clean samples: {is_clean_actual.sum().item()}')
            print(f'  Actual noisy samples: {is_noisy_actual.sum().item()}')
            print(f'  Predicted clean: {is_clean_predicted.sum().item()}')
            print(f'  True Positives (clean→clean): {true_positives}')
            print(f'  False Positives (noisy→clean): {false_positives}')
            print(f'  False Negatives (clean→noisy): {false_negatives}')
            print(f'  Precision: {precision:.4f} (of predicted clean, how many are truly clean)')
            print(f'  Recall: {recall:.4f} (of actual clean, how many are identified)')
        else:
            print(f'\n[Oracle Quality Check] SKIPPED - eval_dataset has no noise_labels attribute')
            print(f'  Available attributes: {[attr for attr in dir(eval_dataset) if not attr.startswith("_")][:20]}')
    
    print('='*50 + '\n')
    
    return prob, pl

def create_model():
    model = get_model(p, p['scan_model'])
    model = model.to(device)
    return model

# ========== 创建日志目录和文件 ==========
log_dir = os.path.join(p['scanmix_dir'], 'log')
Path(log_dir).mkdir(parents=True, exist_ok=True)

test_log = open(os.path.join(p['scanmix_dir'], 'acc.txt'), 'a')
stats_log = open(os.path.join(p['scanmix_dir'], 'stats.txt'), 'a')
fix_log = open(os.path.join(p['scanmix_dir'], 'fix.txt'), 'a')

# 详细训练日志
import datetime
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_log = open(os.path.join(log_dir, f'training_{current_time}.log'), 'w')
train_log.write('='*80 + '\n')
train_log.write(f'Training Log - Started at {current_time}\n')
train_log.write(f'Dataset: {p["dataset"]}, Noise rate: {args.r}, Noise mode: {args.noise_mode}\n')
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
        train_transformations = get_train_transformations(p)
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
    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=p['lr'], momentum=0.9, weight_decay=5e-4)
    
    # 初始化原型管理器
    # 动态获取特征维度（ResNet18-CIFAR是512维）
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        dummy_feat = net1(dummy_input, forward_pass='backbone')
        feature_dim = dummy_feat.shape[1]
    print(f'\n[INFO] Feature dimension detected: {feature_dim}')
    
    prototype_manager1 = None
    prototype_manager2 = None
    train_neighbor_indices_static = None  # 静态邻居库（来自SimCLR，不变）
    train_neighbor_indices_dynamic = None  # 动态邻居库（每20轮更新）

    if args.load_state_dict is not None:
        print('Loading saved state dict from {}'.format(args.load_state_dict))
        checkpoint = torch.load(args.load_state_dict, weights_only=False)
        net1.load_state_dict(checkpoint['net1_state_dict'])
        net2.load_state_dict(checkpoint['net2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        start_epoch = checkpoint['epoch']+1
        
        # 恢复原型管理器（如果存在）
        if 'prototype_manager1' in checkpoint and checkpoint['prototype_manager1'] is not None:
            print('Restoring prototype managers...')
            # 重建原型管理器
            pm1_state = checkpoint['prototype_manager1']
            pm2_state = checkpoint['prototype_manager2']
            
            prototype_manager1 = PrototypeManager(
                num_classes=pm1_state['num_classes'],
                feature_dim=pm1_state['feature_dim'],
                device=device,
                alpha=pm1_state['alpha'],
                queue_size=64,
                dataset_size=50000  # CIFAR-10训练集大小
            )
            prototype_manager1.prototypes = pm1_state['prototypes'].to(device)
            prototype_manager1.update_count = pm1_state['update_count']
            
            prototype_manager2 = PrototypeManager(
                num_classes=pm2_state['num_classes'],
                feature_dim=pm2_state['feature_dim'],
                device=device,
                alpha=pm2_state['alpha'],
                queue_size=64,
                dataset_size=50000  # CIFAR-10训练集大小
            )
            prototype_manager2.prototypes = pm2_state['prototypes'].to(device)
            prototype_manager2.update_count = pm2_state['update_count']
            
            print(f'Prototype managers restored. Net1 updates: {prototype_manager1.update_count}, '
                  f'Net2 updates: {prototype_manager2.update_count}')
        else:
            print('No prototype managers found in checkpoint (training before warmup or old checkpoint format)')
        
        # 恢复lr_sl和augmentation_strategy（如果从warmup后的checkpoint恢复）
        if args.lr_sl is None and start_epoch > p['warmup']:
            # 从配置文件读取lr_sl
            args.lr_sl = p.get('lr_sl', 0.0001)  # 默认0.0001
            print(f'[Checkpoint Resume] Using lr_sl = {args.lr_sl}')
            
            # 根据噪声率设置augmentation_strategy
            if args.r <= 0.6:
                p['augmentation_strategy'] = 'dividemix'
            else:
                p['augmentation_strategy'] = 'ours'
            print(f'[Checkpoint Resume] Set augmentation = {p["augmentation_strategy"]}')
            fix_log.write('Checkpoint Resume: lr_sl = %.8f, augmentation = %s (noise_rate=%.2f)\n'%(args.lr_sl, p['augmentation_strategy'], args.r))
            fix_log.flush()
        
        # test current state
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(start_epoch-1,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.flush()
    else:
        start_epoch = 0

    # ========== Oracle模式：提前加载真实标签（供所有epoch使用） ==========
    true_labels_tensor = None
    if args.use_oracle:
        print('\n' + '='*70)
        print('>>> ORACLE MODE ACTIVE: Loading ground truth labels')
        print('='*70)
        # 获取eval_loader来访问真实标签
        eval_loader = get_loader(p, 'eval_train', meta_info)
        
        # NoisyDataset已经保存了true_labels
        base_dataset = eval_loader.dataset
        if hasattr(base_dataset, 'true_labels'):
            true_labels = base_dataset.true_labels
        elif hasattr(base_dataset, 'dataset') and hasattr(base_dataset.dataset, 'targets'):
            true_labels = base_dataset.dataset.targets
        else:
            raise ValueError('Cannot find true labels in dataset!')
        
        true_labels_tensor = torch.tensor(true_labels, device=device)
        print(f'Loaded {len(true_labels_tensor)} ground truth labels')
        print(f'Label range: [{true_labels_tensor.min().item()}, {true_labels_tensor.max().item()}]')
        
        # 显示完整的类别分布统计
        unique, counts = torch.unique(true_labels_tensor, return_counts=True)
        print('Class distribution (all 50000 samples):')
        for cls, cnt in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            print(f'  Class {cls}: {cnt} samples')
        
        print('='*70 + '\n')

    all_loss = [[],[]] # save the history of losses from two networks
    predicted_noise_rate = None  # 保存预测的噪声率用于后续2D-GMM筛选限制
    
    # 初始化训练过程中使用的变量（用于断点重连）
    noise_level = "Unknown"
    lambda_proto = 0.0
    neighbor_fusion_alpha = 1.0

    for epoch in range(start_epoch, p['num_epochs']+1):   
        lr=p['lr']
        if epoch >= 150:
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
            scanmix_warmup(epoch,net1,optimizer1,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)    
            print('\nWarmup Net2')
            scanmix_warmup(epoch,net2,optimizer2,warmup_trainloader, CEloss, conf_penalty, args.noise_mode, device=device)

            if epoch == p['warmup']-1:
                prob1,_,_=scanmix_eval_train(args,net1,[], epoch, eval_loader, CE, device=device)   
                prob2,_,_=scanmix_eval_train(args,net2,[], epoch, eval_loader, CE, device=device)
                pred1 = (prob1 > p['p_threshold'])      
                pred2 = (prob2 > p['p_threshold'])
                noise1 = len((~pred1).nonzero()[0])/len(pred1)
                noise2 = len((~pred2).nonzero()[0])/len(pred2)
                predicted_noise_rate = (noise1 + noise2) / 2  # 保存到外层变量
                log_info('\n[DIAGNOSIS] PREDICTED NOISE RATE: {:.4f} (Net1: {:.4f}, Net2: {:.4f})'.format(predicted_noise_rate, noise1, noise2))
                fix_log.write('Epoch:%d PREDICTED NOISE RATE: %.4f (Net1: %.4f, Net2: %.4f)\n'%(epoch, predicted_noise_rate, noise1, noise2))
                
                # 从配置文件读取lr_sl
                if args.lr_sl is None:
                    args.lr_sl = p.get('lr_sl', 0.0001)  # 默认0.0001
                log_info(f'[DIAGNOSIS] Using lr_sl = {args.lr_sl}')
                
                # 根据预测噪声率设置augmentation_strategy
                if predicted_noise_rate <= 0.6:
                    p['augmentation_strategy'] = 'dividemix'
                else:
                    p['augmentation_strategy'] = 'ours'
                log_info(f'[DIAGNOSIS] Set augmentation = {p["augmentation_strategy"]}')
                fix_log.write('Set lr_sl = %.8f, augmentation = %s\n'%(args.lr_sl, p['augmentation_strategy']))
                fix_log.flush()
                
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
                
                # 加载K邻居信息（静态库 - 来自SimCLR预训练）
                neighbor_path = os.path.join(p['pretext_dir'], 'topk-train-neighbors.npy')
                if os.path.exists(neighbor_path):
                    train_neighbor_indices_static = torch.from_numpy(np.load(neighbor_path)).long()
                    print(f'Loaded STATIC K-neighbors from {neighbor_path}, shape: {train_neighbor_indices_static.shape}')
                    # 初始时，动态库也使用静态库
                    train_neighbor_indices_dynamic = train_neighbor_indices_static.clone()
                    print(f'Initialized DYNAMIC K-neighbors (will be updated every 20 epochs)')
                else:
                    print(f'WARNING: K-neighbors not found at {neighbor_path}')
                    print('Dual-source neighbor fusion will be disabled')
                    train_neighbor_indices_static = None
                    train_neighbor_indices_dynamic = None
                
                # 提取特征并初始化原型 - Net1
                print('\n[Net1] Extracting features and initializing prototypes...')
                features1, noisy_targets1, predictions1, probs1 = extract_features(net1, eval_loader, device)
                
                # 从配置文件读取原型学习超参数，提供默认值以保持向后兼容
                prototype_alpha_low = p.get('prototype_alpha_low', 0.9)
                prototype_alpha_medium = p.get('prototype_alpha_medium', 0.95)
                prototype_alpha_high = p.get('prototype_alpha_high', 0.99)
                prototype_queue_size = p.get('prototype_queue_size', 64)
                
                # 根据噪声率自适应设置alpha
                if args.r <= 0.4:
                    prototype_alpha = prototype_alpha_low  # 低噪声：更快的更新
                elif args.r <= 0.7:
                    prototype_alpha = prototype_alpha_medium  # 中等噪声
                else:
                    prototype_alpha = prototype_alpha_high  # 高噪声：更保守的更新
                print(f'  Using adaptive momentum alpha={prototype_alpha:.2f} for noise_rate={args.r:.1%}')
                
                prototype_manager1 = PrototypeManager(p['num_classes'], features1.shape[1], device, alpha=prototype_alpha, queue_size=prototype_queue_size, dataset_size=50000)
                
                # ========== Oracle模式：使用真实标签初始化原型 ==========
                if args.use_oracle:
                    print('\n' + '='*70)
                    print('>>> ORACLE MODE: Initializing prototypes with GROUND TRUTH labels')
                    print('='*70)
                    # true_labels_tensor已在主循环前加载，直接使用
                    print(f'Using pre-loaded {len(true_labels_tensor)} ground truth labels')
                    
                    prototype_manager1.initialize_prototypes_oracle(
                        features1.to(device), 
                        true_labels_tensor
                    )
                else:
                    print('>>> Initializing prototypes with PREDICTED labels (adaptive alpha fusion)')
                    prototype_manager1.initialize_prototypes(
                        features1.to(device), 
                        predictions1.to(device),
                        simclr_clusters.to(device), 
                        pred_probs=probs1.to(device)
                    )
                
                # 提取特征并初始化原型 - Net2
                print('\n[Net2] Extracting features and initializing prototypes...')
                features2, noisy_targets2, predictions2, probs2 = extract_features(net2, eval_loader, device)
                
                prototype_manager2 = PrototypeManager(p['num_classes'], features2.shape[1], device, alpha=prototype_alpha, queue_size=prototype_queue_size, dataset_size=50000)
                
                if args.use_oracle:
                    print('>>> ORACLE MODE: Initializing prototypes with GROUND TRUTH labels')
                    # 使用相同的真实标签张量
                    prototype_manager2.initialize_prototypes_oracle(
                        features2.to(device), 
                        true_labels_tensor
                    )
                else:
                    prototype_manager2.initialize_prototypes(
                        features2.to(device), 
                        predictions2.to(device), 
                        simclr_clusters.to(device), 
                        pred_probs=probs2.to(device)
                    )
                
                print('\n' + '='*60)
                print('Prototype Initialization Completed!')
                print('='*60 + '\n')
    
        else:         
            print('\n' + '='*60)
            print(f'Epoch {epoch}: E-Step - Sample Selection with 2D-GMM')
            print('='*60)
            
            # ========== 每20个epoch重新计算动态邻居库（跳过原型初始化epoch） ==========
            # 注意：避免在原型初始化的epoch（warmup结束）更新邻居，此时特征空间不稳定
            if (epoch - p['warmup']) % 20 == 0 and (epoch - p['warmup']) > 0 and train_neighbor_indices_static is not None:
                print('\n' + '='*70)
                print(f'[Dynamic Neighbor Update] Epoch {epoch}: Recomputing KNN')
                print('='*70)
                
                # Step 1: 提取所有样本特征（eval模式，无shuffle）
                print('Step 1: Extracting features from current model...')
                align_loader = get_loader(p, 'align', meta_info)  # 需要创建align模式的loader
                
                net1.eval()
                net2.eval()
                features_list1 = []
                features_list2 = []
                
                with torch.no_grad():
                    for batch in align_loader:
                        # 处理不同的数据格式
                        if isinstance(batch, dict):
                            inputs = batch['image'].to(device)
                        elif isinstance(batch, (tuple, list)):
                            inputs = batch[0].to(device)
                        else:
                            raise ValueError(f'Unknown batch format: {type(batch)}')
                        
                        feat1 = net1(inputs, forward_pass='backbone')
                        feat2 = net2(inputs, forward_pass='backbone')
                        features_list1.append(feat1.cpu())
                        features_list2.append(feat2.cpu())
                
                all_features1 = torch.cat(features_list1, dim=0)
                all_features2 = torch.cat(features_list2, dim=0)
                all_features = (all_features1 + all_features2) / 2  # 平均两个网络的特征
                
                # Step 2: L2归一化
                all_features = F.normalize(all_features, dim=1)
                print(f'Feature shape: {all_features.shape}')
                
                # Step 3: 重新计算KNN
                print('Step 2: Computing KNN with cosine similarity...')
                num_neighbors = train_neighbor_indices_static.shape[1]
                
                # 使用PyTorch计算余弦相似度（已L2归一化，所以直接矩阵乘法）
                similarity_matrix = torch.mm(all_features, all_features.t())
                
                # 对每个样本找出Top-K邻居（排除自己）
                train_neighbor_indices_dynamic = torch.zeros_like(train_neighbor_indices_static)
                for i in range(similarity_matrix.shape[0]):
                    # 将自己的相似度设为-1，避免选到自己
                    similarity_matrix[i, i] = -1
                    _, indices = torch.topk(similarity_matrix[i], num_neighbors, largest=True)
                    train_neighbor_indices_dynamic[i] = indices
                
                print(f'Updated dynamic neighbors, shape: {train_neighbor_indices_dynamic.shape}')
                print('='*70 + '\n')
                
                net1.train()
                net2.train()
            
            # ========== 每5个epoch重新计算噪声率 ==========
            if (epoch - p['warmup']) % 5 == 0:
                print('\n[Noise Rate Re-estimation]')
                print(f'Re-calculating noise rate at epoch {epoch}...')
                temp_prob1, _, _ = scanmix_eval_train(args, net1, [], epoch, eval_loader, CE, device=device)
                temp_prob2, _, _ = scanmix_eval_train(args, net2, [], epoch, eval_loader, CE, device=device)
                temp_pred1 = (temp_prob1 > p['p_threshold'])
                temp_pred2 = (temp_prob2 > p['p_threshold'])
                temp_noise1 = len((~temp_pred1).nonzero()[0]) / len(temp_pred1)
                temp_noise2 = len((~temp_pred2).nonzero()[0]) / len(temp_pred2)
                predicted_noise_rate = (temp_noise1 + temp_noise2) / 2
                log_info(f'Updated noise rate: {predicted_noise_rate:.4f} (Net1: {temp_noise1:.4f}, Net2: {temp_noise2:.4f})')
                fix_log.write('Epoch:%d Updated PREDICTED NOISE RATE: %.4f (Net1: %.4f, Net2: %.4f)\n' % (epoch, predicted_noise_rate, temp_noise1, temp_noise2))
                fix_log.flush()
            
            # 使用2D-GMM进行样本筛选（传递Oracle参数和预测噪声率）
            if args.use_oracle:
                print(f'\n[Oracle Mode Active] Using ground truth labels for prototype distance in 2D-GMM')
                prob1, pl_1 = eval_train_2dgmm(net1, eval_loader, prototype_manager1, device, use_oracle=True, true_labels_tensor=true_labels_tensor, predicted_noise_rate=predicted_noise_rate)
                prob2, pl_2 = eval_train_2dgmm(net2, eval_loader, prototype_manager2, device, use_oracle=True, true_labels_tensor=true_labels_tensor, predicted_noise_rate=predicted_noise_rate)
            else:
                prob1, pl_1 = eval_train_2dgmm(net1, eval_loader, prototype_manager1, device, predicted_noise_rate=predicted_noise_rate)
                prob2, pl_2 = eval_train_2dgmm(net2, eval_loader, prototype_manager2, device, predicted_noise_rate=predicted_noise_rate)
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])
            
            clean_ratio1 = pred1.sum() / len(pred1)
            clean_ratio2 = pred2.sum() / len(pred2)
            labeled_count1 = pred2.sum().item()
            labeled_count2 = pred1.sum().item()
            if epoch == p['warmup'] + 1 or epoch % 2 == 1:
                log_info(f'[DIAGNOSIS] Clean sample ratio - Net1: {clean_ratio1:.4f}, Net2: {clean_ratio2:.4f}')
                log_info(f'[DIAGNOSIS] Labeled sample count - Net1: {int(labeled_count1)}, Net2: {int(labeled_count2)}')
                fix_log.write('Epoch:%d Clean sample ratio - Net1: %.4f, Net2: %.4f\n'%(epoch, clean_ratio1, clean_ratio2))
                fix_log.write('Epoch:%d Labeled sample count - Net1: %d, Net2: %d\n'%(epoch, int(labeled_count1), int(labeled_count2)))
                fix_log.flush()

            print('[DM] Train Net1')
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            
            # ========== Oracle模式配置 ==========
            if args.use_oracle:
                lambda_proto = args.lambda_proto_oracle
                noise_level = "Oracle"
                log_info(f'\n[Oracle Prototype Strategy] Epoch {epoch}')
                print(f'  lambda_proto = {lambda_proto:.4f}')
                
                if lambda_proto > 0:
                    print(f'  Mode: Oracle Prototypes + Contrastive Loss')
                    print(f'  - Prototypes: Ground Truth labels')
                    print(f'  - Contrastive Loss: Ground Truth labels (weight={lambda_proto:.2f})')
                else:
                    print(f'  Mode: Oracle Prototypes for 2D-GMM ONLY (Testing Data Partitioning)')
                    print(f'  - Prototypes initialized/updated with: Ground Truth labels')
                    print(f'  - Prototypes used for: 2D-GMM sample selection ONLY')
                    print(f'  - Contrastive Loss: DISABLED (lambda_proto=0)')
                    print(f'  - Purpose: Test perfect prototype impact on data partitioning')
                
                print('='*70)
                
                # 获取真实标签映射（用于训练中的原型更新）
                base_dataset = labeled_trainloader.dataset.dataset
                true_labels_map = {i: base_dataset.targets[i] for i in range(len(base_dataset.targets))}
            else:
                # 标准模式：根据预测的噪声率自适应设置原型对比损失权重
                # 从配置文件读取lambda_proto相关参数
                lambda_proto_low_start = p.get('lambda_proto_low_start', 0.5)
                lambda_proto_low_end = p.get('lambda_proto_low_end', 1.0)
                lambda_proto_low_rampup = p.get('lambda_proto_low_rampup', 30)
                lambda_proto_medium = p.get('lambda_proto_medium', 0.5)
                lambda_proto_high = p.get('lambda_proto_high', 0.2)
                
                if predicted_noise_rate is not None:
                    if predicted_noise_rate <= 0.3:
                        # 低噪声：从start逐渐增长到end
                        warmup_end = p['warmup']
                        rampup_epochs = lambda_proto_low_rampup
                        rampup_end = warmup_end + rampup_epochs
                        
                        if epoch < rampup_end:
                            # 线性增长
                            progress = (epoch - warmup_end) / rampup_epochs  # 0 -> 1
                            lambda_proto = lambda_proto_low_start + (lambda_proto_low_end - lambda_proto_low_start) * progress
                        else:
                            lambda_proto = lambda_proto_low_end
                        
                        noise_level = "Low"
                        log_info(f'\n[Prototype Strategy] Epoch {epoch}: Low Noise Mode')
                        log_info(f'  Predicted noise rate: {predicted_noise_rate:.1%}')
                        log_info(f'  lambda_proto = {lambda_proto:.3f} (Epoch {warmup_end}→{rampup_end}: {lambda_proto_low_start:.1f}→{lambda_proto_low_end:.1f})')
                        
                    elif predicted_noise_rate <= 0.7:
                        lambda_proto = lambda_proto_medium
                        noise_level = "Medium"
                        log_info(f'\n[Prototype Strategy] Epoch {epoch}: Medium Noise Mode')
                        log_info(f'  Predicted noise rate: {predicted_noise_rate:.1%}')
                        log_info(f'  lambda_proto = {lambda_proto:.2f} (fixed)')
                    else:
                        lambda_proto = lambda_proto_high
                        noise_level = "High"
                        log_info(f'\n[Prototype Strategy] Epoch {epoch}: High Noise Mode')
                        log_info(f'  Predicted noise rate: {predicted_noise_rate:.1%}')
                        log_info(f'  lambda_proto = {lambda_proto:.2f} (fixed)')
                else:
                    # 如果还没预测出噪声率（理论上不应该到这里），默认为0
                    lambda_proto = 0.0
                    noise_level = "Waiting"
                    log_info(f'\n[Prototype Strategy] Epoch {epoch}: Waiting for noise rate prediction')
                    print(f'  lambda_proto = {lambda_proto:.2f} (will be set after warmup)')
                
                true_labels_map = None
                print(f'  Prototypes used for: 2D-GMM sample selection + Training')
                print(f'  Training method: DivideMix + Prototype Contrastive Loss')
                print('='*70)
            
            # ========== 计算双源邻居融合权重 α ==========
            # α: 静态邻居（SimCLR）的权重，动态邻居权重为(1-α)
            # 从配置文件读取邻居融合参数
            neighbor_fusion_alpha_early = p.get('neighbor_fusion_alpha_early', 1.0)
            neighbor_fusion_alpha_late = p.get('neighbor_fusion_alpha_late', 0.8)
            neighbor_fusion_switch_epoch = p.get('neighbor_fusion_switch_epoch', 100)
            
            if epoch <= neighbor_fusion_switch_epoch:
                neighbor_fusion_alpha = neighbor_fusion_alpha_early
                print(f'\n[Dual-Source Neighbor Fusion]')
                print(f'  Epoch {epoch}: α (static weight) = {neighbor_fusion_alpha:.2f}')
                if neighbor_fusion_alpha == 1.0:
                    print(f'  Using ONLY static SimCLR neighbors (dynamic weight = 0.0)')
                else:
                    print(f'  Static: {neighbor_fusion_alpha:.1%}, Dynamic: {1-neighbor_fusion_alpha:.1%}')
                print(f'  Reason: Building stable feature space in early training')
            else:
                neighbor_fusion_alpha = neighbor_fusion_alpha_late
                print(f'\n[Dual-Source Neighbor Fusion]')
                print(f'  Epoch {epoch}: α (static weight) = {neighbor_fusion_alpha:.2f}')
                print(f'  Static: {neighbor_fusion_alpha:.1%}, Dynamic: {1-neighbor_fusion_alpha:.1%}')
                print(f'  Gradually incorporating dynamic neighbors from current model')
            print('='*70)
            
            # 训练Net1
            scanmix_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device, prototype_manager=prototype_manager1, lambda_proto=lambda_proto, neighbor_indices_static=train_neighbor_indices_static, neighbor_indices_dynamic=train_neighbor_indices_dynamic, neighbor_fusion_alpha=neighbor_fusion_alpha, use_oracle=args.use_oracle, true_labels_map=true_labels_map) # train net1  
            
            print('\n[DM] Train Net2')
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            
            # 为Net2获取真实标签映射（如果使用Oracle模式）
            if args.use_oracle:
                base_dataset = labeled_trainloader.dataset.dataset
                true_labels_map = {i: base_dataset.targets[i] for i in range(len(base_dataset.targets))}
            
            # 训练Net2
            scanmix_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device, prototype_manager=prototype_manager2, lambda_proto=lambda_proto, neighbor_indices_static=train_neighbor_indices_static, neighbor_indices_dynamic=train_neighbor_indices_dynamic, neighbor_fusion_alpha=neighbor_fusion_alpha, use_oracle=args.use_oracle, true_labels_map=true_labels_map) # train net2
            
            if not args.dividemix_only:
                # 确保lr_sl已设置
                if args.lr_sl is None:
                    args.lr_sl = p.get('lr_sl', 0.0001)
                    log_info(f'[DIAGNOSIS] Initializing lr_sl = {args.lr_sl}')
                
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = args.lr_sl    
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = args.lr_sl
                if epoch == p['warmup'] + 1:
                    log_info(f'[DIAGNOSIS] Switched to SCAN training, lr_sl = {args.lr_sl}')
                    fix_log.write('Epoch:%d Switched to SCAN training, lr_sl = %.8f\n'%(epoch, args.lr_sl))
                    fix_log.flush()  
                meta_info['predicted_labels'] = pl_2   
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net1')
                scanmix_scan(neighbor_dataloader, net1, criterion_sl, optimizer1, epoch, device,
                           neighbor_indices_static=train_neighbor_indices_static,
                           neighbor_indices_dynamic=train_neighbor_indices_dynamic,
                           neighbor_fusion_alpha=neighbor_fusion_alpha)
                meta_info['predicted_labels'] = pl_1  
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net2')
                scanmix_scan(neighbor_dataloader, net2, criterion_sl, optimizer2, epoch, device,
                           neighbor_indices_static=train_neighbor_indices_static,
                           neighbor_indices_dynamic=train_neighbor_indices_dynamic,
                           neighbor_fusion_alpha=neighbor_fusion_alpha)

        acc = scanmix_test(epoch,net1,net2,test_loader, device=device)
        
        # ========== 详细的Epoch总结 ==========
        log_info(f'\n{"="*80}')
        log_info(f'EPOCH {epoch} SUMMARY')
        log_info(f'{"="*80}')
        log_info(f'Test Accuracy: {acc:.2f}%')
        
        if epoch >= p['warmup']:
            if predicted_noise_rate is not None:
                log_info(f'Predicted Noise Rate: {predicted_noise_rate:.1%}')
            if 'clean_ratio1' in locals() and 'clean_ratio2' in locals():
                log_info(f'Clean Sample Ratio: Net1={clean_ratio1:.1%}, Net2={clean_ratio2:.1%}')
            if 'labeled_count1' in locals() and 'labeled_count2' in locals():
                log_info(f'Labeled Samples: Net1={int(labeled_count1)}, Net2={int(labeled_count2)}')
            if not args.dividemix_only:
                if 'noise_level' in locals():
                    log_info(f'Lambda Proto: {lambda_proto:.3f} ({noise_level} noise mode)')
                else:
                    log_info(f'Lambda Proto: {lambda_proto:.3f}')
                log_info(f'Neighbor Fusion Alpha: {neighbor_fusion_alpha:.2f}')
        
        log_info(f'{"="*80}\n')
        
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush()
        
        if epoch >= p['warmup'] and prototype_manager1 is not None:
            if epoch == p['warmup'] + 1 or epoch % 2 == 1:
                log_info('\n' + '-'*60)
                log_info('[DIAGNOSIS] Prototype Statistics:')
                stats1 = prototype_manager1.get_prototype_stats()
                stats2 = prototype_manager2.get_prototype_stats()
                if stats1 is not None:
                    log_info(f"[Net1] Updates: {stats1['update_count']}, "
                          f"Avg change: {stats1['avg_inter_similarity']:.4f}, "
                          f"Max similarity: {stats1['max_inter_similarity']:.4f}")
                    fix_log.write('Epoch:%d [Net1] Updates: %d, Avg change: %.4f, Max similarity: %.4f\n'%(
                        epoch, stats1['update_count'], stats1['avg_inter_similarity'], stats1['max_inter_similarity']))
                if stats2 is not None:
                    log_info(f"[Net2] Updates: {stats2['update_count']}, "
                          f"Avg change: {stats2['avg_inter_similarity']:.4f}, "
                          f"Max similarity: {stats2['max_inter_similarity']:.4f}")
                    fix_log.write('Epoch:%d [Net2] Updates: %d, Avg change: %.4f, Max similarity: %.4f\n'%(
                        epoch, stats2['update_count'], stats2['avg_inter_similarity'], stats2['max_inter_similarity']))
                fix_log.flush()
                log_info('-'*60 + '\n')
            
            # 保存原型
            if epoch % 10 == 0:  # 每10个epoch保存一次
                prototype_manager1.save_prototypes(
                    os.path.join(p['scanmix_dir'], f'prototypes_net1_epoch{epoch}.pth'))
                prototype_manager2.save_prototypes(
                    os.path.join(p['scanmix_dir'], f'prototypes_net2_epoch{epoch}.pth'))
        
        # 准备原型管理器状态（如果存在）
        pm1_state = None
        pm2_state = None
        if prototype_manager1 is not None:
            pm1_state = {
                'prototypes': prototype_manager1.prototypes.cpu(),
                'update_count': prototype_manager1.update_count,
                'num_classes': prototype_manager1.num_classes,
                'feature_dim': prototype_manager1.feature_dim,
                'alpha': prototype_manager1.alpha
            }
        if prototype_manager2 is not None:
            pm2_state = {
                'prototypes': prototype_manager2.prototypes.cpu(),
                'update_count': prototype_manager2.update_count,
                'num_classes': prototype_manager2.num_classes,
                'feature_dim': prototype_manager2.feature_dim,
                'alpha': prototype_manager2.alpha
            }
        
        torch.save({
                    'net1_state_dict': net1.state_dict(),
                    'net2_state_dict': net2.state_dict(),
                    'epoch': epoch,
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'prototype_manager1': pm1_state,
                    'prototype_manager2': pm2_state,
                    }, os.path.join(p['scanmix_dir'], 'savedDicts/checkpoint.json'))

if __name__ == "__main__":
    main()