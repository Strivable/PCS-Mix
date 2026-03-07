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
from utils.train_utils import scanmix_train, scanmix_eval_train, scanmix_warmup, scanmix_scan

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

class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比损失：拉近样本与目标原型，推远与其他原型
    """
    def __init__(self, temperature=0.1):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, target_labels, prototypes):
        """
        Args:
            features: (batch_size, feature_dim) 样本特征
            target_labels: (batch_size,) 目标标签（干净样本用修正标签，噪声样本用伪标签）
            prototypes: (num_classes, feature_dim) 类原型
        Returns:
            loss: 原型对比损失
        """
        # 归一化
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(prototypes, dim=1)
        
        # 计算余弦相似度 (batch_size, num_classes)
        similarities = torch.mm(features_norm, prototypes_norm.t()) / self.temperature
        
        # 计算对比损失（InfoNCE）
        # 目标：最大化与目标原型的相似度，同时最小化与其他原型的相似度
        loss = F.cross_entropy(similarities, target_labels)
        
        return loss

class PrototypeManager:
    """
    原型管理类：实现双视角聚类（有监督+无监督）和原型融合
    """
    def __init__(self, num_classes, feature_dim, device, alpha=0.5):
        """
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            device: 设备
            alpha: 融合权重，alpha*有监督 + (1-alpha)*无监督
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.alpha = alpha
        self.prototypes = None  # 融合后的原型 (num_classes, feature_dim)
        self.update_count = 0   # 更新次数
        self.prototype_history = []  # 原型变化历史
        
    def compute_class_centers(self, features, labels):
        """
        计算类中心
        Args:
            features: (N, feature_dim) 特征
            labels: (N,) 标签
        Returns:
            centers: (num_classes, feature_dim) 类中心
        """
        centers = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centers[c] = features[mask].mean(dim=0)
        return centers
    
    def hungarian_match(self, centers_a, centers_b):
        """
        使用匈牙利算法对齐两组类中心
        Args:
            centers_a: (num_classes, feature_dim) 视角A的类中心
            centers_b: (num_classes, feature_dim) 视角B的类中心
        Returns:
            mapping: (num_classes,) 视角B到视角A的映射
        """
        # 计算余弦相似度矩阵
        centers_a_norm = F.normalize(centers_a, dim=1)
        centers_b_norm = F.normalize(centers_b, dim=1)
        similarity = torch.mm(centers_a_norm, centers_b_norm.t())  # (K, K)
        
        # 转换为代价矩阵（相似度越高代价越低）
        cost_matrix = -similarity.cpu().numpy()
        
        # 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 创建映射：col_ind[i] -> row_ind[i]
        mapping = np.zeros(self.num_classes, dtype=np.int64)
        mapping[col_ind] = row_ind
        
        return mapping
    
    def initialize_prototypes(self, features, pred_labels, simclr_clusters):
        """
        初始化原型：双视角聚类+匈牙利对齐+加权融合
        Args:
            features: (N, feature_dim) 当前网络提取的特征
            pred_labels: (N,) 视角A-有监督：网络预测的标签
            simclr_clusters: (N,) 视角B-无监督：SimCLR聚类标签
        """
        print('\n=== Initializing Prototypes with Dual-View Clustering ===')
        
        # 视角A（有监督）：根据网络预测计算类中心
        print('Computing supervised centers (View A)...')
        centers_supervised = self.compute_class_centers(features, pred_labels)
        
        # 视角B（无监督）：根据SimCLR聚类计算类中心
        print('Computing unsupervised centers (View B)...')
        centers_unsupervised = self.compute_class_centers(features, simclr_clusters)
        
        # 匈牙利算法对齐
        print('Aligning clusters with Hungarian algorithm...')
        mapping = self.hungarian_match(centers_supervised, centers_unsupervised)
        print(f'Cluster mapping (unsupervised -> supervised): {mapping}')
        
        # 重新排列无监督类中心
        centers_unsupervised_aligned = centers_unsupervised[mapping]
        
        # 加权融合得到初始原型
        self.prototypes = self.alpha * centers_supervised + (1 - self.alpha) * centers_unsupervised_aligned
        self.prototypes = F.normalize(self.prototypes, dim=1)  # 归一化
        
        print(f'Prototypes initialized with shape: {self.prototypes.shape}')
        print(f'Fusion weight: alpha={self.alpha} (supervised) + {1-self.alpha} (unsupervised)')
        
    def update_prototypes(self, features, labels, momentum=0.9):
        if self.prototypes is None:
            raise ValueError("Prototypes not initialized!")
        
        old_prototypes = self.prototypes.clone()
        
        current_centers = self.compute_class_centers(features, labels)
        
        unique_labels = torch.unique(labels)
        if len(unique_labels) < self.num_classes:
            print(f'  [WARNING] Only {len(unique_labels)}/{self.num_classes} classes in batch')
        
        current_centers = F.normalize(current_centers, dim=1)
        
        self.prototypes = momentum * self.prototypes + (1 - momentum) * current_centers
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        cosine_change = 1 - torch.sum(old_prototypes * self.prototypes, dim=1)
        avg_change = cosine_change.mean().item()
        max_change = cosine_change.max().item()
        
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            print(f'  [Prototype Update #{self.update_count}] '
                  f'Avg change: {avg_change:.6f}, Max change: {max_change:.6f}')
        
        return avg_change, max_change
    
    def get_prototype_stats(self):
        """
        获取原型的统计信息
        """
        if self.prototypes is None:
            return None
        
        # 计算原型之间的相似度
        similarities = torch.mm(self.prototypes, self.prototypes.t())
        # 移除对角线（自己与自己的相似度=1）
        mask = torch.eye(self.num_classes, device=self.device).bool()
        similarities = similarities.masked_fill(mask, 0)
        
        avg_inter_similarity = similarities.sum() / (self.num_classes * (self.num_classes - 1))
        max_inter_similarity = similarities.max()
        
        return {
            'update_count': self.update_count,
            'avg_inter_similarity': avg_inter_similarity.item(),
            'max_inter_similarity': max_inter_similarity.item(),
            'prototype_norm': torch.norm(self.prototypes, dim=1).mean().item()
        }
    
    def save_prototypes(self, path):
        """
        保存原型到文件
        """
        if self.prototypes is not None:
            torch.save({
                'prototypes': self.prototypes.cpu(),
                'update_count': self.update_count,
                'num_classes': self.num_classes,
                'feature_dim': self.feature_dim,
                'alpha': self.alpha
            }, path)
    
    def load_prototypes(self, path):
        """
        从文件加载原型
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.prototypes = checkpoint['prototypes'].to(self.device)
        self.update_count = checkpoint.get('update_count', 0)
        print(f'Loaded prototypes from {path}, update_count={self.update_count}')
    
    def compute_prototype_distances(self, features):
        """
        计算特征到所有原型的距离
        Args:
            features: (N, feature_dim)
        Returns:
            distances: (N, num_classes) 余弦距离
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not initialized!")
        
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        
        # 余弦相似度
        similarities = torch.mm(features_norm, prototypes_norm.t())
        # 转换为距离
        distances = 1 - similarities
        
        return distances

@torch.no_grad()
def extract_features(net, dataloader, device):
    """
    提取所有样本的特征
    Args:
        net: 网络模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        all_features: (N, feature_dim) 特征
        all_targets: (N,) 真实标签
        all_predictions: (N,) 预测标签
    """
    net.eval()
    all_features = []
    all_targets = []
    all_predictions = []
    
    print('Extracting features from all samples...')
    for batch_idx, batch in enumerate(dataloader):
        # 处理不同的数据格式
        if isinstance(batch, dict):
            # 字典格式：{'image': ..., 'target': ...}
            inputs = batch['image'].to(device)
            targets = batch['target']
        elif isinstance(batch, (tuple, list)):
            # 元组格式：(img, target, index) 或 (img, target)
            inputs = batch[0].to(device)
            targets = batch[1]
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')
        
        # 提取特征（使用backbone，不经过分类头）
        features = net(inputs, forward_pass='backbone')
        # 获取预测（使用DivideMix head）
        outputs = net(inputs, forward_pass='dm')
        predictions = outputs.argmax(dim=1)
        
        all_features.append(features.cpu())
        all_targets.append(targets)
        all_predictions.append(predictions.cpu())
        
        if (batch_idx + 1) % 50 == 0:
            print(f'  Processed {batch_idx + 1}/{len(dataloader)} batches')
    
    all_features = torch.cat(all_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    print(f'Feature extraction completed: {all_features.shape}')
    return all_features, all_targets, all_predictions

@torch.no_grad()
def eval_train_2dgmm(net, eval_loader, prototype_manager, device, tau=0.1):
    """
    使用2D-GMM进行样本筛选（基于交叉熵损失和原型距离损失）
    
    Args:
        net: 网络模型
        eval_loader: 评估数据加载器
        prototype_manager: 原型管理器
        device: 设备
        tau: 温度参数（默认0.1）
    
    Returns:
        prob: (N,) 每个样本被判定为干净样本的概率
        pl: (N,) 预测标签
    """
    print('\n=== 2D-GMM Sample Selection ===')
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
        
        # 计算原型距离损失：取与预测类别原型的负相似度
        proto_sim = similarities[torch.arange(len(predicted)), predicted]
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
    print('='*50 + '\n')
    
    return prob, pl

def create_model():
    model = get_model(p, p['scan_model'])
    model = model.to(device)
    return model

test_log = open(os.path.join(p['scanmix_dir'], 'acc.txt'), 'w')
stats_log = open(os.path.join(p['scanmix_dir'], 'stats.txt'), 'w')
fix_log = open(os.path.join(p['scanmix_dir'], 'fix.txt'), 'w')

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
    feature_dim = 128  # 根据模型特征维度设置
    prototype_manager1 = None
    prototype_manager2 = None

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
                alpha=pm1_state['alpha']
            )
            prototype_manager1.prototypes = pm1_state['prototypes'].to(device)
            prototype_manager1.update_count = pm1_state['update_count']
            
            prototype_manager2 = PrototypeManager(
                num_classes=pm2_state['num_classes'],
                feature_dim=pm2_state['feature_dim'],
                device=device,
                alpha=pm2_state['alpha']
            )
            prototype_manager2.prototypes = pm2_state['prototypes'].to(device)
            prototype_manager2.update_count = pm2_state['update_count']
            
            print(f'Prototype managers restored. Net1 updates: {prototype_manager1.update_count}, '
                  f'Net2 updates: {prototype_manager2.update_count}')
        else:
            print('No prototype managers found in checkpoint (training before warmup or old checkpoint format)')
        
        # test current state
        test_loader = get_loader(p, 'test', meta_info)
        acc = scanmix_test(start_epoch-1,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(start_epoch-1,acc))
        test_log.flush()
    else:
        start_epoch = 0

    all_loss = [[],[]] # save the history of losses from two networks

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
                predicted_noise = (noise1 + noise2) / 2
                print('\n[DIAGNOSIS] PREDICTED NOISE RATE: {:.4f} (Net1: {:.4f}, Net2: {:.4f})'.format(predicted_noise, noise1, noise2))
                fix_log.write('Epoch:%d PREDICTED NOISE RATE: %.4f (Net1: %.4f, Net2: %.4f)\n'%(epoch, predicted_noise, noise1, noise2))
                if predicted_noise <= 0.6:
                    args.lr_sl = 0.00001
                    p['augmentation_strategy'] = 'dividemix'
                else:
                    args.lr_sl = 0.001
                    p['augmentation_strategy'] = 'ours'
                print(f'[DIAGNOSIS] Set lr_sl = {args.lr_sl}, augmentation = {p["augmentation_strategy"]}')
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
                
                # 提取特征并初始化原型 - Net1
                print('\n[Net1] Extracting features and initializing prototypes...')
                features1, targets1, predictions1 = extract_features(net1, eval_loader, device)
                prototype_manager1 = PrototypeManager(p['num_classes'], features1.shape[1], device, alpha=0.6)
                prototype_manager1.initialize_prototypes(features1.to(device), predictions1.to(device), simclr_clusters.to(device))
                
                # 提取特征并初始化原型 - Net2
                print('\n[Net2] Extracting features and initializing prototypes...')
                features2, targets2, predictions2 = extract_features(net2, eval_loader, device)
                prototype_manager2 = PrototypeManager(p['num_classes'], features2.shape[1], device, alpha=0.6)
                prototype_manager2.initialize_prototypes(features2.to(device), predictions2.to(device), simclr_clusters.to(device))
                
                print('\n' + '='*60)
                print('Prototype Initialization Completed!')
                print('='*60 + '\n')
    
        else:         
            print('\n' + '='*60)
            print(f'Epoch {epoch}: E-Step - Sample Selection with 2D-GMM')
            print('='*60)
            
            # 使用2D-GMM进行样本筛选（替代原始的GMM）
            prob1, pl_1 = eval_train_2dgmm(net1, eval_loader, prototype_manager1, device)
            prob2, pl_2 = eval_train_2dgmm(net2, eval_loader, prototype_manager2, device)
                
            pred1 = (prob1 > p['p_threshold'])      
            pred2 = (prob2 > p['p_threshold'])
            
            clean_ratio1 = pred1.sum() / len(pred1)
            clean_ratio2 = pred2.sum() / len(pred2)
            labeled_count1 = pred2.sum().item()
            labeled_count2 = pred1.sum().item()
            if epoch == p['warmup'] + 1 or epoch % 2 == 1:
                print(f'[DIAGNOSIS] Clean sample ratio - Net1: {clean_ratio1:.4f}, Net2: {clean_ratio2:.4f}')
                print(f'[DIAGNOSIS] Labeled sample count - Net1: {int(labeled_count1)}, Net2: {int(labeled_count2)}')
                fix_log.write('Epoch:%d Clean sample ratio - Net1: %.4f, Net2: %.4f\n'%(epoch, clean_ratio1, clean_ratio2))
                fix_log.write('Epoch:%d Labeled sample count - Net1: %d, Net2: %d\n'%(epoch, int(labeled_count1), int(labeled_count2)))
                fix_log.flush()

            print('[DM] Train Net1')
            meta_info['probability'] = prob2
            meta_info['pred'] = pred2
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            
            print(f'[DEBUG] labeled_trainloader: {len(labeled_trainloader)} batches, dataset size: {len(labeled_trainloader.dataset) if hasattr(labeled_trainloader, "dataset") else "N/A"}')
            print(f'[DEBUG] unlabeled_trainloader: {len(unlabeled_trainloader)} batches')
            sys.stdout.flush()
            
            if prototype_manager1 is None:
                print('\n*** CRITICAL ERROR: prototype_manager1 is None! ***\n')
                sys.stdout.flush()
            else:
                print(f'\n*** prototype_manager1 OK: update_count={prototype_manager1.update_count} ***\n')
                sys.stdout.flush()
            
            scanmix_train(p, epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device, prototype_manager=prototype_manager1, lambda_proto=1.0) # train net1  
            
            print('\n[DM] Train Net2')
            meta_info['probability'] = prob1
            meta_info['pred'] = pred1
            labeled_trainloader, unlabeled_trainloader = get_loader(p, 'train', meta_info)
            scanmix_train(p, epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, criterion_dm, args.lambda_u, device=device, prototype_manager=prototype_manager2, lambda_proto=1.0) # train net2       
            
            if not args.dividemix_only:
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = args.lr_sl    
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = args.lr_sl
                if epoch == p['warmup'] + 1:
                    print(f'[DIAGNOSIS] Switched to SCAN training, lr_sl = {args.lr_sl}')
                    fix_log.write('Epoch:%d Switched to SCAN training, lr_sl = %.8f\n'%(epoch, args.lr_sl))
                    fix_log.flush()  
                meta_info['predicted_labels'] = pl_2   
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net1')
                scanmix_scan(neighbor_dataloader, net1, criterion_sl, optimizer1, epoch, device)
                meta_info['predicted_labels'] = pl_1  
                neighbor_dataloader = get_loader(p, 'neighbors', meta_info)
                print('\n[SL] Train Net2')
                scanmix_scan(neighbor_dataloader, net2, criterion_sl, optimizer2, epoch, device)

        acc = scanmix_test(epoch,net1,net2,test_loader, device=device)
        print('\nEpoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush()
        
        if epoch >= p['warmup'] and prototype_manager1 is not None:
            if epoch == p['warmup'] + 1 or epoch % 2 == 1:
                print('\n' + '-'*60)
                print('[DIAGNOSIS] Prototype Statistics:')
                stats1 = prototype_manager1.get_prototype_stats()
                stats2 = prototype_manager2.get_prototype_stats()
                if stats1 is not None:
                    print(f"[Net1] Updates: {stats1['update_count']}, "
                          f"Avg change: {stats1['avg_inter_similarity']:.4f}, "
                          f"Max similarity: {stats1['max_inter_similarity']:.4f}")
                    fix_log.write('Epoch:%d [Net1] Updates: %d, Avg change: %.4f, Max similarity: %.4f\n'%(
                        epoch, stats1['update_count'], stats1['avg_inter_similarity'], stats1['max_inter_similarity']))
                if stats2 is not None:
                    print(f"[Net2] Updates: {stats2['update_count']}, "
                          f"Avg change: {stats2['avg_inter_similarity']:.4f}, "
                          f"Max similarity: {stats2['max_inter_similarity']:.4f}")
                    fix_log.write('Epoch:%d [Net2] Updates: %d, Avg change: %.4f, Max similarity: %.4f\n'%(
                        epoch, stats2['update_count'], stats2['avg_inter_similarity'], stats2['max_inter_similarity']))
                fix_log.flush()
                print('-'*60 + '\n')
            
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