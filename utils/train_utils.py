import torch
import torch.nn as nn
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class PrototypeManager:
    """
    原型管理类：实现双视角聚类（有监督+无监督）和原型融合
    支持动量队列更新机制
    """
    def __init__(self, num_classes, feature_dim, device, alpha=0.5, queue_size=128, dataset_size=50000):
        """
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            device: 设备
            alpha: 融合权重，alpha*有监督 + (1-alpha)*无监督
            queue_size: 每个类的队列大小（存储历史特征）
            dataset_size: 数据集大小，用于初始化全局特征缓存
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.alpha = alpha
        self.prototypes = None  # 融合后的原型 (num_classes, feature_dim)
        self.update_count = 0   # 更新次数
        self.prototype_history = []  # 原型变化历史
        
        # ========== 动量队列 ==========
        self.queue_size = queue_size
        # 为每个类初始化一个FIFO队列，存储特征
        self.feature_queues = [[] for _ in range(num_classes)]  # List of lists
        self.queue_initialized = False
        
        # ========== 全局特征缓存（用于拓扑门控） ==========
        self.dataset_size = dataset_size
        self.feature_bank = torch.zeros(dataset_size, feature_dim).to(device)  # 存储所有样本的最新特征
        self.feature_bank_initialized = torch.zeros(dataset_size, dtype=torch.bool).to(device)  # 标记哪些特征已初始化
        
    def compute_class_centers(self, features, labels, robust=False, top_alpha=0.5):
        """
        计算类中心（支持鲁棒估计）
        Args:
            features: (N, feature_dim) 特征
            labels: (N,) 标签
            robust: 是否使用鲁棒估计（trimmed mean）
            top_alpha: 使用最内部的top_alpha比例样本（0.5表示使用最内部50%）
        Returns:
            centers: (num_classes, feature_dim) 类中心
        """
        centers = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                
                if robust and mask.sum() > 2:
                    # 鲁棒中心：选择距离簇中心最近的top_alpha%样本
                    temp_center = class_features.mean(dim=0)
                    distances = torch.norm(class_features - temp_center.unsqueeze(0), dim=1)
                    k = max(1, int(len(distances) * top_alpha))
                    _, indices = torch.topk(distances, k, largest=False)
                    centers[c] = class_features[indices].mean(dim=0)
                else:
                    centers[c] = class_features.mean(dim=0)
        return centers
    
    def compute_cluster_confidence(self, features, labels, pred_probs=None):
        """
        计算每个簇的置信度分数
        Args:
            features: (N, feature_dim)
            labels: (N,) 簇标签
            pred_probs: (N, K) 预测概率（可选）
        Returns:
            scores: (num_classes,) 每个簇的置信度分数
        """
        scores = torch.zeros(self.num_classes).to(self.device)
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() < 2:
                scores[c] = 0.0
                continue
                
            class_features = features[mask]
            
            # 1. 簇内方差（越小越好）
            center = class_features.mean(dim=0)
            variance = torch.mean(torch.norm(class_features - center, dim=1))
            
            # 2. 簇大小（归一化）
            cluster_size = mask.sum().float() / len(labels)
            
            # 3. 如果有预测概率，计算平均置信度
            if pred_probs is not None:
                avg_confidence = pred_probs[mask].max(dim=1)[0].mean()
            else:
                avg_confidence = 0.5
            
            # 综合得分：置信度高、方差小、大小适中
            score = avg_confidence * cluster_size / (variance + 1e-6)
            scores[c] = score
        
        # 归一化到[0, 1]
        if scores.max() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return scores
    
    def hungarian_match(self, centers_a, centers_b, confidence_a=None, confidence_b=None, temperature=1.0):
        """
        使用改进的匈牙利算法对齐两组类中心
        
        改进：
        1. 同时使用余弦相似度和欧氏距离构建代价矩阵
        2. 温度增强+L2归一化
        3. 置信度加权
        
        Args:
            centers_a: (num_classes, feature_dim) 视角A的类中心
            centers_b: (num_classes, feature_dim) 视角B的类中心
            confidence_a: (num_classes,) 视角A的置信度（可选）
            confidence_b: (num_classes,) 视角B的置信度（可选）
            temperature: 温度参数，控制匹配的锐化程度
        Returns:
            mapping: (num_classes,) 视角B到视角A的映射
        """
        # 1. 计算余弦相似度矩阵
        centers_a_norm = F.normalize(centers_a, dim=1)
        centers_b_norm = F.normalize(centers_b, dim=1)
        cosine_similarity = torch.mm(centers_a_norm, centers_b_norm.t())  # (K, K)
        
        # 2. 计算欧氏距离矩阵
        dist_matrix = torch.cdist(centers_a, centers_b, p=2)  # (K, K)
        # 归一化到[0, 1]
        dist_matrix_norm = dist_matrix / (dist_matrix.max() + 1e-8)
        
        # 3. 组合相似度：余弦相似度 - 归一化欧氏距离
        combined_similarity = cosine_similarity - 0.5 * dist_matrix_norm  # (K, K)
        
        # 4. L2归一化相似度矩阵
        row_norms = torch.norm(combined_similarity, p=2, dim=1, keepdim=True)
        combined_similarity = combined_similarity / (row_norms + 1e-8)
        
        # 5. 温度增强（锐化相似度分布）
        combined_similarity = combined_similarity / temperature
        
        # 6. 置信度加权
        if confidence_a is not None and confidence_b is not None:
            conf_weight = torch.outer(confidence_a, confidence_b)  # (K, K)
            combined_similarity = combined_similarity * conf_weight
        
        # 7. 转换为代价矩阵（相似度越高代价越低）
        cost_matrix = -combined_similarity.cpu().numpy()
        
        # 8. 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 9. 创建映射
        mapping = np.zeros(self.num_classes, dtype=np.int64)
        mapping[col_ind] = row_ind
        
        # 10. 计算匹配质量
        matched_similarity = combined_similarity[row_ind, col_ind].mean().item()
        matched_cosine = cosine_similarity[row_ind, col_ind].mean().item()
        matched_dist = dist_matrix_norm[row_ind, col_ind].mean().item()
        print(f'  Hungarian matching quality (combined): {matched_similarity:.4f}')
        print(f'  - Cosine similarity: {matched_cosine:.4f}')
        print(f'  - Normalized distance: {matched_dist:.4f}')
        
        return mapping
    
    def initialize_prototypes_oracle(self, features, true_labels):
        """
        Oracle初始化：直接使用真实标签（Ground Truth）构建类原型
        
        这是理想情况下的原型初始化，用于测试完美类原型的性能上限
        
        Args:
            features: (N, feature_dim) 特征张量
            true_labels: (N,) 真实标签（未添加噪声）
        """
        print('\n' + '='*60)
        print('=== Oracle Prototype Initialization (Ground Truth) ===')
        print('='*60)
        
        features = features.to(self.device)
        true_labels = true_labels.to(self.device)
        
        # 直接使用真实标签计算类中心
        print('Computing class centers using GROUND TRUTH labels...')
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        
        for c in range(self.num_classes):
            mask = (true_labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                self.prototypes[c] = class_features.mean(dim=0)
                print(f'  Class {c}: {mask.sum().item()} samples')
            else:
                print(f'  Class {c}: WARNING - No samples found!')
        
        # 归一化
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        # 保存信息
        self.prototype_info = {
            'initialization': 'oracle',
            'num_classes': self.num_classes,
            'use_ground_truth': True
        }
        
        print(f'Oracle prototypes initialized. Shape: {self.prototypes.shape}')
        print('='*60 + '\n')
    
    def initialize_prototypes(self, features, pred_labels, simclr_clusters, pred_probs=None, use_supervised_only=False):
        """
        初始化原型：使用改进的匈牙利算法对齐有监督和无监督聚类中心
        
        改进策略：
        1. 鲁棒中心估计（trimmed mean）
        2. 计算簇置信度
        3. 温度增强的匈牙利匹配（带置信度加权）
        4. 自适应融合权重
        5. 迭代优化（可选）
        
        Args:
            features: (N, feature_dim) 特征
            pred_labels: (N,) 预测标签或真实标签
            simclr_clusters: (N,) SimCLR聚类标签
            pred_probs: (N, num_classes) 预测概率（可选）
            use_supervised_only: 是否只使用有监督聚类（alpha=1.0，忽略无监督）
        """
        if use_supervised_only:
            print('\n=== Initializing Prototypes with Supervised-Only (God View) ===')
        else:
            print('\n=== Initializing Prototypes with Enhanced Hungarian Matching ===')
        
        device = self.device
        num_classes = self.num_classes
        
        features = features.to(device)
        pred_labels = pred_labels.to(device)
        simclr_clusters = simclr_clusters.to(device)
        if pred_probs is not None:
            pred_probs = pred_probs.to(device)
        
        # 1. 计算鲁棒的有监督中心（使用trimmed mean）
        print('Computing robust supervised centers...')
        supervised_centers = self.compute_class_centers(features, pred_labels, robust=True, top_alpha=0.7)
        
        # 如果只使用有监督聚类，直接使用有监督中心
        if use_supervised_only:
            self.prototypes = F.normalize(supervised_centers, dim=1)
            print(f'  Using supervised-only prototypes (alpha=1.0)')
            print('  Skipping unsupervised clustering and Hungarian matching')
            # 保存信息
            self.prototype_info = {
                'initialization': 'supervised_only',
                'num_classes': num_classes,
                'supervised_centers': supervised_centers.cpu()
            }
            return
        
        # 2. 计算鲁棒的无监督中心
        print('Computing robust unsupervised centers...')
        num_clusters = num_classes
        unsupervised_centers = torch.zeros(num_clusters, self.feature_dim).to(device)
        
        for j in range(num_clusters):
            mask = (simclr_clusters == j)
            if mask.sum() > 0:
                cluster_features = features[mask]
                # 鲁棒估计
                if mask.sum() > 2:
                    temp_center = cluster_features.mean(dim=0)
                    distances = torch.norm(cluster_features - temp_center, dim=1)
                    k = max(1, int(len(distances) * 0.7))
                    _, indices = torch.topk(distances, k, largest=False)
                    unsupervised_centers[j] = cluster_features[indices].mean(dim=0)
                else:
                    unsupervised_centers[j] = cluster_features.mean(dim=0)
        
        # 3. 计算簇置信度
        print('Computing cluster confidence scores...')
        supervised_confidence = self.compute_cluster_confidence(features, pred_labels, pred_probs)
        unsupervised_confidence = self.compute_cluster_confidence(features, simclr_clusters, None)
        
        print(f'  Supervised confidence: mean={supervised_confidence.mean():.3f}, std={supervised_confidence.std():.3f}')
        print(f'  Unsupervised confidence: mean={unsupervised_confidence.mean():.3f}, std={unsupervised_confidence.std():.3f}')
        
        # 4. 使用改进的匈牙利算法对齐（温度增强+置信度加权）
        print('Aligning centers with enhanced Hungarian algorithm...')
        temperature = 0.5  # 锐化相似度分布
        mapping = self.hungarian_match(
            supervised_centers, 
            unsupervised_centers,
            confidence_a=supervised_confidence,
            confidence_b=unsupervised_confidence,
            temperature=temperature
        )
        
        # 应用映射
        aligned_unsupervised_centers = unsupervised_centers[mapping]
        aligned_unsupervised_confidence = unsupervised_confidence[mapping]
        
        # 5. 自适应融合权重（基于置信度）
        print('Computing adaptive fusion weights...')
        # 对于每个类，根据两个视角的相对置信度调整融合权重
        adaptive_alpha = torch.zeros(num_classes).to(device)
        for c in range(num_classes):
            sup_conf = supervised_confidence[c]
            unsup_conf = aligned_unsupervised_confidence[c]
            # 归一化权重：置信度高的视角权重更大
            total_conf = sup_conf + unsup_conf + 1e-6
            adaptive_alpha[c] = sup_conf / total_conf
        
        # 限制在合理范围内 [0.3, 0.7]
        adaptive_alpha = torch.clamp(adaptive_alpha, 0.3, 0.7)
        print(f'  Adaptive alpha: mean={adaptive_alpha.mean():.3f}, std={adaptive_alpha.std():.3f}')
        
        # 6. 融合两个视角的中心（使用自适应权重）
        self.prototypes = torch.zeros_like(supervised_centers)
        for c in range(num_classes):
            alpha_c = adaptive_alpha[c]
            self.prototypes[c] = alpha_c * supervised_centers[c] + (1 - alpha_c) * aligned_unsupervised_centers[c]
        
        # 归一化
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        # 7. 可选：迭代优化（重新分配样本并重新计算中心）
        print('Performing iterative refinement...')
        self.prototypes = self._iterative_refinement(features, pred_labels, simclr_clusters, self.prototypes)
        
        # 保存信息
        self.cluster_mapping = mapping
        self.adaptive_alpha = adaptive_alpha
        self.supervised_confidence = supervised_confidence
        self.unsupervised_confidence = aligned_unsupervised_confidence
        
        print(f'Prototypes initialized. Shape: {self.prototypes.shape}')
        print('='*60)
    
    def _iterative_refinement(self, features, pred_labels, simclr_clusters, initial_prototypes, num_iterations=2):
        """
        迭代优化原型
        
        策略：重新分配样本到最近的原型，过滤异常值，重新计算中心
        
        Args:
            features: (N, feature_dim)
            pred_labels: (N,)
            simclr_clusters: (N,)
            initial_prototypes: (K, feature_dim)
            num_iterations: 迭代次数
        Returns:
            refined_prototypes: (K, feature_dim)
        """
        prototypes = initial_prototypes.clone()
        
        for iter_idx in range(num_iterations):
            # 计算每个样本到各原型的距离
            features_norm = F.normalize(features, dim=1)
            prototypes_norm = F.normalize(prototypes, dim=1)
            similarities = torch.mm(features_norm, prototypes_norm.t())  # (N, K)
            
            # 为每个样本分配最近的原型
            assigned_labels = similarities.argmax(dim=1)
            
            # 对于每个类，只保留距离原型最近的样本（过滤异常值）
            new_prototypes = torch.zeros_like(prototypes)
            for c in range(self.num_classes):
                # 找出被分配到类c的样本
                mask = (assigned_labels == c)
                if mask.sum() < 2:
                    new_prototypes[c] = prototypes[c]  # 保持不变
                    continue
                
                class_features = features[mask]
                class_similarities = similarities[mask, c]
                
                # 只保留相似度高于中位数的样本
                threshold = class_similarities.median()
                high_quality_mask = class_similarities >= threshold
                
                if high_quality_mask.sum() > 0:
                    new_prototypes[c] = class_features[high_quality_mask].mean(dim=0)
                else:
                    new_prototypes[c] = prototypes[c]
            
            prototypes = F.normalize(new_prototypes, dim=1)
            print(f'  Iteration {iter_idx+1}/{num_iterations} completed')
        
        return prototypes
    
    def update_prototypes(self, features, labels, momentum=0.9, indices=None, neighbor_indices=None, 
                         model=None, data_loader=None, consistency_threshold=0.6, confidence_scores=None,
                         topology_threshold=0.7, use_oracle=False):
        """
        基于动量队列的原型更新（拓扑一致性门控版）
        
        流程：
        1. 拓扑一致性门控：只有当样本与原型的相似度 与 样本邻居均值与原型的相似度一致时才更新
        2. 置信度过滤入队
        3. 队列内离群点移除（15%）
        4. EMA更新原型
        
        Oracle模式（use_oracle=True）：
        - 跳过拓扑一致性门控
        - 跳过置信度过滤
        - 直接使用所有干净样本的特征更新原型
        
        Args:
            topology_threshold: 拓扑一致性阈值（默认0.7，较低值更宽松）
            labels: (N,) 标签（Oracle模式下传入真实标签，标准模式下传入预测标签）
            use_oracle: 是否使用Oracle模式（跳过过滤，使用所有干净样本）
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not initialized!")
        
        # 首次更新时打印配置
        if self.update_count == 0:
            print(f'\n  [Prototype Update Config]')
            if use_oracle:
                print(f'    *** ORACLE MODE: Using ALL clean samples (no filtering) ***')
            else:
                print(f'    Topology threshold: {topology_threshold:.2f}')
                print(f'    Queue size: {self.queue_size}')
            print(f'    Momentum: {momentum:.2f}')
            if not use_oracle:
                print(f'    Outlier removal ratio: 15%')
        
        old_prototypes = self.prototypes.clone()
        
        features_np = features.detach()
        labels_np = labels.detach()
        
        # ========== 更新全局特征缓存 ==========
        if indices is not None:
            for idx, feat in zip(indices, features_np):
                idx_val = idx.item() if hasattr(idx, 'item') else idx
                if idx_val < self.dataset_size:
                    self.feature_bank[idx_val] = feat
                    self.feature_bank_initialized[idx_val] = True
        
        enqueue_stats = {'accepted': 0, 'rejected_conf': 0, 'rejected_topo': 0}
        
        # ========== Oracle模式：直接使用所有干净样本 ==========
        if use_oracle:
            # 直接计算每个类的特征均值（无过滤）
            direct_class_centers = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
            class_counts = torch.zeros(self.num_classes).to(self.device)
            
            for c in range(self.num_classes):
                mask = (labels_np == c)
                if mask.sum() > 0:
                    class_features = features_np[mask]
                    direct_class_centers[c] = class_features.mean(dim=0)
                    class_counts[c] = mask.sum()
                    enqueue_stats['accepted'] += mask.sum().item()
                else:
                    # 该类无样本，保持当前原型
                    direct_class_centers[c] = self.prototypes[c]
            
            # 归一化
            direct_class_centers = F.normalize(direct_class_centers, dim=1)
            
            # EMA更新（使用真实标签直接计算的类中心）
            new_prototypes = momentum * self.prototypes + (1 - momentum) * direct_class_centers
            
            # Safety check
            norms = torch.norm(new_prototypes, dim=1, keepdim=True)
            safe_mask = (norms > 1e-6)
            
            if safe_mask.all():
                self.prototypes = new_prototypes
            else:
                print(f"  [Warning] {(~safe_mask).sum().item()} prototypes have near-zero norm.")
                self.prototypes = torch.where(safe_mask, new_prototypes, self.prototypes)
            
            self.prototypes = F.normalize(self.prototypes, dim=1)
            
            # 计算变化
            cosine_change = 1 - torch.sum(old_prototypes * self.prototypes, dim=1)
            avg_change = cosine_change.mean().item()
            max_change = cosine_change.max().item()
            self.update_count += 1
            
            # 打印统计（每100次更新打印一次，避免过多输出）
            if self.update_count % 100 == 0:
                total_samples = enqueue_stats['accepted']
                print(f'  [Oracle Update #{self.update_count}] {total_samples} samples, change: avg={avg_change:.4f}, max={max_change:.4f}')
            
            return avg_change, max_change
        
        for c in range(self.num_classes):
            mask = (labels_np == c)
            if mask.sum() > 0:
                class_features = features_np[mask]
                class_indices_in_batch = torch.where(mask)[0]
                
                # 逐个样本判断是否入队
                for i, feat in enumerate(class_features):
                    feat_norm = F.normalize(feat.unsqueeze(0), dim=1)
                    proto_norm = F.normalize(self.prototypes[c].unsqueeze(0), dim=1)
                    conf_new = torch.cosine_similarity(feat_norm, proto_norm).item()
                    
                    # ========== 拓扑一致性门控检查（使用真实邻居特征） ==========
                    topology_pass = True
                    if neighbor_indices is not None and indices is not None and len(self.feature_queues[c]) >= 3:
                        # 获取当前样本在原始数据集中的索引
                        batch_idx_local = class_indices_in_batch[i].item()
                        if batch_idx_local < len(indices):
                            original_idx = indices[batch_idx_local].item()
                            
                            if original_idx < len(neighbor_indices):
                                # 获取该样本的K个邻居索引
                                k_neighbors = neighbor_indices[original_idx][:10]  # 取前10个邻居
                                
                                # 从全局特征缓存中提取邻居的实际特征
                                valid_neighbors = []
                                for neighbor_idx in k_neighbors:
                                    neighbor_idx_val = neighbor_idx.item() if hasattr(neighbor_idx, 'item') else neighbor_idx
                                    if neighbor_idx_val < self.dataset_size and self.feature_bank_initialized[neighbor_idx_val]:
                                        valid_neighbors.append(self.feature_bank[neighbor_idx_val])
                                
                                # 如果有足够的有效邻居（至少3个），进行拓扑检查
                                if len(valid_neighbors) >= 3:
                                    neighbor_features = torch.stack(valid_neighbors, dim=0)
                                    neighbors_mean = neighbor_features.mean(dim=0, keepdim=True)
                                    neighbors_mean_norm = F.normalize(neighbors_mean, dim=1)
                                    
                                    # 计算对齐度: S_align = CosineSimilarity(P_y, Neighbors_Mean(x))
                                    s_align = torch.cosine_similarity(proto_norm, neighbors_mean_norm).item()
                                    
                                    # 拓扑一致性判断
                                    if s_align < topology_threshold:
                                        topology_pass = False
                                        enqueue_stats['rejected_topo'] += 1
                    
                    if not topology_pass:
                        continue  # 拓扑一致性检查失败，跳过该样本
                    
                    # 置信度过滤入队
                    if len(self.feature_queues[c]) < self.queue_size:
                        self.feature_queues[c].append(feat.clone())
                        enqueue_stats['accepted'] += 1
                    else:
                        # 队列已满，找出队列中置信度最低的样本
                        queue_features = torch.stack(self.feature_queues[c], dim=0)
                        queue_features_norm = F.normalize(queue_features, dim=1)
                        proto_norm_expanded = proto_norm.expand(len(queue_features), -1)
                        conf_queue = torch.cosine_similarity(queue_features_norm, proto_norm_expanded)
                        
                        min_conf_idx = conf_queue.argmin().item()
                        min_conf_value = conf_queue[min_conf_idx].item()
                        
                        # 如果新样本置信度高于队列中最差的，则替换
                        if conf_new > min_conf_value:
                            self.feature_queues[c].pop(min_conf_idx)
                            self.feature_queues[c].append(feat.clone())
                            enqueue_stats['accepted'] += 1
                        else:
                            enqueue_stats['rejected_conf'] += 1
        
        # ========== 检查队列填充状态 ==========
        # 统计每个类的队列填充情况
        queue_sizes = [len(q) for q in self.feature_queues]
        min_queue_size = min(queue_sizes) if queue_sizes else 0
        avg_queue_size = sum(queue_sizes) / len(queue_sizes) if queue_sizes else 0
        
        # 计算队列填充率
        fill_ratio = min_queue_size / self.queue_size if self.queue_size > 0 else 0
        
        # 自适应填充阈值：alpha越小（低噪声），要求越低
        # alpha=0.9: 30%填充即可（低噪声样本质量好）
        # alpha=0.99: 50%填充（高噪声需要更多样本）
        if self.alpha <= 0.9:
            min_fill_threshold = 0.3  # 低噪声：30%
        elif self.alpha <= 0.95:
            min_fill_threshold = 0.4  # 中等噪声：40%
        else:
            min_fill_threshold = 0.5  # 高噪声：50%
        
        if fill_ratio < min_fill_threshold:
            if self.update_count == 0 or (self.update_count < 10):
                print(f'  [Queue Filling] Min queue: {min_queue_size}/{self.queue_size} ({fill_ratio:.1%}), '
                      f'Avg queue: {avg_queue_size:.1f}, Skipping prototype update until {min_fill_threshold:.0%} filled')
            return 0.0, 0.0  # 返回零变化
        
        # 队列已充分填充，开始进行原型更新
        if self.update_count == 0:
            print(f'  [Queue Ready] All queues filled ≥{min_fill_threshold:.0%} (alpha={self.alpha:.2f}), Starting prototype updates')
            print(f'  Queue stats: min={min_queue_size}, max={max(queue_sizes)}, avg={avg_queue_size:.1f}')
        
        # 队列内二次清洗：移除离群点
        queue_centers = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        outlier_removal_ratio = 0.15  # 移除15%最远的样本
        
        for c in range(self.num_classes):
            if len(self.feature_queues[c]) > 2:  # 至少需要3个样本才进行清洗
                queue_features = torch.stack(self.feature_queues[c], dim=0)
                
                # 计算队列均值中心
                temp_center = queue_features.mean(dim=0)
                
                # 计算每个样本到中心的距离
                distances = torch.norm(queue_features - temp_center.unsqueeze(0), dim=1)
                
                # 计算保留的样本数量
                k_keep = max(1, int(len(distances) * (1 - outlier_removal_ratio)))
                
                # 选择距离最近的k_keep个样本
                _, keep_indices = torch.topk(distances, k_keep, largest=False)
                
                # 使用清洗后的样本计算队列中心
                queue_centers[c] = queue_features[keep_indices].mean(dim=0)
                
            elif len(self.feature_queues[c]) > 0:
                # 样本太少，直接使用所有样本
                queue_features = torch.stack(self.feature_queues[c], dim=0)
                queue_centers[c] = queue_features.mean(dim=0)
            else:
                # 队列为空，使用当前原型
                queue_centers[c] = self.prototypes[c]
        
        # 归一化队列中心
        queue_centers = F.normalize(queue_centers, dim=1)
        
        # ========== 3. 球面EMA更新原型（在归一化空间中插值） ==========
        # 关键改进：在归一化空间中进行球面插值，避免归一化放大方向变化
        # 使用Slerp（球面线性插值）的近似方法
        
        # 计算旧原型和队列中心之间的余弦相似度
        cos_theta = torch.sum(self.prototypes * queue_centers, dim=1, keepdim=True)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # 对于接近的向量（cos > 0.99），使用线性插值
        # 对于差异大的向量，限制更新幅度
        similar_mask = (cos_theta.squeeze() > 0.99)
        
        # 标准EMA更新
        new_prototypes = momentum * self.prototypes + (1 - momentum) * queue_centers
        
        # 对于差异大的原型，额外减小更新步长（再乘以0.5）
        if not similar_mask.all():
            dissimilar_indices = ~similar_mask
            extra_damping = 0.5  # 额外阻尼系数
            new_prototypes[dissimilar_indices] = (
                (momentum + (1-momentum)*extra_damping) * self.prototypes[dissimilar_indices] + 
                (1-momentum)*(1-extra_damping) * queue_centers[dissimilar_indices]
            )
        
        # Safety check: Avoid zero vectors before normalization
        norms = torch.norm(new_prototypes, dim=1, keepdim=True)
        safe_mask = (norms > 1e-6)
        
        if safe_mask.all():
             self.prototypes = new_prototypes
        else:
             print(f"  [Warning] {(~safe_mask).sum().item()} prototypes have near-zero norm. Skipping update for them.")
             self.prototypes = torch.where(safe_mask, new_prototypes, self.prototypes)

        self.prototypes = F.normalize(self.prototypes, dim=1)
            
        cosine_change = 1 - torch.sum(old_prototypes * self.prototypes, dim=1)
        avg_change = cosine_change.mean().item()
        max_change = cosine_change.max().item()
        self.update_count += 1
        
        # ========== 详细的更新统计（注释掉避免输出过多） ==========
        # total_samples = enqueue_stats['accepted'] + enqueue_stats['rejected_conf'] + enqueue_stats['rejected_topo']
        # if total_samples > 0:
        #     accept_rate = enqueue_stats['accepted'] / total_samples * 100
        #     print(f'  [Prototype Update #{self.update_count}]')
        #     print(f'    Samples: {enqueue_stats["accepted"]}/{total_samples} accepted ({accept_rate:.1f}%)')
        #     print(f'    Rejected: topo={enqueue_stats["rejected_topo"]}, conf={enqueue_stats["rejected_conf"]}')
        #     print(f'    Prototype change: avg={avg_change:.4f}, max={max_change:.4f}')
        #     
        #     # 显示每个类的队列状态
        #     queue_sizes = [len(q) for q in self.feature_queues]
        #     print(f'    Queue sizes: min={min(queue_sizes)}, max={max(queue_sizes)}, avg={sum(queue_sizes)/len(queue_sizes):.1f}')
        
        # 打印队列状态（保留每100次的周期性打印）
        if self.update_count % 100 == 0:
            queue_sizes = [len(q) for q in self.feature_queues]
            avg_queue_size = sum(queue_sizes) / len(queue_sizes)
            total_samples = enqueue_stats['accepted'] + enqueue_stats['rejected_conf'] + enqueue_stats['rejected_topo']
            accept_rate = enqueue_stats['accepted'] / total_samples if total_samples > 0 else 0
            topo_reject_rate = enqueue_stats['rejected_topo'] / total_samples if total_samples > 0 else 0
            print(f'  [Prototype Update #{self.update_count}] '
                  f'Avg change: {avg_change:.6f}, Max change: {max_change:.6f}, '
                  f'Avg queue: {avg_queue_size:.1f}, '
                  f'Accept: {accept_rate:.1%}, TopoReject: {topo_reject_rate:.1%}')
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
    提取所有样本的特征和预测概率
    Args:
        net: 网络模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        all_features: (N, feature_dim) 特征
        all_targets: (N,) 真实标签
        all_predictions: (N,) 预测标签
        all_probs: (N, num_classes) 预测概率分布
    """
    net.eval()
    all_features = []
    all_targets = []
    all_predictions = []
    all_probs = []
    
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
        probs = torch.softmax(outputs, dim=1)
        predictions = probs.argmax(dim=1)
        
        all_features.append(features.cpu())
        all_targets.append(targets)
        all_predictions.append(predictions.cpu())
        all_probs.append(probs.cpu())
        
        if (batch_idx + 1) % 50 == 0:
            print(f'  Processed {batch_idx + 1}/{len(dataloader)} batches')
    
    all_features = torch.cat(all_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    print(f'Feature extraction completed: {all_features.shape}')
    return all_features, all_targets, all_predictions, all_probs

def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)


def scanmix_train(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion,lambda_u,device,
                  prototype_manager=None,lambda_proto=1.0,neighbor_indices_static=None,neighbor_indices_dynamic=None,neighbor_fusion_alpha=0.8,use_oracle=False,true_labels_map=None):
    """
    ScanMix训练函数（支持Oracle模式和双源邻居融合）
    
    Args:
        use_oracle: 是否使用Oracle模式
        true_labels_map: 真实标签映射字典 {index: true_label}
        neighbor_indices_static: 静态邻居库（SimCLR，不变）
        neighbor_indices_dynamic: 动态邻居库（每50轮更新）
        neighbor_fusion_alpha: 静态邻居权重（α），动态邻居权重为(1-α)
    """
    return _scanmix_train_impl(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,
                               criterion,lambda_u,device,prototype_manager,lambda_proto,neighbor_indices_static,neighbor_indices_dynamic,neighbor_fusion_alpha,
                               use_oracle,true_labels_map)


def _scanmix_train_impl(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion,lambda_u,device,
                  prototype_manager=None,lambda_proto=1.0,neighbor_indices_static=None,neighbor_indices_dynamic=None,neighbor_fusion_alpha=0.8,use_oracle=False,true_labels_map=None):
    """
    ScanMix训练函数（内部实现，支持双源邻居融合）
    """
    net.train()
    net2.eval()
    
    labeled_losses = AverageMeter('Labelled Loss', ':.4e')
    unlabeled_losses = AverageMeter('Unlabelled Loss', ':.4e')
    
    update_counter = [0]
    
    if prototype_manager is not None:
        proto_losses = AverageMeter('Prototype Loss', ':.4e')
        progress = ProgressMeter(len(labeled_trainloader),
            [labeled_losses, unlabeled_losses, proto_losses],
            prefix="Epoch: [{}]".format(epoch))
        print(f'\n*** PROTOTYPE MANAGER ACTIVE for Epoch {epoch} ***')
        sys.stdout.flush()
    else:
        progress = ProgressMeter(len(labeled_trainloader),
            [labeled_losses, unlabeled_losses],
            prefix="Epoch: [{}]".format(epoch))
        print(f'\n*** WARNING: NO PROTOTYPE MANAGER for Epoch {epoch} ***')
        sys.stdout.flush()
    
    if prototype_manager is not None:
        proto_criterion = nn.CrossEntropyLoss()
        temperature = 0.1

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//p['batch_size'])+1
    
    print(f'\n@@@ ABOUT TO START LOOP: {len(labeled_trainloader)} batches, prototype_manager={prototype_manager is not None} @@@')
    sys.stdout.flush()
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, indices_x) in enumerate(labeled_trainloader):
        if batch_idx == 0:
            if prototype_manager is not None:
                print(f'\n*** batch 0: prototype_manager is NOT None ***')
            else:
                print(f'\n*** batch 0: prototype_manager IS None! ***')
            sys.stdout.flush()
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot (scanmix_train with indices_x)
        labels_x = torch.zeros(batch_size, p['num_classes']).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u, forward_pass='dm')
            outputs_u12 = net(inputs_u2, forward_pass='dm')
            outputs_u21 = net2(inputs_u, forward_pass='dm')
            outputs_u22 = net2(inputs_u2, forward_pass='dm')            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/p['T']) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x, forward_pass='dm')
            outputs_x2 = net(inputs_x2, forward_pass='dm')            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/p['T']) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(p['alpha'], p['alpha'])        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input, forward_pass='dm')
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:],lambda_u, epoch+batch_idx/num_iter, p['warmup'])
        
        # ========== 特征提取（用于原型更新，无论lambda_proto是否为0） ==========
        features_x_clean = None
        target_labels_x_clean = None
        
        if prototype_manager is not None and epoch > p['warmup'] and prototype_manager.prototypes is not None:
            # 提取特征（用于原型更新或原型损失计算）
            features_x = net(inputs_x, forward_pass='backbone')
            features_x2 = net(inputs_x2, forward_pass='backbone')
            features_x_clean = torch.cat([features_x, features_x2], dim=0)
            
            # 获取标签（Oracle模式使用真实标签，标准模式使用伪标签）
            if use_oracle and true_labels_map is not None:
                true_labels_x = torch.tensor([true_labels_map[idx.item()] for idx in indices_x], device=device)
                target_labels_x_clean = torch.cat([true_labels_x, true_labels_x], dim=0)
            else:
                target_labels_x = targets_x.argmax(dim=1)
                target_labels_x_clean = torch.cat([target_labels_x, target_labels_x], dim=0)
        
        # ========== 原型对比损失计算（仅当lambda_proto>0时） ==========
        L_proto = 0
        L_neighbor = 0
        proto_accuracy = 0
        
        if prototype_manager is not None and epoch > p['warmup'] and prototype_manager.prototypes is not None and lambda_proto > 0:
            if batch_idx == 0:
                print(f'\n*** COMPUTING PROTOTYPE LOSS at batch 0, epoch {epoch} ***')
                if use_oracle:
                    print(f'  [Oracle Mode] Using GROUND TRUTH labels for prototype contrastive loss')
                sys.stdout.flush()
            try:
                # 计算原型对比损失（参与反向传播）
                features_norm = F.normalize(features_x_clean, dim=1)
                prototypes_norm = F.normalize(prototype_manager.prototypes, dim=1)
                similarities = torch.mm(features_norm, prototypes_norm.t()) / temperature
                L_proto = proto_criterion(similarities, target_labels_x_clean)
                
                # ========== 邻居一致性损失改进：Weak-to-Strong + 锐化KL散度 ==========
                # 1. 核心逻辑改进：从"对称"转向"非对称教师制"
                #    - 弱增强视图作为"教师"（锚点），经过detach停止梯度
                #    - 强增强视图作为"学生"，被教师指导
                # 2. 数学度量改进：从MSE转向锐化KL散度
                #    - 对弱增强分配概率进行温度锐化（T=0.5），产生"硬决策"
                #    - 使用KL散度衡量分布差异，具有更强的拉动力
                
                features_x_norm = F.normalize(features_x, dim=1)
                features_x2_norm = F.normalize(features_x2, dim=1)
                prototypes_norm = F.normalize(prototype_manager.prototypes, dim=1)
                
                # 弱增强视图（inputs_x）作为"教师"
                proto_assign_weak = F.softmax(torch.mm(features_x_norm, prototypes_norm.t()) / temperature, dim=1)
                
                # 温度锐化处理（T=0.5）：q̃_w = Normalize(q_w^(1/T))
                T_sharpen = 0.5
                proto_assign_weak_sharpened = torch.pow(proto_assign_weak, 1.0 / T_sharpen)
                proto_assign_weak_sharpened = proto_assign_weak_sharpened / proto_assign_weak_sharpened.sum(dim=1, keepdim=True)
                proto_assign_weak_teacher = proto_assign_weak_sharpened.detach()  # 停止梯度
                
                # 强增强视图（inputs_x2）作为"学生"
                proto_assign_strong = F.softmax(torch.mm(features_x2_norm, prototypes_norm.t()) / temperature, dim=1)
                
                # KL散度损失：D_KL(teacher || student)
                # 注意：F.kl_div的input应该是log概率，target是概率
                L_neighbor = F.kl_div(proto_assign_strong.log(), proto_assign_weak_teacher, reduction='batchmean')
                
                if batch_idx == 0:
                    print(f'  Neighbor consistency loss: {L_neighbor.item():.4f}')
                
                proto_losses.update(L_proto.item())
                
            except Exception as e:
                print(f'\n*** EXCEPTION in prototype code at batch {batch_idx}: {e} ***')
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
        
        # regularization
        prior = torch.ones(p['num_classes'])/p['num_classes']
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + penalty
        if prototype_manager is not None and lambda_proto > 0:
            if batch_idx == 0:
                print(f'  [Batch {batch_idx}] Applying prototype loss: lambda_proto={lambda_proto:.4f}')
                l_proto_val = L_proto.item() if isinstance(L_proto, torch.Tensor) else L_proto
                l_neighbor_val = L_neighbor.item() if isinstance(L_neighbor, torch.Tensor) else L_neighbor
                print(f'    L_proto={l_proto_val:.4f}, L_neighbor={l_neighbor_val:.4f}')
            
            # 邻居一致性损失权重为原型损失的一半
            lambda_neighbor = 0.5 * lambda_proto
            loss = loss + lambda_proto * L_proto + lambda_neighbor * L_neighbor
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'  [Batch {batch_idx}] Model parameters updated (optimizer.step completed)')

        # ⭐ 原型更新（batch级更新）
        if prototype_manager is not None and epoch > p['warmup']:
            with torch.no_grad():
                if features_x_clean is not None and features_x_clean.size(0) > 0:
                    update_counter[0] += 1
                    if batch_idx % 100 == 0:
                        print(f'  [Batch {batch_idx}] Updating prototypes...')
                    
                    # 扩展indices匹配features (2N)
                    indices_expanded = torch.cat([indices_x, indices_x], dim=0)
                    
                    # ========== Oracle模式：使用真实标签替换伪标签 ==========
                    if use_oracle and true_labels_map is not None:
                        # 用真实标签替换伪标签
                        true_labels_batch = torch.cat([
                            torch.tensor([true_labels_map[idx.item()] for idx in indices_x], device=device),
                            torch.tensor([true_labels_map[idx.item()] for idx in indices_x], device=device)
                        ], dim=0)
                        
                        if batch_idx == 0:
                            print(f'  [Oracle Mode] Using GROUND TRUTH labels for prototype update')
                            print(f'  [Oracle Mode] Using ALL clean samples (no filtering)')
                        
                        # Oracle模式：传入use_oracle=True跳过过滤
                        prototype_manager.update_prototypes(
                            features=features_x_clean.detach(),
                            labels=true_labels_batch,  # 使用真实标签
                            indices=indices_expanded,
                            neighbor_indices=neighbor_indices_static,  # 使用静态邻居用于拓扑门控
                            momentum=prototype_manager.alpha,  # 使用配置的alpha值
                            use_oracle=True  # 跳过拓扑门控和置信度过滤
                        )
                    else:
                        # 标准模式：使用伪标签，降低拓扑阈值以允许更多样本更新原型
                        prototype_manager.update_prototypes(
                            features=features_x_clean.detach(),
                            labels=target_labels_x_clean,  # 使用伪标签
                            indices=indices_expanded,
                            neighbor_indices=neighbor_indices_static,  # 使用静态邻居用于拓扑门控
                            momentum=prototype_manager.alpha,  # 使用配置的alpha值
                            topology_threshold=0.3,  # 降低阈值到0.3，在高噪声场景下允许更多样本通过
                            use_oracle=False  # 使用标准的拓扑门控和置信度过滤
                        )
                    
                    if update_counter[0] == 1:
                        print(f'\n*** FIRST PROTOTYPE UPDATE in Epoch {epoch}! ***')
                        sys.stdout.flush()
                elif batch_idx % 100 == 0:
                    print(f'  [Batch {batch_idx}] Skipping prototype update (no features extracted)')
        elif prototype_manager is not None and epoch <= p['warmup'] and batch_idx == 0:
            print(f'  [Warmup] Skipping prototype operations until epoch {p["warmup"]+1}')

        labeled_losses.update(Lx.item())
        unlabeled_losses.update(Lu.item())

        if batch_idx % 25 == 0:
            progress.display(batch_idx)
    
    if prototype_manager is not None:
        print(f'\n*** Epoch {epoch} completed: {update_counter[0]} prototype updates made ***')
        print(f'*** Prototype manager update_count: {prototype_manager.update_count} ***\n')
        sys.stdout.flush()

def scanmix_big_train(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion,lambda_u,device):
    net.train()
    net2.eval() #fix one network and train the other
    
    final_loss = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(labeled_trainloader),
        [final_loss],
        prefix="Epoch: [{}]".format(epoch))

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//p['batch_size'])+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, p['num_classes']).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u, forward_pass='dm')
            outputs_u12 = net(inputs_u2, forward_pass='dm')
            outputs_u21 = net2(inputs_u, forward_pass='dm')
            outputs_u22 = net2(inputs_u2, forward_pass='dm')            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/p['T']) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x, forward_pass='dm')
            outputs_x2 = net(inputs_x2, forward_pass='dm')            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/p['T']) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(p['alpha'], p['alpha'])        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input, forward_pass='dm')
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(p['num_classes'])/p['num_classes']
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss.update(loss.item())

        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_warmup(epoch,net,optimizer,dataloader,criterion, conf_penalty, noise_mode, device):
    net.train()
    losses = AverageMeter('CE-Loss', ':.4e')
    progress = ProgressMeter(len(dataloader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):    
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        with torch.no_grad():
            input_features = net(inputs, forward_pass='backbone')
        outputs = net(input_features, forward_pass='dm_head')      
        loss = criterion(outputs, labels)  
        if noise_mode=='asym' or 'semantic' in noise_mode:  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        losses.update(L.item()) 
        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_big_warmup(p,epoch,net,optimizer,dataloader,criterion, conf_penalty, noise_mode, device):
    net.train()
    losses = AverageMeter('CE-Loss', ':.4e')
    progress = ProgressMeter(len(dataloader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, batch in enumerate(dataloader): 
        inputs, labels = batch['image'].to(device), batch['target'].to(device)     
        optimizer.zero_grad()
        with torch.no_grad():
            input_features = net(inputs, forward_pass='backbone')
        outputs = net(input_features, forward_pass='dm_head')      
        loss = criterion(outputs, labels)  
        if p['dataset'] in  ['webvision', 'mini_imagenet_red', 'mini_imagenet32_red']:   
            L = loss
        else:
            raise NotImplementedError()
        L.backward()  
        optimizer.step() 
        losses.update(L.item()) 
        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_eval_train(args,model,all_loss,epoch,eval_loader,criterion,device):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs, forward_pass='dm')
            _, predicted = torch.max(outputs, 1) 
            loss = criterion(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                pl[index[b]]  = predicted[b]        
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalised losses for each image
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss,pl


def scanmix_big_eval_train(p,args,model,epoch,eval_loader,criterion,device,output):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset))    
    processed = AverageMeter('Eval train')
    progress = ProgressMeter(len(eval_loader),
        [processed],
        prefix="Epoch: [{}]".format(epoch))
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader): 
            inputs, targets = batch['image'].to(device), batch['target'].to(device) 
            index = batch['meta']['index']
            outputs = model(inputs, forward_pass='dm')
            _, predicted = torch.max(outputs, 1) 
            loss = criterion(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                pl[index[b]]  = predicted[b]
            if batch_idx % 25 == 0:
                progress.display(batch_idx)
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalised losses for each image
    input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    if (p['dataset'] == 'webvision'):
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    else:
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-3)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    output['prob'] = prob
    output['pl'] = pl         



def scanmix_scan(train_loader, model, criterion, optimizer, epoch, device, update_cluster_head_only=False, 
                 neighbor_indices_static=None, neighbor_indices_dynamic=None, neighbor_fusion_alpha=0.8):
    """ 
    Train w/ SCAN-Loss (支持双源邻居加权融合)
    
    Args:
        neighbor_indices_static: 静态邻居库（SimCLR）
        neighbor_indices_dynamic: 动态邻居库（当前模型）
        neighbor_fusion_alpha: 静态邻居权重α，动态邻居权重为(1-α)
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to(device, non_blocking=True)
        neighbors = batch['neighbor'].to(device, non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='sl_head')
            neighbors_output = model(neighbors_features, forward_pass='sl_head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors, forward_pass='sl')
            neighbors_output = model(neighbors, forward_pass='sl')     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(total_loss.item())
        consistency_losses.update(consistency_loss.item())
        entropy_losses.update(entropy_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
        torch.cuda.empty_cache()