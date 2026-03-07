"""
NEU-CLS Surface Defect Database Dataset
NEU-CLS包含6类钢材表面缺陷，每类300张图片，共1800张图片
类别: Crazing(Cr), Inclusion(In), Patches(Pa), Pitted_surface(PS), Rolled-in_scale(RS), Scratches(Sc)
图片尺寸: 200x200 灰度图
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json

class NEUCLSDataset(Dataset):
    """
    NEU-CLS数据集加载器
    数据集结构 (扁平化，所有图片在根目录下):
    NEU-CLS/
        ├── Cr_1.bmp, Cr_2.bmp, ..., Cr_300.bmp  (Crazing)
        ├── In_1.bmp, In_2.bmp, ..., In_300.bmp  (Inclusion)
        ├── Pa_1.bmp, Pa_2.bmp, ..., Pa_300.bmp  (Patches)
        ├── PS_1.bmp, PS_2.bmp, ..., PS_300.bmp  (Pitted Surface)
        ├── RS_1.bmp, RS_2.bmp, ..., RS_300.bmp  (Rolled-in Scale)
        └── Sc_1.bmp, Sc_2.bmp, ..., Sc_300.bmp  (Scratches)
    """
    
    # 类别映射
    class_to_idx = {
        'Cr': 0,  # Crazing
        'In': 1,  # Inclusion
        'Pa': 2,  # Patches
        'PS': 3,  # Pitted Surface
        'RS': 4,  # Rolled-in Scale
        'Sc': 5   # Scratches
    }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        """
        Args:
            root: 数据集根目录路径（所有图片所在目录）
            split: 'train' or 'test' (默认8:2划分)
            transform: 图像变换
            target_transform: 标签变换
        """
        super(NEUCLSDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载数据
        self.data = []
        self.targets = []
        self.image_paths = []
        
        # 读取根目录下所有图片文件
        all_files = sorted([f for f in os.listdir(root) if f.endswith(('.bmp', '.jpg', '.png'))])
        
        # 按类别分组
        class_files = {class_name: [] for class_name in self.class_to_idx.keys()}
        for img_file in all_files:
            # 从文件名提取类别前缀 (例如: Cr_1.bmp -> Cr)
            class_prefix = img_file.split('_')[0]
            if class_prefix in self.class_to_idx:
                class_files[class_prefix].append(img_file)
        
        # 对每个类别进行训练/测试划分
        for class_name, class_idx in self.class_to_idx.items():
            image_files = sorted(class_files[class_name])
            if len(image_files) == 0:
                print(f'Warning: No images found for class {class_name}')
                continue
            
            # 划分训练集和测试集 (8:2)
            num_images = len(image_files)
            split_point = int(num_images * 0.8)
            
            if split == 'train':
                selected_files = image_files[:split_point]
            else:  # test
                selected_files = image_files[split_point:]
            
            # 添加到数据集
            for img_file in selected_files:
                img_path = os.path.join(root, img_file)
                self.image_paths.append(img_path)
                self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
        print(f'NEU-CLS {split} set: {len(self.targets)} images loaded')
        
        # 统计类别分布
        unique, counts = np.unique(self.targets, return_counts=True)
        print(f'Class distribution: {dict(zip(unique, counts))}')
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        """
        Returns:
            dict with keys: 'image', 'target', 'meta'
        """
        img_path = self.image_paths[index]
        target = int(self.targets[index])
        
        # 读取图像 (NEU-CLS是灰度图)
        img = Image.open(img_path).convert('RGB')  # 转为RGB以兼容预训练模型
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return {'image': img, 'target': target, 'meta': {'index': index, 'class_name': self.idx_to_class[target]}}


class NoisyNEUCLS(Dataset):
    """
    带噪声标签的NEU-CLS数据集
    """
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 noise_file=None, noise_mode='sym', noise_rate=0.0):
        """
        Args:
            root: 数据集根目录
            split: 'train' or 'test'
            transform: 图像变换
            target_transform: 标签变换
            noise_file: 噪声标签文件路径 (JSON格式)
            noise_mode: 'sym' (对称噪声) or 'asym' (非对称噪声)
            noise_rate: 噪声率 (0.0-1.0)
        """
        super(NoisyNEUCLS, self).__init__()
        
        # 加载干净数据集
        self.clean_dataset = NEUCLSDataset(root, split, transform=None, target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.noise_mode = noise_mode
        self.noise_rate = noise_rate
        
        # 保存真实标签
        self.true_labels = self.clean_dataset.targets.copy()
        
        # 加载或生成噪声标签
        if noise_file is not None and os.path.exists(noise_file):
            print(f'Loading noise labels from {noise_file}')
            with open(noise_file, 'r') as f:
                noise_data = json.load(f)
            self.noise_labels = np.array(noise_data['noise_labels'])
            self.actual_noise_rate = noise_data.get('noise_rate', noise_rate)
            print(f'Loaded {len(self.noise_labels)} noisy labels, actual noise rate: {self.actual_noise_rate:.2%}')
        else:
            print(f'Generating {noise_mode} noise with rate {noise_rate:.2%}')
            self.noise_labels = self._generate_noise(noise_mode, noise_rate)
            self.actual_noise_rate = (self.noise_labels != self.true_labels).sum() / len(self.true_labels)
            print(f'Generated noise labels, actual noise rate: {self.actual_noise_rate:.2%}')
            
            # 保存噪声标签
            if noise_file is not None:
                os.makedirs(os.path.dirname(noise_file), exist_ok=True)
                with open(noise_file, 'w') as f:
                    json.dump({
                        'noise_labels': self.noise_labels.tolist(),
                        'noise_rate': float(self.actual_noise_rate),
                        'noise_mode': noise_mode
                    }, f, indent=2)
                print(f'Saved noise labels to {noise_file}')
    
    def _generate_noise(self, noise_mode, noise_rate):
        """生成噪声标签"""
        num_samples = len(self.true_labels)
        num_classes = 6  # NEU-CLS有6个类别
        noise_labels = self.true_labels.copy()
        
        if noise_mode == 'sym':
            # 对称噪声：随机翻转标签到其他类别
            num_noise = int(noise_rate * num_samples)
            noise_indices = np.random.choice(num_samples, num_noise, replace=False)
            
            for idx in noise_indices:
                original_label = noise_labels[idx]
                # 随机选择一个不同的类别
                other_labels = [l for l in range(num_classes) if l != original_label]
                noise_labels[idx] = np.random.choice(other_labels)
        
        elif noise_mode == 'asym':
            # 非对称噪声：类别间存在混淆倾向
            # 例如：Crazing -> Scratches, Inclusion -> Patches 等
            transition_matrix = {
                0: [5],     # Cr -> Sc (都是线性缺陷)
                1: [2],     # In -> Pa (都是块状缺陷)
                2: [1, 3],  # Pa -> In, PS
                3: [4],     # PS -> RS (都是表面缺陷)
                4: [3],     # RS -> PS
                5: [0]      # Sc -> Cr
            }
            
            for class_id in range(num_classes):
                class_indices = np.where(self.true_labels == class_id)[0]
                num_noise = int(noise_rate * len(class_indices))
                noise_indices = np.random.choice(class_indices, num_noise, replace=False)
                
                for idx in noise_indices:
                    noise_labels[idx] = np.random.choice(transition_matrix[class_id])
        
        return noise_labels
    
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, index):
        item = self.clean_dataset[index]
        
        # 使用噪声标签
        item['target'] = int(self.noise_labels[index])
        
        # 应用变换
        if self.transform is not None:
            item['image'] = self.transform(item['image'])
        
        if self.target_transform is not None:
            item['target'] = self.target_transform(item['target'])
        
        return item
