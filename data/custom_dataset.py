"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
import random
from PIL import Image
import pdb

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)
        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None, predicted_labels=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.predicted_labels = predicted_labels
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        
        # Validate neighbor indices match dataset size
        if self.indices.shape[0] != len(self.dataset):
            raise ValueError(
                f"Neighbor indices size mismatch: indices.shape[0]={self.indices.shape[0]}, "
                f"but dataset size={len(self.dataset)}. "
                f"This usually means the neighbor file was generated with a different dataset configuration "
                f"(e.g., different num_classes). Please regenerate the neighbor files with the current config."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        if self.predicted_labels is not None:
            label = self.predicted_labels[index]
            neighbor_indices = self.indices[index]
            neighbor_labels = self.predicted_labels[neighbor_indices]
            same_label_neighbors = np.where(neighbor_labels == label)[0]
            valid_neighbors = neighbor_indices[same_label_neighbors]
            if valid_neighbors.size == 0:
                valid_neighbors = neighbor_indices
        else:
            valid_neighbors = self.indices[index]
        neighbor_index = np.random.choice(valid_neighbors, 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        # Handle None transform case with ToTensor fallback
        if self.anchor_transform is not None:
            anchor['image'] = self.anchor_transform(anchor['image'])
        else:
            anchor['image'] = transforms.ToTensor()(anchor['image'])
        
        if self.neighbor_transform is not None:
            neighbor['image'] = self.neighbor_transform(neighbor['image'])
        else:
            neighbor['image'] = transforms.ToTensor()(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output


""" 
    NoisyDataset
"""
class NoisyDataset(Dataset):
    def __init__(self, dataset, meta_info):
        super(NoisyDataset, self).__init__()
        self.transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset 
        self.probability = meta_info['probability']
        self.mode = meta_info['mode']

        r = meta_info['r']
        pred = meta_info['pred']
        noise_file = meta_info['noise_file']
        noise_mode = meta_info['noise_mode']
        dataset_name = meta_info['dataset']

        if os.path.exists(noise_file):
            noise = json.load(open(noise_file,"r"))
            # Handle both dict format and direct list format
            if isinstance(noise, dict):
                noise_labels = noise['noise_labels']
            elif isinstance(noise, list):
                noise_labels = noise
            else:
                raise ValueError(f"Unexpected noise file format in {noise_file}: {type(noise)}")
        else:
            raise NotImplementedError()

        self.noise_labels = noise_labels
        
        # 存储真实标签（从原始数据集获取）
        # 兼容不同的数据集结构
        if hasattr(self.dataset, 'targets'):
            # CIFAR等数据集直接有targets属性
            self.true_labels = self.dataset.targets
        else:
            # 其他数据集通过__getitem__获取
            self.true_labels = [self.dataset.__getitem__(i)['target'] for i in range(len(self.dataset))]
        
        if self.mode == 'labeled':
            self.indices = pred.nonzero(as_tuple=True)[0].tolist()
        elif self.mode == 'unlabeled':
            self.indices = (~pred).nonzero(as_tuple=True)[0].tolist()
        elif self.mode == 'all' or self.mode == 'pretext':
            self.indices = list(range(len(self.dataset)))
        elif self.mode == 'neighbor':
            # self.indices = pred.nonzero()[0]
            self.indices = list(range(len(self.dataset)))
        else:
            raise ValueError('Invalid noisy dataset mode')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        true_index = self.indices[index]
        image = self.dataset.__getitem__(true_index)
        if self.mode=='labeled':
            img, target, prob = image, self.noise_labels[true_index], self.probability[true_index]
            img = img['image']
            if self.transform is not None:
                img1 = self.transform(img) 
                img2 = self.transform(img)
            else:
                # 当transform为None时，至少需要转换为tensor
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                img1 = to_tensor(img)
                img2 = to_tensor(img)
            return img1, img2, target, prob, true_index            
        elif self.mode=='unlabeled':
            img = image
            img = img['image']
            if self.transform is not None:
                img1 = self.transform(img) 
                img2 = self.transform(img)
            else:
                # 当transform为None时，至少需要转换为tensor
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                img1 = to_tensor(img)
                img2 = to_tensor(img)
            return img1, img2
        elif self.mode=='all':
            img, target = image, self.noise_labels[true_index]
            img = img['image']
            if self.transform is not None:
                img = self.transform(img)
            else:
                # 当transform为None时，至少需要转换为tensor
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                img = to_tensor(img)
            return img, target, true_index
        elif self.mode=='neighbor':
            return image
        elif self.mode=='pretext':
            img = image['image']
            if self.transform is not None:
                img = self.transform(img)
            return {'image': img, 'target': self.noise_labels[true_index]}
    
    def get_true_labels_map(self):
        """
        返回索引到真实标签的映射字典
        用于Oracle模式测试
        """
        return {i: label for i, label in enumerate(self.true_labels)}
    
    def get_true_label(self, index):
        """
        获取指定索引的真实标签
        """
        return self.true_labels[index]

