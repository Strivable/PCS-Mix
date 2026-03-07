"""
生成NEU-CLS数据集的噪声标签文件
"""
import numpy as np
import json
import os
import argparse
from pathlib import Path

def generate_neucls_noise(data_root, noise_mode='sym', noise_rate=0.4, output_file=None, seed=123):
    """
    为NEU-CLS数据集生成噪声标签
    
    Args:
        data_root: NEU-CLS数据集根目录
        noise_mode: 'sym' (对称) or 'asym' (非对称)
        noise_rate: 噪声率 (0.0-1.0)
        output_file: 输出JSON文件路径
        seed: 随机种子
    """
    np.random.seed(seed)
    
    # NEU-CLS类别
    class_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # 收集训练集样本 (前80%)
    all_labels = []
    all_paths = []
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(data_root, class_name)
        if not os.path.exists(class_dir):
            print(f'Warning: {class_dir} not found!')
            continue
        
        # 获取图片文件
        image_files = sorted([f for f in os.listdir(class_dir) 
                             if f.endswith(('.bmp', '.jpg', '.png'))])
        
        # 训练集：前80%
        split_point = int(len(image_files) * 0.8)
        train_files = image_files[:split_point]
        
        for img_file in train_files:
            all_paths.append(os.path.join(class_name, img_file))
            all_labels.append(class_idx)
    
    all_labels = np.array(all_labels)
    num_samples = len(all_labels)
    
    print(f'Total training samples: {num_samples}')
    print(f'Class distribution: {np.bincount(all_labels)}')
    
    # 生成噪声标签
    noise_labels = all_labels.copy()
    
    if noise_mode == 'sym':
        # 对称噪声
        num_noise = int(noise_rate * num_samples)
        noise_indices = np.random.choice(num_samples, num_noise, replace=False)
        
        print(f'\nGenerating symmetric noise: {num_noise} samples')
        
        for idx in noise_indices:
            original_label = noise_labels[idx]
            # 随机选择其他类别
            other_labels = [l for l in range(num_classes) if l != original_label]
            noise_labels[idx] = np.random.choice(other_labels)
    
    elif noise_mode == 'asym':
        # 非对称噪声 - 基于缺陷相似性
        # Crazing <-> Scratches (线性缺陷)
        # Inclusion <-> Patches (块状缺陷)
        # Pitted Surface <-> Rolled-in Scale (表面缺陷)
        
        transition_matrix = {
            0: [5],      # Cr -> Sc
            1: [2],      # In -> Pa
            2: [1, 3],   # Pa -> In, PS
            3: [4],      # PS -> RS
            4: [3],      # RS -> PS
            5: [0]       # Sc -> Cr
        }
        
        print(f'\nGenerating asymmetric noise with transitions:')
        for k, v in transition_matrix.items():
            print(f'  Class {class_names[k]} -> {[class_names[i] for i in v]}')
        
        for class_id in range(num_classes):
            class_indices = np.where(all_labels == class_id)[0]
            num_noise = int(noise_rate * len(class_indices))
            noise_indices = np.random.choice(class_indices, num_noise, replace=False)
            
            print(f'  Class {class_names[class_id]}: {len(class_indices)} samples, '
                  f'{num_noise} noisy ({num_noise/len(class_indices)*100:.1f}%)')
            
            for idx in noise_indices:
                noise_labels[idx] = np.random.choice(transition_matrix[class_id])
    
    # 计算实际噪声率
    actual_noise_rate = (noise_labels != all_labels).sum() / num_samples
    
    print(f'\n=== Noise Statistics ===')
    print(f'Target noise rate: {noise_rate:.2%}')
    print(f'Actual noise rate: {actual_noise_rate:.2%}')
    print(f'Noisy samples: {(noise_labels != all_labels).sum()} / {num_samples}')
    
    # 统计每个类别的噪声情况
    print(f'\nPer-class noise statistics:')
    for class_id in range(num_classes):
        class_mask = all_labels == class_id
        class_total = class_mask.sum()
        class_noisy = ((all_labels == class_id) & (noise_labels != class_id)).sum()
        print(f'  {class_names[class_id]}: {class_noisy}/{class_total} '
              f'({class_noisy/class_total*100:.1f}%) noisy')
    
    # 保存到文件
    if output_file is None:
        output_file = f'noise/neucls/{noise_rate:.2f}{"_asym" if noise_mode == "asym" else ""}.json'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    noise_data = {
        'noise_labels': noise_labels.tolist(),
        'true_labels': all_labels.tolist(),
        'noise_rate': float(noise_rate),
        'actual_noise_rate': float(actual_noise_rate),
        'noise_mode': noise_mode,
        'num_samples': int(num_samples),
        'num_classes': num_classes,
        'class_names': class_names,
        'image_paths': all_paths,
        'seed': seed
    }
    
    with open(output_file, 'w') as f:
        json.dump(noise_data, f, indent=2)
    
    print(f'\n✓ Noise labels saved to: {output_file}')
    
    return noise_labels, actual_noise_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noise labels for NEU-CLS dataset')
    parser.add_argument('--data_root', type=str, default='./datasets/NEU-CLS/',
                       help='NEU-CLS dataset root directory')
    parser.add_argument('--noise_mode', type=str, default='sym', choices=['sym', 'asym'],
                       help='Noise type: symmetric or asymmetric')
    parser.add_argument('--noise_rate', type=float, default=0.4,
                       help='Noise rate (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 验证数据集路径
    if not os.path.exists(args.data_root):
        print(f'Error: Data root not found: {args.data_root}')
        print('Please specify correct path with --data_root')
        exit(1)
    
    # 生成噪声
    generate_neucls_noise(
        data_root=args.data_root,
        noise_mode=args.noise_mode,
        noise_rate=args.noise_rate,
        output_file=args.output,
        seed=args.seed
    )
    
    print('\n=== Usage ===')
    print('To train with this noise file:')
    print(f'  python my_ScanMix.py \\')
    print(f'    --config_env configs/env.yml \\')
    print(f'    --config_exp configs/scanmix/scanmix_neucls.yml \\')
    print(f'    --noise_mode {args.noise_mode} \\')
    print(f'    --r {args.noise_rate} \\')
    print(f'    --seed {args.seed}')
