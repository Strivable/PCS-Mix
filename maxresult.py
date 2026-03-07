import re

best_epoch = None
best_acc = -1.0

with open('./results/cifar-100/0.5-sym_acc1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        match = re.match(r'Epoch:(\d+)\s+Accuracy:(\d+\.\d+)', line.strip())
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

print(f'最高 Accuracy: {best_acc}，对应 Epoch: {best_epoch}')