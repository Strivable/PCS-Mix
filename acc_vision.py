import matplotlib.pyplot as plt

# 读取数据
epochs = []
accuracies = []
# F:/Pyworkspace/noisy_labels/ScanMix/results/cifar-10/scanmix/r=0.2_sym_fix_top/acc.txt  r=0.2_sym_onlynew2d/acc.txt'

with open('F:/Pyworkspace/noisy_labels/ScanMix/results/cifar-10/scanmix/r=0.2_sym_fix_top/acc.txt', 'r') as f:
   for line in f:
       parts = line.strip().split()
       epoch = int(parts[0].split(':')[1])
       acc = float(parts[1].split(':')[1])
       epochs.append(epoch)
       accuracies.append(acc)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, 'b-', linewidth=1.5, marker='o', markersize=3, markevery=5)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('CIFAR-10 Training Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标注关键信息
plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='acc')
plt.legend()

# 设置y轴范围，突出变化
plt.ylim([80, 100])

plt.tight_layout()
# plt.savefig('accuracy_curve.png', dpi=300)
plt.show()

# 打印关键统计
print(f"初始精度: {accuracies[0]:.2f}%")
print(f"最终精度: {accuracies[-1]:.2f}%")
print(f"最大精度: {max(accuracies):.2f}% (Epoch {accuracies.index(max(accuracies))})")
print(f"提升幅度: {accuracies[-1] - accuracies[0]:.2f}%")