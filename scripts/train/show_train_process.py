import matplotlib.pyplot as plt

# Dữ liệu lấy từ log
epochs = list(range(1, 11))
train_acc = [0.4932, 0.7915, 0.9247, 0.9666, 0.9807, 0.9803, 0.9816, 0.9872, 0.9859, 0.9863]
val_acc   = [0.7717, 0.8756, 0.9732, 0.9890, 0.9858, 0.9858, 0.9953, 0.9921, 0.9921, 0.9953]

# Tìm epoch tốt nhất
best_val = max(val_acc)
best_epoch = val_acc.index(best_val) + 1

plt.style.use("seaborn-v0_8")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Accuracy chart
axes[0].plot(epochs, train_acc, marker='o', linewidth=2, label='Train Accuracy')
axes[0].plot(epochs, val_acc, marker='s', linewidth=2, label='Validation Accuracy')
axes[0].scatter(best_epoch, best_val, color='red', s=100, zorder=5, label=f'Best Val: {best_val:.4f}')
axes[0].annotate(
    f'Best Epoch = {best_epoch}\nVal Acc = {best_val:.4f}',
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 0.3, best_val - 0.08),
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=10,
    color='red'
)
axes[0].set_title('Training vs Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(epochs)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# 2. Generalization gap
gap = [t - v for t, v in zip(train_acc, val_acc)]
axes[1].plot(epochs, gap, marker='d', color='purple', linewidth=2)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Generalization Gap (Train Acc - Val Acc)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gap')
axes[1].set_xticks(epochs)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()