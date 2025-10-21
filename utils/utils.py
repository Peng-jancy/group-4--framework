# utils/utils.py
# 可以放一些常用函数，比如绘制损失曲线，混淆矩阵等
import matplotlib.pyplot as plt

def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
