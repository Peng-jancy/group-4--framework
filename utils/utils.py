#放一些常用函数，比如绘制损失曲线等
import matplotlib.pyplot as plt

def plot_loss(loss_list):#绘制损失函数
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
