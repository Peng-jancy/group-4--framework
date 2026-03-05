import numpy as np
import matplotlib.pyplot as plt

def plot_final_model(loss_list, train_acc_list, test_acc_list, train_epochs):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,4))

    epochs = range(train_epochs)

    ax1.plot(epochs, loss_list)
    ax1.set_title("Training Loss")

    ax2.plot(epochs, train_acc_list,label="Train")
    ax2.plot(epochs, test_acc_list,label="Test")
    ax2.legend()

    final_train=np.mean(train_acc_list[-5:])
    final_test=np.mean(test_acc_list[-5:])

    ax3.bar(["Train","Test"],[final_train,final_test])

    plt.show()