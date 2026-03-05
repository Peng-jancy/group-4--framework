from sklearn.datasets import fetch_openml
import numpy as np


def load_mnist(total_samples,test_num):

    X,y=fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)

    X=X[:total_samples].astype(np.float32)/255
    y=y[:total_samples].astype(int)

    X_test=X[-test_num:].reshape(-1,1,28,28)
    y_test=y[-test_num:]

    X_train=X[:-test_num].reshape(-1,1,28,28)
    y_train=y[:-test_num]

    return X_train,y_train,X_test,y_test