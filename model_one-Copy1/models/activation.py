import numpy as np

class ActivationFunction:

    @staticmethod
    def sigmoid(x):
        x=np.clip(x,-500,500)
        return 1/(1+np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0,x)

    @staticmethod
    def softmax(x):
        x_shifted=x-np.max(x,axis=-1,keepdims=True)
        exp_x=np.exp(x_shifted)
        return exp_x/(np.sum(exp_x,axis=-1,keepdims=True)+1e-12)


class Gradient:

    @staticmethod
    def sigmoid_gradient(z):
        s=ActivationFunction.sigmoid(z)
        return s*(1-s)

    @staticmethod
    def relu_gradient(z):
        return (z>0).astype(np.float32)