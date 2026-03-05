import numpy as np
from .activation import ActivationFunction,Gradient
from utils.metrics import calculate_accuracy

class my_flow:
    def __init__(self, hidden_layers, input_data, label_data, categories_size, activation_function_label_list, train_numder=100, lr=0.001, lamda=0.01):
        self.label_data = label_data        
        self.input_data = input_data.astype(np.float32)
        self.batch_size = input_data.shape[0]
        self.parameters = input_data.shape[1]
        self.categories_size = categories_size
        self.hidden_layers = hidden_layers
        self.a_f = activation_function_label_list
        self.train_numder = train_numder
        self.lr = lr
        self.lamda = lamda
        self.output = None
        self.size_list = [self.parameters] + self.hidden_layers + [self.categories_size]
        self.theta_list = self.__init_theta_list__()
        self.m = [np.zeros_like(theta) for theta in self.theta_list]
        self.v = [np.zeros_like(theta) for theta in self.theta_list]
        self.t = 0
        self.gradients = [None] * len(self.theta_list)
        
    # 全连接层参数初始化（Xavier初始化）
    def __init_theta_list__(self):
        theta_list = []
        for i in range(len(self.size_list)-1):
            af = self.a_f[i]
            in_dim = self.size_list[i]
            out_dim = self.size_list[i+1]
            if af in [1]:
                scale = np.sqrt(2.0 / (in_dim + out_dim))
                theta = np.random.normal(0.0, scale, size=(in_dim+1, out_dim)).astype(np.float32)
            else:
                scale = np.sqrt(6.0 / (in_dim + out_dim))
                theta = np.random.uniform(-scale, scale, size=(in_dim+1, out_dim)).astype(np.float32)
            theta_list.append(theta)
        return theta_list
        
    # 标签独热编码：适配交叉熵损失
    def one_hot(self):
        categories = []
        for k in range(self.categories_size):
            categories.append((self.label_data == k).astype(np.float32))
        return np.array(categories).T.astype(np.float32)
    
    # 计算当前批次训练准确率
    def accuracy(self):
        return calculate_accuracy(self.output, self.label_data)
        
    # 全连接层前向传播
    def forward(self):
        self.zs = []
        self.activations = [self.input_data]
        X = self.input_data.copy().astype(np.float32)
        for i in range(len(self.size_list)-1):
            X = np.insert(arr=X, obj=0, axis=1, values=1).astype(np.float32)
            z = X @ self.theta_list[i]
            self.zs.append(z)
            af_type = self.a_f[i]
            if af_type == 1:
                X = ActivationFunction.relu(z)
            elif af_type == 4:
                X = ActivationFunction.softmax(z)
            self.activations.append(X)
        self.output = X

    # 测试专用前向传播：独立计算
    def test_forward(self, test_input):
        zs = []
        activations = [test_input]
        X = test_input.copy().astype(np.float32)
        for i in range(len(self.size_list)-1):
            X = np.insert(arr=X, obj=0, axis=1, values=1).astype(np.float32)
            z = X @ self.theta_list[i]
            zs.append(z)
            af_type = self.a_f[i]
            if af_type == 1:
                X = ActivationFunction.relu(z)
            elif af_type == 4:
                X = ActivationFunction.softmax(z)
            activations.append(X)
        return X
        
    # 带L2正则化的交叉熵损失
    def loss(self):
        y_true = self.one_hot()
        y_pred = self.output 
        eps = 1e-12
        J = -np.sum(y_true * np.log(y_pred + eps)) / self.batch_size
        theta_sum = 0
        for theta in self.theta_list:
            theta_sum += np.sum(theta[1:, :] ** 2) 
        J += (self.lamda / (2 * self.batch_size)) * theta_sum
        return J 
      
    # 全连接层反向传播
    def backward(self):
        y_true = self.one_hot()
        y_pred = self.output
        delta = y_pred - y_true
        for i in reversed(range(len(self.theta_list))):
            A_prev_no_bias = self.activations[i] 
            A_prev = np.insert(A_prev_no_bias, 0, 1, axis=1).astype(np.float32)
            dW = (A_prev.T @ delta) / self.batch_size 
            dW[1:, :] += (self.lamda / self.batch_size) * self.theta_list[i][1:, :]
            self.gradients[i] = dW
            if i > 0:
                delta = delta @ self.theta_list[i][1:, :].T
                z_prev = self.zs[i - 1]
                delta = delta * Gradient.relu_gradient(z_prev)

    # Adam优化器更新全连接层参数
    def adam_update(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        for i in range(len(self.theta_list)):
            grad = self.gradients[i]
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * grad ** 2
            m_corrected = self.m[i] / (1 - beta1 ** self.t)
            v_corrected = self.v[i] / (1 - beta2 ** self.t)
            self.theta_list[i] -= self.lr * m_corrected / (np.sqrt(v_corrected) + epsilon)
