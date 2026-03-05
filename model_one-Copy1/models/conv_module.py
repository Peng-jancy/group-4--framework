import numpy as np
from .activation import ActivationFunction,Gradient

class ConvModule:
    def __init__(self, conv_config, input_data):
        self.conv_config = conv_config
        self.raw_input = input_data
        self.N, self.C_in, self.H_in, self.W_in = self.raw_input.shape
        self.conv_params = {}
        self.conv_cache = {}
        self.conv_gradients = {}
        self.conv_adam = {}
        self.__init_conv_params__()

    # 初始化卷积/池化层参数
    def __init_conv_params__(self):
        H, W = self.H_in, self.W_in
        C_in = self.C_in
        for idx, config in enumerate(self.conv_config):
            if config['type'] == 'conv':
                out_channels = config['out_channels']
                kernel_size = config['kernel_size']
                scale = np.sqrt(6.0 / (C_in * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
                kernel = np.random.uniform(-scale, scale, (out_channels, C_in, kernel_size, kernel_size)).astype(np.float32)
                bias = np.zeros(out_channels, dtype=np.float32)
                self.conv_params[f'kernel_{idx}'] = kernel
                self.conv_params[f'bias_{idx}'] = bias
                self.conv_gradients[f'dkernel_{idx}'] = np.zeros_like(kernel)
                self.conv_gradients[f'dbias_{idx}'] = np.zeros_like(bias)
                self.conv_adam[f'm_kernel_{idx}'] = np.zeros_like(kernel)
                self.conv_adam[f'v_kernel_{idx}'] = np.zeros_like(kernel)
                self.conv_adam[f'm_bias_{idx}'] = np.zeros_like(bias)
                self.conv_adam[f'v_bias_{idx}'] = np.zeros_like(bias)
                H = (H + 2 * config['padding'] - kernel_size) // config['stride'] + 1
                W = (W + 2 * config['padding'] - kernel_size) // config['stride'] + 1
                C_in = out_channels
            elif config['type'] == 'pool':
                pool_size = config['pool_size']
                stride = config['stride']
                H = (H - pool_size) // stride + 1
                W = (W - pool_size) // stride + 1
        self.final_C = C_in
        self.final_H = H
        self.final_W = W

    # 图像填充：保证卷积后尺寸不变
    def pad_image(self, x, padding):
        if padding == 0:
            return x
        if len(x.shape) == 3:
            return np.pad(x, ((0,0), (padding,padding), (padding,padding)), mode='constant').astype(np.float32)
        else:
            return np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant').astype(np.float32)

    # 卷积层前向传播
    def conv2d_forward(self, x, kernel, bias, stride, padding, af_type, layer_idx):
        N, C_in, H_in, W_in = x.shape
        out_channels, _, k_h, k_w = kernel.shape
        x_padded = self.pad_image(x, padding)
        H_out = (H_in + 2 * padding - k_h) // stride + 1
        W_out = (W_in + 2 * padding - k_w) // stride + 1
        output = np.zeros((N, out_channels, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for oc in range(out_channels):
                current_kernel = kernel[oc]
                current_bias = bias[oc]
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x_padded[n, :, h_start:h_start+k_h, w_start:w_start+k_w]
                        output[n, oc, i, j] = np.sum(window * current_kernel) + current_bias
        z = output.copy().astype(np.float32)
        if af_type == 1:
            output = ActivationFunction.relu(z)
        self.conv_cache[f'x_padded_{layer_idx}'] = x_padded
        self.conv_cache[f'kernel_{layer_idx}'] = kernel
        self.conv_cache[f'stride_{layer_idx}'] = stride
        self.conv_cache[f'padding_{layer_idx}'] = padding
        self.conv_cache[f'af_type_{layer_idx}'] = af_type
        self.conv_cache[f'z_{layer_idx}'] = z
        return output

    # 池化层前向传播
    def max_pool2d_forward(self, x, pool_size, stride, layer_idx):
        N, C, H_in, W_in = x.shape
        H_out = (H_in - pool_size) // stride + 1
        W_out = (W_in - pool_size) // stride + 1        
        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        mask = np.zeros_like(x, dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = x[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                        max_val = np.max(region)
                        output[n, c, i, j] = max_val
                        mask[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size] = (region == max_val)
        self.conv_cache[f'pool_mask_{layer_idx}'] = mask
        return output

    # 卷积模块整体前向传播
    def forward(self):
        x = self.raw_input.copy().astype(np.float32)
        self.conv_cache = {}
        for idx, config in enumerate(self.conv_config):
            self.conv_cache[f'input_{idx}'] = x
            if config['type'] == 'conv':
                kernel = self.conv_params[f'kernel_{idx}']
                bias = self.conv_params[f'bias_{idx}']
                x = self.conv2d_forward(x, kernel, bias, config['stride'], config['padding'], config['af'], idx)
            elif config['type'] == 'pool':
                x = self.max_pool2d_forward(x, config['pool_size'], config['stride'], idx)
        self.conv_output = x
        self.conv_output_flat = x.reshape(self.N, -1).astype(np.float32)
        return self.conv_output_flat

    # 测试专用前向传播：独立计算，不污染训练缓存
    def test_forward(self, test_input):
        x = test_input.copy().astype(np.float32)
        conv_cache = {}
        for idx, config in enumerate(self.conv_config):
            conv_cache[f'input_{idx}'] = x
            if config['type'] == 'conv':
                kernel = self.conv_params[f'kernel_{idx}']
                bias = self.conv_params[f'bias_{idx}']
                N, C_in, H_in, W_in = x.shape
                out_channels, _, k_h, k_w = kernel.shape
                # 修复：使用config['stride']而非未定义的stride
                stride = config['stride']
                padding = config['padding']
                x_padded = self.pad_image(x, padding)
                H_out = (H_in + 2 * padding - k_h) // stride + 1
                W_out = (W_in + 2 * padding - k_w) // stride + 1
                output = np.zeros((N, out_channels, H_out, W_out), dtype=np.float32)
                for n in range(N):
                    for oc in range(out_channels):
                        current_kernel = kernel[oc]
                        current_bias = bias[oc]
                        for i in range(H_out):
                            for j in range(W_out):
                                h_start = i * stride
                                w_start = j * stride
                                window = x_padded[n, :, h_start:h_start+k_h, w_start:w_start+k_w]
                                output[n, oc, i, j] = np.sum(window * current_kernel) + current_bias
                z = output.copy().astype(np.float32)
                x = ActivationFunction.relu(z)
            elif config['type'] == 'pool':
                N, C, H_in, W_in = x.shape
                pool_size = config['pool_size']
                stride = config['stride']
                H_out = (H_in - pool_size) // stride + 1
                W_out = (W_in - pool_size) // stride + 1
                output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
                for n in range(N):
                    for c in range(C):
                        for i in range(H_out):
                            for j in range(W_out):
                                h_start = i * stride
                                w_start = j * stride
                                region = x[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                                max_val = np.max(region)
                                output[n, c, i, j] = max_val
                x = output
        conv_output_flat = x.reshape(test_input.shape[0], -1).astype(np.float32)
        return conv_output_flat

    # 卷积层反向传播
    def conv2d_backward(self, d_out, layer_idx):
        x_padded = self.conv_cache[f'x_padded_{layer_idx}']
        kernel = self.conv_cache[f'kernel_{layer_idx}']
        stride = self.conv_cache[f'stride_{layer_idx}']
        padding = self.conv_cache[f'padding_{layer_idx}']
        af_type = self.conv_cache[f'af_type_{layer_idx}']
        z = self.conv_cache[f'z_{layer_idx}']
        x_input = self.conv_cache[f'input_{layer_idx}']
        N, out_channels, H_out, W_out = d_out.shape
        _, C_in, k_h, k_w = kernel.shape
        N, C_in, H_in, W_in = x_input.shape
        dz = d_out * Gradient.relu_gradient(z)
        d_bias = np.sum(dz, axis=(0, 2, 3)) / N
        d_kernel = np.zeros_like(kernel, dtype=np.float32)
        for oc in range(out_channels):
            for c in range(C_in):
                for i in range(k_h):
                    for j in range(k_w):
                        for h in range(H_out):
                            for w in range(W_out):
                                h_pad = h * stride + i
                                w_pad = w * stride + j
                                d_kernel[oc, c, i, j] += np.sum(x_padded[:, c, h_pad, w_pad] * dz[:, oc, h, w])
        d_kernel /= N
        d_x_padded = np.zeros_like(x_padded, dtype=np.float32)
        for n in range(N):
            for oc in range(out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        w_start = w * stride
                        d_x_padded[n, :, h_start:h_start+k_h, w_start:w_start+k_w] += kernel[oc] * dz[n, oc, h, w]
        d_x = d_x_padded[:, :, padding:padding+H_in, padding:padding+H_in]
        return d_x, d_kernel, d_bias

    # 池化层反向传播
    def max_pool2d_backward(self, d_out, layer_idx):
        mask = self.conv_cache[f'pool_mask_{layer_idx}']
        pool_size = self.conv_config[layer_idx]['pool_size']
        stride = self.conv_config[layer_idx]['stride']
        x_input = self.conv_cache[f'input_{layer_idx}']
        N, C, H_out, W_out = d_out.shape
        d_x = np.zeros_like(x_input, dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        d_x[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size] += mask[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size] * d_out[n, c, i, j]
        return d_x

    # 卷积模块整体反向传播
    def backward(self, fc_grad):
        d_conv = fc_grad.reshape(self.N, self.final_C, self.final_H, self.final_W).astype(np.float32)
        for idx in reversed(range(len(self.conv_config))):
            config = self.conv_config[idx]
            if config['type'] == 'pool':
                d_conv = self.max_pool2d_backward(d_conv, idx)
            elif config['type'] == 'conv':
                d_conv, d_kernel, d_bias = self.conv2d_backward(d_conv, idx)
                self.conv_gradients[f'dkernel_{idx}'] = d_kernel
                self.conv_gradients[f'dbias_{idx}'] = d_bias

    # Adam优化器更新卷积层参数
    def adam_update(self, lr, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for idx, config in enumerate(self.conv_config):
            if config['type'] == 'conv':
                kernel_key = f'kernel_{idx}'
                d_kernel = self.conv_gradients[f'dkernel_{idx}']
                m = self.conv_adam[f'm_kernel_{idx}']
                v = self.conv_adam[f'v_kernel_{idx}']
                m = beta1 * m + (1 - beta1) * d_kernel
                v = beta2 * v + (1 - beta2) * (d_kernel ** 2)
                m_corrected = m / (1 - beta1 ** t)
                v_corrected = v / (1 - beta2 ** t)
                self.conv_params[kernel_key] -= lr * m_corrected / (np.sqrt(v_corrected) + epsilon)
                self.conv_adam[f'm_kernel_{idx}'] = m
                self.conv_adam[f'v_kernel_{idx}'] = v

                bias_key = f'bias_{idx}'
                d_bias = self.conv_gradients[f'dbias_{idx}']
                m = self.conv_adam[f'm_bias_{idx}']
                v = self.conv_adam[f'v_bias_{idx}']
                m = beta1 * m + (1 - beta1) * d_bias
                v = beta2 * v + (1 - beta2) * (d_bias ** 2)
                m_corrected = m / (1 - beta1 ** t)
                v_corrected = v / (1 - beta2 ** t)
                self.conv_params[bias_key] -= lr * m_corrected / (np.sqrt(v_corrected) + epsilon)

# 全连接层网络：实现多层感知机的前向/反向传播
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
