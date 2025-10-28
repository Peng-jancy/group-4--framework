import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from config import batch_size

class TensorDataset:
    """自定义实现的TensorDataset"""
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("至少需要一个张量")
        
        self.tensors = tensors
        self.length = len(tensors[0])  # 假设所有张量的第一个维度都是样本数
        
        # 检查所有张量的第一个维度是否相同
        for i, tensor in enumerate(tensors):
            if len(tensor) != self.length:
                raise ValueError(f"张量 {i} 的长度 ({len(tensor)}) 与第一个张量的长度 ({self.length}) 不匹配")
    
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            # 单个索引
            return tuple(tensor[index] for tensor in self.tensors)
        elif isinstance(index, slice):
            # 切片索引
            return tuple(tensor[index] for tensor in self.tensors)
        else:
            # 其他索引类型（如列表、数组等）
            return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        """返回数据集的长度"""
        return self.length

class DataLoader:
    """自定义实现的DataLoader"""
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # 生成索引列表
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """返回迭代器"""
        if self.shuffle:
            # 如果需要打乱，重新生成随机索引
            indices = np.random.permutation(self.indices).tolist()
        else:
            indices = self.indices[:]
        
        # 按批次分组
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # 获取批次数据
            batch_data = []
            for idx in batch_indices:
                item = self.dataset[idx]
                batch_data.append(item)
            
            # 将批次数据组织成张量格式
            if len(batch_data) == 0:
                continue
            
            # 如果数据项是元组（如特征和标签），分别组织
            if isinstance(batch_data[0], tuple):
                # 多个张量的情况
                result = []
                for j in range(len(batch_data[0])):
                    tensor_data = [item[j] for item in batch_data]
                    # 转换为numpy数组
                    result.append(np.stack(tensor_data, axis=0))
                yield tuple(result)
            else:
                # 单个张量的情况
                yield np.stack(batch_data, axis=0)
    
    def __len__(self):
        """返回批次数量"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
def load_data(csv_path="iris.csv", test_size=0.2):

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 假设最后一列是标签（类别），其余为特征
    X = df.iloc[:, :-1].values  # 所有行，除最后一列外的特征
    y = df.iloc[:, -1].values   # 最后一列是标签（字符串或整数）

    # 如果标签是字符串（如 "Iris-setosa"），需要编码为整数
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object':  # 判断是否为字符串类型
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y  # 分层抽样，保持类别比例
    )
    
    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
