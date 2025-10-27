import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import batch_size

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
