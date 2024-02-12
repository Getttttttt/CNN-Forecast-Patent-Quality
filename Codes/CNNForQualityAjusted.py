import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from captum.attr import IntegratedGradients
from torch.autograd import Variable


# 读取CSV文件
csv_path = 'networkRecord.csv'
df = pd.read_csv(csv_path)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 定义MAPE损失函数
def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100


# 读取CSV文件
csv_path = 'networkRecord.csv'
df = pd.read_csv(csv_path)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 新的数据读取逻辑
        identifier = (self.dataframe.iloc[idx, 0], self.dataframe.iloc[idx, 26])
        channel1 = self.dataframe.iloc[idx, 1:25].values.astype('float32')
        channel2 = self.dataframe.iloc[idx, [25] + list(range(27, 32))].values.astype('int32')
        channel3 = np.array([self.dataframe.iloc[idx, 26]]).astype('int32')  # 确保是一维数组
        channel4 = self.dataframe.iloc[idx, 33:39].values.astype('int32')
        channel1 = channel1.astype('float32')
        channel2 = channel2.astype('float32')
        channel3 = channel3.astype('float32')
        channel4 = channel4.astype('float32')
        output_label = self.dataframe.iloc[idx, 32]
        
        # 将数据写入文本文件
        with open("data_loading_log.txt", "a") as file:
            file.write(f"Loaded data: {identifier}, {channel1}, {channel2}, {channel3}, {channel4}, {output_label}\n")


        if self.transform:
            channel1 = self.transform(channel1)
            channel2 = self.transform(channel2)
            channel3 = self.transform(channel3)
            channel4 = self.transform(channel4)

        return identifier, (channel1, channel2, channel3, channel4), output_label

# 定义转换器
class CustomTransform:
    def __call__(self, input_features):
        # 添加你自己的转换操作
        input_features = (input_features - 0.5) / 0.5  # 归一化
        return input_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义处理第一个通道的卷积层（浮点型数据）
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        # 定义处理第二个通道的卷积层（整型数据）
        self.conv2_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        # 定义处理第三个通道的卷积层（整型数据）
        self.conv3 = nn.Conv1d(1, 32, kernel_size=1, stride=1)
        self.fc_channel3 = nn.Linear(1, 32)

        # 定义处理第四个通道的卷积层（整型数据）
        self.conv4_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool1d(2, 2)

        # 定义全连接层
        self.fc1 = nn.Linear(224, 120)  # 假设每个卷积层输出64个特征，并且有4个通道
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # 定义自适应池化层以统一输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        channel1, channel2, channel3, channel4 = x
        
        channel1 = channel1.float()
        channel2 = channel2.float()
        channel3 = channel3.float()
        channel4 = channel4.float()

        # 处理第一个通道
        channel1 = channel1.unsqueeze(1)  # 确保维度是 [batch_size, 1, length]
        channel1 = F.relu(self.conv1_1(channel1))
        channel1 = F.relu(self.conv1_2(channel1))
        channel1 = self.pool(channel1)

        # 处理第二个通道
        channel2 = channel2.unsqueeze(1)
        channel2 = F.relu(self.conv2_1(channel2))
        channel2 = F.relu(self.conv2_2(channel2))
        channel2 = self.pool(channel2)

        # 处理第三个通道
        channel3 = channel3.view(-1, 1)  # 调整形状为 [batch_size, 1]
        channel3 = F.relu(self.fc_channel3(channel3))
        channel3 = channel3.unsqueeze(-1)  # 调整形状为 [batch_size, 32, 1]

        # 处理第四个通道
        channel4 = channel4.unsqueeze(1)
        channel4 = F.relu(self.conv4_1(channel4))
        channel4 = F.relu(self.conv4_2(channel4))
        channel4 = self.pool(channel4)

        # 应用自适应池化层
        channel1 = self.adaptive_pool(channel1).squeeze(-1)
        channel2 = self.adaptive_pool(channel2).squeeze(-1)
        channel3 = self.adaptive_pool(channel3).squeeze(-1)
        channel4 = self.adaptive_pool(channel4).squeeze(-1)

        # 合并来自四个通道的特征
        combined = torch.cat((channel1, channel2, channel3, channel4), dim=1)

        # 展平特征以用于全连接层
        combined = torch.flatten(combined, 1)

        # 通过全连接层
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = self.fc3(combined)
        
        output = combined

        return output.squeeze(-1)



# Function to calculate mean squared error on the test set
def calculate_mae(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _, inputs, labels in test_loader:
            outputs = model(inputs)
            mae_loss = criterion(outputs, labels)
            total_loss += mae_loss

    mse = total_loss / len(test_loader)
    return mse

if __name__ == "__main__":

    # 创建训练和测试数据集实例
    custom_transform = CustomTransform()

    train_dataset = CustomDataset(train_df, transform=custom_transform)
    test_dataset = CustomDataset(test_df, transform=custom_transform)

    # 创建数据加载器
    batch_size = 40
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Define your neural network
    net = Net()

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)



    # Training loop
    for epoch in range(30):
        running_loss = 0.0
        print(epoch)
        for i, data in enumerate(train_loader, 0):
            _, inputs_tuple, labels = data

            # 分别处理每个通道的数据
            inputs_processed = []
            for inputs in inputs_tuple:
                inputs = inputs.to(torch.float32)  # 转换为float32，而不是double
                inputs_processed.append(inputs)

            optimizer.zero_grad()

            outputs = net(inputs_processed)  # 现在inputs_processed是一个包含处理后通道数据的列表
            labels = labels.float()  # 确保标签也是float类型

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:4d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    # 测试模型并保存结果到CSV
    net.eval()  # 将模型设置为评估模式
    results = []
    with torch.no_grad():
        for identifiers, inputs_tuple, labels in test_loader:
            inputs_processed = [inp.to(torch.float32) for inp in inputs_tuple]
            outputs = net(inputs_processed)

            # 假设 identifiers 是两个相同长度列表的元组
            for id1, id2, output, label in zip(identifiers[0], identifiers[1], outputs, labels):
                deviation = output.item() - label.item()
                results.append([(id1, id2), output.item(), label.item(), deviation])



    # 将结果保存到CSV文件
    results_df = pd.DataFrame(results, columns=['ID', 'Prediction', 'Actual', 'Deviation'])
    results_df.to_csv('SGD_test_results.csv', index=False)


    # Calculate and print mean squared error on the test set
    test_mae = calculate_mae(net, test_loader, criterion)
    print(f'Mean Squared Error on Test Set: {test_mae}')


    # Save the model
    PATH = './SGD_ajusted_net.pth'
    torch.save(net.state_dict(), PATH)
    
    # 获取一个样本输入
    sample_data = next(iter(train_loader))
    sample_inputs = sample_data[1]  # 这里假设 sample_data[1] 是输入数据

    # 转换为 Variable 并确保输入形状与网络期望的形状相匹配
    processed_inputs = [Variable(inp.float()) for inp in sample_inputs]

    # 使用网络进行一次前向传播
    output = net(processed_inputs)

    # 为每个输入创建一个单独的条目
    input_dict = {f'input_{i}': inp for i, inp in enumerate(processed_inputs)}

    # 可视化计算图
    dot = make_dot(output, params=dict(list(net.named_parameters()) + list(input_dict.items())))
    
    # 渲染并保存计算图到文件
    dot.render('SGD_network_graph', format='png')
    


