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
        identifier = self.dataframe.iloc[idx, 0]  # 第一列标识符
        output_label = self.dataframe.iloc[idx, 1]  # 第二列输出标签
        input_features = self.dataframe.iloc[idx, 2:].values.astype('float32')  # 第三列到第十二列输入特征

        if self.transform:
            input_features = self.transform(input_features)

        return identifier, input_features, output_label


# 定义转换器
class CustomTransform:
    def __call__(self, input_features):
        # 添加你自己的转换操作
        input_features = (input_features - 0.5) / 0.5  # 归一化，这里使用与你提供的一致的均值和标准差
        return input_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1D Convolutional layer for processing input sequence
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for further processing
        self.fc1 = nn.Linear(32 * 2, 120)  # Adjust the input size based on the output size of convolutions
        self.fc2 = nn.Linear(120, 84)
        # Output layer with a single neuron for regression (outputting an integer)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = x.view(-1,11)
        x = x.unsqueeze(1)  # Add a channel dimension for 1D convolution
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)  # Remove the extra dimension added earlier

# Function to calculate mean squared error on the test set
def calculate_mse(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _, inputs, labels in test_loader:
            outputs = model(inputs)
            mse_loss = criterion(outputs, labels)
            total_loss += mse_loss

    mse = total_loss / len(test_loader)
    return mse

if __name__ == "__main__":

    # 创建训练和测试数据集实例
    custom_transform = CustomTransform()

    train_dataset = CustomDataset(train_df, transform=custom_transform)
    test_dataset = CustomDataset(test_df, transform=custom_transform)

    # 创建数据加载器
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Define your neural network
    net = Net()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)

    # Training loop
    for epoch in range(5):
        running_loss = 0.0
        print(epoch)
        for i, data in enumerate(train_loader, 0):
            _, inputs, labels = data

            # Convert inputs to double precision
            inputs = inputs.double()

            # Ensure labels are of type Float
            labels = labels.float()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')


    # Calculate and print mean squared error on the test set
    test_mse = calculate_mse(net, test_loader, criterion)
    print(f'Mean Squared Error on Test Set: {test_mse}')


    # Save the model
    PATH = './your_net.pth'
    torch.save(net.state_dict(), PATH)

