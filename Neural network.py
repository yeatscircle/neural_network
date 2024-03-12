import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as transforms


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# model = NN(20, 10)
# # 第一个变量为batch_size,第二个为维度
# tensor = torch.randn(10, 20)
# print(model.forward(tensor).shape)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
n_epochs = 1

# Download data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = NN(input_size, num_classes).to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device=device), targets.to(device=device)

        # Get to correct shape
        # data.shape[0] 获取的是当前批次数据的第一个维度
        # -1 告诉 reshape 函数自动计算剩余的维度，以便保持数据中元素的总数不变。
        data = data.reshape(data.shape[0], -1)


        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        # pytorch的默认梯度使用是累积而非替代,所以需要这个步骤
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        x = x.reshape(x.shape[0], -1)


        with torch.no_grad():
            scores = model(x)
            # 表示省略第一个返回的值
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            # 批次数量
            num_samples += predictions.size(0)
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()
    return num_correct / num_samples


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
