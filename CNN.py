import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as transforms


# CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            # W_out = [(W_in + 2 * P_w - K_w) / S_w] + 1
            # H_out = [(H_in + 2 * P_h - K_h) / S_h] + 1
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
n_epochs = 1

# Download data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device=device), targets.to(device=device)

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
