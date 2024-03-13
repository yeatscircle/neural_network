import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as transforms


# RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

        '''
        重述一下代码过程:
        pytorch已经把所有的RNN过程分析好了
        首先hidden_size指的是中间元个数,当然作为输出也是可以的
        然后我先初始化了一个h0拿来当作第一次rnn的input,其中的num_layers是因为每一组都要input一个H
        self.rnn的返回为两个结果,一个是整体的output-->[batch_size,sequence_length,hidden_size]
        第二个元素是最后的output,即[batch_size,hidden_size]
        
        同样,他们可以被gru给替代掉且gru的引入使得rnn正确率提升
        '''

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 其中的_应该是最后隐藏层的状态
        # out, _ = self.rnn(x, h0)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def save_checkpoint(state, filename='model.pth.tar'):
    print('You are trying save model!')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('You are trying reading model')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
n_epochs = 5
load_model = True

# Download data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('model.pth.tar'))

# Train Network
for epoch in range(n_epochs):
    if epoch == 2:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

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
        x = x.to(device=device).squeeze(1)
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
