import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FinalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.conv4 = nn.Conv2d(128, 512, 9, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12800, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ExperimentalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 1)
        self.conv4 = nn.Conv2d(128, 512, 7, 1)
        self.conv5 = nn.Conv2d(512, 1024, 7, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch=1, title='', test_loader=None):
    model.train()
    if title != '':
        print(title)
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train, Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i + 1, epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        if test_loader is not None:
            test(model=model, device=device, test_loader=test_loader)


def test(model: nn.Module, device: str, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    k = 3
    correct_i = np.zeros(k)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if device != 'cpu':
        return pred.cpu().numpy()
    return pred.numpy()


def get_dataloader(digits, y, batch_size_test=1000, device='cpu'):
    tensor_x = torch.Tensor((255 - digits).reshape((-1, 1, 28, 28))).to(device)
    tensor_y = torch.LongTensor(y.reshape((-1))).to(device)
    grid_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    grid_dataloader = torch.utils.data.DataLoader(grid_dataset, batch_size=batch_size_test, shuffle=False)
    grid_dataloader.dataset.tensors = tuple(t.to(device) for t in grid_dataloader.dataset.tensors)
    return grid_dataloader


def get_test_and_train_dataloader(digits, y, batch_size_test=1000, device='cpu', split_size=0.7):
    tensor_x = torch.Tensor((255 - digits).reshape((-1, 1, 28, 28)))
    tensor_y = torch.LongTensor(y.reshape((-1)))
    div_ = 81 * int(split_size * (len(tensor_y) // 81))
    train_x = tensor_x[:div_, ...]
    train_y = tensor_y[:div_, ...]
    test_x = tensor_x[div_:, ...]
    test_y = tensor_y[div_:, ...]
    test_grid_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_grid_dataloader = torch.utils.data.DataLoader(test_grid_dataset, batch_size=batch_size_test, shuffle=False)
    test_grid_dataloader.dataset.tensors = tuple(t.to(device) for t in test_grid_dataloader.dataset.tensors)

    train_grid_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_grid_dataloader = torch.utils.data.DataLoader(train_grid_dataset, batch_size=batch_size_test, shuffle=False)
    train_grid_dataloader.dataset.tensors = tuple(t.to(device) for t in train_grid_dataloader.dataset.tensors)

    return test_grid_dataloader, train_grid_dataloader
