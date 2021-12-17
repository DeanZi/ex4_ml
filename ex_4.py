import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def normalize_data(train_x, test_x):
    def normalize(train_x):
        train_x = train_x / 255
        return train_x

    return normalize(train_x), normalize(test_x)


def data_to_tensors(train_x, train_y, test_x):
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    test_x = torch.from_numpy(test_x)
    return train_x, train_y, test_x


def receive_data(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x, max_rows=10)
    train_y = np.loadtxt(train_y, max_rows=10)
    test_x = np.loadtxt(test_x, max_rows=10)
    return train_x, train_y, test_x


class Model_A_to_D(nn.Module):
    def __init__(self, image_size, use_dropout=False, use_batch_norm=False, batch_norm_before=False):
        super(Model_A_to_D, self).__init__()
        self.image_size = image_size
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.batch_norm_before = batch_norm_before

        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

        if self.use_dropout:
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        if self.use_batch_norm:
            self.fc0_bn = nn.BatchNorm1d(100)
            self.fc1_bn = nn.BatchNorm1d(50)
            self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        if not self.use_batch_norm:
            x = self.normal_forward(x)
        elif self.use_batch_norm and self.batch_norm_before:
            x = self.forward_with_batch_norm_before(x)
        elif self.use_batch_norm and not self.batch_norm_before:
            x = self.forward_with_batch_norm_after(x)
        output = F.log_softmax(x, dim=1)
        return output

    def normal_forward(self, x):
        x = F.relu(self.fc0(x))
        if self.use_dropout:
            x = self.dropout1(x)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        return x

    def forward_with_batch_norm_before(self, x):
        x = F.relu(self.fc0_bn(self.fc0(x)))
        if self.use_dropout:
            x = self.dropout1(x)
            x = F.relu(self.fc1_bn(self.fc1(x)))
            x = self.dropout2(x)
            x = F.relu(self.fc2_bn(self.fc2(x)))
        else:
            x = F.relu(self.fc1_bn(self.fc1(x)))
            x = F.relu(self.fc2_bn(self.fc2(x)))
        return x

    def forward_with_batch_norm_after(self, x):
        x = F.relu(self.fc0(x))
        x = self.fc0_bn(x)
        if self.use_dropout:
            x = self.dropout1(x)
            x = F.relu(self.fc1(x))
            x = self.fc1_bn(x)
            x = self.dropout2(x)
            x = F.relu(self.fc2(x))
            x = self.fc2_bn(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.fc1_bn(x)
            x = F.relu(self.fc2(x))
            x = self.fc2_bn(x)

        return x


def train(model, train_loader, optimizer, epoch):
    for e in range(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def model_A(lr=0.01):
    model_A = Model_A_to_D(image_size=28 * 28)
    optimizer = optim.SGD(model_A.parameters(), lr=lr)
    train(model_A, train_loader, optimizer, 10)


def model_B(lr=0.01, use_dropout=False, use_batch_norm=False, batch_norm_before=False):
    model_B = Model_A_to_D(image_size=28 * 28, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                           batch_norm_before=batch_norm_before)
    optimizer = optim.Adam(model_B.parameters(), lr=lr)
    train(model_B, train_loader, optimizer, 10)


def model_C(lr=0.01):
    model_B(lr, use_dropout=True)


def model_D(lr=0.01):
    print("batch norm before")
    model_B(lr, use_batch_norm=True, batch_norm_before=True)
    print("batch norm after")
    model_B(lr, use_batch_norm=True, batch_norm_before=False)



if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x, test_x = normalize_data(train_x, test_x)
    train_x, train_y, test_x = data_to_tensors(train_x, train_y, test_x)
    data_set_train = TensorDataset(train_x, train_y)
    train_loader = DataLoader(data_set_train, shuffle=True, batch_size=5)
    print('A')
    model_A()
    print('B')
    model_B()
    print('C')
    model_C()
    print('D')
    model_D()
