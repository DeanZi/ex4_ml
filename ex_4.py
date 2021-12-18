import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from models_a_to_d import ModelType1
from models_e_and_f import ModelType2


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


def model_a(lr=0.01):
    model = ModelType1(image_size=28 * 28)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, 10)


def model_b(lr=0.01, use_dropout=False, use_batch_norm=False, batch_norm_before=False):
    model = ModelType1(image_size=28 * 28, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                       batch_norm_before=batch_norm_before)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, 10)


def model_c(lr=0.01):
    model_b(lr, use_dropout=True)


def model_d(lr=0.01):
    print("batch norm before")
    model_b(lr, use_batch_norm=True, batch_norm_before=True)
    print("batch norm after")
    model_b(lr, use_batch_norm=True, batch_norm_before=False)


def model_e(lr=0.01):
    model = ModelType2(image_size=28 * 28, model_letter='E')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, 10)


def model_f(lr=0.01):
    model = ModelType2(image_size=28 * 28, model_letter='F')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, 10)


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x, test_x = normalize_data(train_x, test_x)
    train_x, train_y, test_x = data_to_tensors(train_x, train_y, test_x)
    data_set_train = TensorDataset(train_x, train_y)
    train_loader = DataLoader(data_set_train, shuffle=True, batch_size=5)
    print('A')
    model_a()
    print('B')
    model_b()
    print('C')
    model_c()
    print('D')
    model_d()
    print('E')
    model_e()
    print('F')
    model_f()
