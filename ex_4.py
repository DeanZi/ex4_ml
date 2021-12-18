import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
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


def train(model, optimizer, epoch, validate, min_validation_loss=np.inf):
    for e in range(epoch):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            e + 1, train_loss / len(train_loader)))

        if validate:
            validation_loss = 0.0
            model.eval()
            for data, labels in validation_loader:
                target = model(data)
                loss = F.nll_loss(target, labels)
                validation_loss += loss.item()

            print(f'Validation Epoch {e + 1} \t\t Validation Loss: { validation_loss / len(validation_loader)}')

            if min_validation_loss > validation_loss:
                min_validation_loss = validation_loss
                torch.save(model.state_dict(), 'saved_model.pth')


def model_a(lr=0.01, validate=True):
    model = ModelType1(image_size=28 * 28)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, optimizer, 10, validate)


def model_b(lr=0.01, use_dropout=False, use_batch_norm=False, batch_norm_before=False, validate=True):
    model = ModelType1(image_size=28 * 28, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                       batch_norm_before=batch_norm_before)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, validate)


def model_c(lr=0.01, validate=True):
    model_b(lr, use_dropout=True, validate=validate)


def model_d(lr=0.01, validate=True):
    print("batch norm before")
    model_b(lr, use_batch_norm=True, batch_norm_before=True, validate=validate)
    print("batch norm after")
    model_b(lr, use_batch_norm=True, batch_norm_before=False, validate=validate)


def model_e(lr=0.01, validate=True):
    model = ModelType2(image_size=28 * 28, model_letter='E')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, validate)


def model_f(lr=0.01, validate=True):
    model = ModelType2(image_size=28 * 28, model_letter='F')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, validate)


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x, test_x = normalize_data(train_x, test_x)
    train_x, train_y, test_x = data_to_tensors(train_x, train_y, test_x)
    data_set_train = TensorDataset(train_x, train_y)
    train_set, validation_set = random_split(data_set_train, [8, 2])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    validation_loader = DataLoader(validation_set)
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
