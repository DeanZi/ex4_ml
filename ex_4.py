import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from models_a_to_d import ModelType1
from models_e_and_f import ModelType2
import matplotlib.pyplot as plt


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
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    test_x = np.loadtxt(test_x, max_rows=10)
    return train_x, train_y, test_x


def plot_model_loss(train_losses, validation_losses, epochs_list, model):
    plt.plot(epochs_list, train_losses, color='r', label='train')
    plt.plot(epochs_list, validation_losses, color='g', label='validation')
    plt.ylabel('Average loss')
    plt.xlabel('Epoch')
    plt.title(f"Loss per epoch (model {model.model_letter})")
    plt.legend()
    plt.show()


def plot_model_acc(train_accuracies, validation_accuracies, epochs_list, model):
    plt.plot(epochs_list, train_accuracies, color='r', label='train')
    plt.plot(epochs_list, validation_accuracies, color='g', label='validation')
    plt.ylabel('Average accuracy')
    plt.xlabel('Epoch')
    plt.title(f"Accuracy per epoch (model {model.model_letter})")
    plt.legend()
    plt.show()


def validate(model, validation_losses, validation_accuracies, train_accuracies, e, train_set_validation=False):
    validation_loss = 0.0
    correct_predictions = 0
    model.eval()
    if train_set_validation:
        data = train_loader
    else:
        data = validation_loader
    for data, labels in data:
        target = model(data)
        loss = F.nll_loss(target, labels)
        validation_loss += loss.item()
        _, prediction = target.max(1)
        correct_predictions += (prediction == labels).sum()
    if not train_set_validation:
        print(
            '\nValidation set: Accuracy: {:.0f}%\n'.format(100. * correct_predictions / len(validation_loader.dataset)))
        print(f'Validation Epoch {e + 1} \t\t Validation Loss: {validation_loss / len(validation_loader)}')
        validation_losses.append(validation_loss / len(validation_loader))
        validation_accuracies.append(100. * correct_predictions / len(validation_loader.dataset))
    else:
        train_accuracies.append(100. * correct_predictions / len(train_loader.dataset))


def train(model, optimizer, epoch, with_validation):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
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
        train_losses.append(train_loss / len(train_loader))

        if with_validation:
            validate(model, validation_losses, validation_accuracies, train_accuracies, e)
            validate(model, validation_losses, validation_accuracies, train_accuracies, e, train_set_validation=True)

    plot_model_loss(train_losses, validation_losses, [i for i in range(1, 10 + 1)], model)
    plot_model_acc(train_accuracies, validation_accuracies, [i for i in range(1, 10 + 1)], model)


def model_a(lr=0.01, with_validation=True):
    model = ModelType1(image_size=28 * 28, model_letter='A')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, optimizer, 10, with_validation)


def model_b(lr=0.01, use_dropout=False, use_batch_norm=False, batch_norm_before=False, with_validation=True, model_letter='B'):
    model = ModelType1(image_size=28 * 28, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                       batch_norm_before=batch_norm_before, model_letter=model_letter)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, with_validation)


def model_c(lr=0.01, with_validation=True):
    model_b(lr, use_dropout=True, with_validation=with_validation, model_letter='C')


def model_d(lr=0.01, with_validation=True):
    print("batch norm before")
    model_b(lr, use_batch_norm=True, batch_norm_before=True, with_validation=with_validation, model_letter='D')
    print("batch norm after")
    model_b(lr, use_batch_norm=True, batch_norm_before=False, with_validation=with_validation, model_letter='D')


def model_e(lr=0.01, with_validation=True):
    model = ModelType2(image_size=28 * 28, model_letter='E')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, with_validation)


def model_f(lr=0.01, with_validation=True):
    model = ModelType2(image_size=28 * 28, model_letter='F')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, 10, with_validation)


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x, test_x = normalize_data(train_x, test_x)
    train_x, train_y, test_x = data_to_tensors(train_x, train_y, test_x)
    data_set_train = TensorDataset(train_x, train_y)
    train_set, validation_set = random_split(data_set_train, [44000, 11000])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False)
    validation_loader = DataLoader(validation_set)
    # print('A')
    # model_a()
    # print('B')
    # train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    # model_b()
    # print('C')
    # train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    # model_c()
    # print('D')
    # train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    # model_d()
    # print('E')
    # train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    # model_e()
    # print('F')
    # train_loader = DataLoader(train_set, batch_size=512, shuffle=False)
    # model_f()
