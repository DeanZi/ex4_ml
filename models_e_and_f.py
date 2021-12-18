import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelType2(nn.Module):
    def __init__(self, image_size, model_letter):
        super(ModelType2, self).__init__()
        self.model_letter = model_letter
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        if self.model_letter == 'E':
            return self.forward_e(x)
        else:
            return self.forward_f(x)

    def forward_e(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        output = F.log_softmax(x, dim=1)
        return output

    def forward_f(self, x):
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        output = F.log_softmax(x, dim=1)
        return output
