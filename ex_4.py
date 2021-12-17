import torch
import torch.nn as nn
import torch.nn.functional as F



class Model_A_to_D(nn.Module):
    def __init__(self, image_size=28*28):
        super(Model_A_to_D, self).__init__()
        self.fc1 = nn.Linear(image_size, 100)
        self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        x = F.relu(self.fc1)
        x = F.relu(self.fc2)
        output = F.log_softmax(x)
        return output

if __name__ == '__main__':


