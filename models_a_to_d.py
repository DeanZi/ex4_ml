import torch.nn as nn
import torch.nn.functional as F


class ModelType1(nn.Module):
    def __init__(self, image_size, use_dropout=False, use_batch_norm=False, batch_norm_before=False):
        super(ModelType1, self).__init__()
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
