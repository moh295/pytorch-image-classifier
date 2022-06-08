import torch
import torch.nn as nn
import torch.nn.functional as F



class Net100x2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 60, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 48, 5)
        self.fc1 = nn.Linear(48 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        #print('x0',x.size())
        x = self.pool(F.relu(self.conv1(x)))
        #print('x1', x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print('x2', x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print('x3', x.size())
        x = F.relu(self.fc1(x))
        #print('x4', x.size())
        x = F.relu(self.fc2(x))
        #print('x5', x.size())
        x = self.fc3(x)
        #print('x6', x.size())
        return x
