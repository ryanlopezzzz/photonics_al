from torch import nn
import torch.nn.functional as F
import torch
from bayesian_layer import AnalyticLinear

class RegressionAnalyticBNN(nn.Module):
    def __init__(self):
        super(RegressionAnalyticBNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = AnalyticLinear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output, kl, output_variance = self.fc2(x)
        output = torch.squeeze(output)
        output_variance = torch.squeeze(output_variance)
        return output, kl, output_variance