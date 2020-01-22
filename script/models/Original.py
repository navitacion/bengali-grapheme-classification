from torch import nn
import torch
from torchsummary import summary

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_features=64, out_features=256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)

        # Grapheme_root_num:  168
        self.fc_g = nn.Linear(in_features=256, out_features=168)
        # Vowel_diacritic_num:  11
        self.fc_v = nn.Linear(in_features=256, out_features=11)
        # Consonant_diacritic_num:  7
        self.fc_c = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pool(x)
        x = x.squeeze()

        x = self.fc(x)
        x = self.bn4(x)
        x = self.relu4(x)

        g = self.fc_g(x)
        v = self.fc_v(x)
        c = self.fc_c(x)

        return g, v, c
