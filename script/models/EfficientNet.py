from efficientnet_pytorch import EfficientNet
from torch import nn
import torch


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.base = EfficientNet.from_pretrained('efficientnet-b7', num_classes=256)

        # Grapheme_root_num:  168
        self.fc_g = nn.Linear(in_features=256, out_features=168)
        # Vowel_diacritic_num:  11
        self.fc_v = nn.Linear(in_features=256, out_features=11)
        # Consonant_diacritic_num:  7
        self.fc_c = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):

        x = self.base(x)

        g = self.fc_g(x)
        v = self.fc_v(x)
        c = self.fc_c(x)

        return g, v, c


class Mymodel_2(nn.Module):
    def __init__(self):
        super(Mymodel_2, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

        self.base = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1000)

        self.block = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )

        # Grapheme_root_num:  168
        self.fc_g = nn.Linear(in_features=256, out_features=168)
        # Vowel_diacritic_num:  11
        self.fc_v = nn.Linear(in_features=256, out_features=11)
        # Consonant_diacritic_num:  7
        self.fc_c = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):
        x = self.conv0(x)

        x = self.base(x)
        x = self.block(x)

        g = self.fc_g(x)
        v = self.fc_v(x)
        c = self.fc_c(x)

        return g, v, c

