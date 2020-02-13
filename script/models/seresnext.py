import pretrainedmodels
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential


class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext50_32x4d',
                 in_channels=1, pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

        self.block = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Grapheme_root_num:  168
        self.fc_g = nn.Linear(in_features=256, out_features=168)
        # Vowel_diacritic_num:  11
        self.fc_v = nn.Linear(in_features=256, out_features=11)
        # Consonant_diacritic_num:  7
        self.fc_c = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):
        x = self.conv0(x)
        x = self.base.features(x)

        x = self.pool(x)
        x = x.squeeze()

        x = self.block(x)

        g = self.fc_g(x)
        v = self.fc_v(x)
        c = self.fc_c(x)

        return g, v, c
