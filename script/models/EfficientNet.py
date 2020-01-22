from efficientnet_pytorch import EfficientNet
from torch import nn



class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.efn = EfficientNet.from_pretrained('efficientnet-b4', num_classes=256)

        # Grapheme_root_num:  168
        self.fc_g = nn.Linear(in_features=256, out_features=168)
        # Vowel_diacritic_num:  11
        self.fc_v = nn.Linear(in_features=256, out_features=11)
        # Consonant_diacritic_num:  7
        self.fc_c = nn.Linear(in_features=256, out_features=7)

    def forward(self, x):

        x = self.efn(x)

        g = self.fc_g(x)
        v = self.fc_v(x)
        c = self.fc_c(x)

        return g, v, c
