import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class CoolSystem(pl.LightningModule):

    def __init__(self, net, dataloader_dict, weights_dict, optimizer, device):
        super(CoolSystem, self).__init__()

        self.net = net
        self.dataloader_dict = dataloader_dict

        self.criterion_g = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['g']).to(device))
        self.criterion_v = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['v']).to(device))
        self.criterion_c = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['c']).to(device))
        self.optimizer = optimizer

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, target_g, target_v, target_c, _ = batch
        outputs_g, outputs_v, outputs_c = self.forward(x)

        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        x, target_g, target_v, target_c, _ = batch
        outputs_g, outputs_v, outputs_c = self.forward(x)

        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        return [self.optimizer]


    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader_dict['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader_dict['val']

