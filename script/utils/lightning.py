import torch
from torch import nn

from sklearn.metrics import recall_score
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class LightningSystem(pl.LightningModule):

    def __init__(self, net, dataloader_dict, weights_dict, optimizer, device):
        super(LightningSystem, self).__init__()

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
        # Loss
        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c

        # Recall Score
        scores = []
        pred_g = torch.softmax(outputs_g, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_g.tolist(), pred_g, average='macro'))
        pred_v = torch.softmax(outputs_v, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_v.tolist(), pred_v, average='macro'))
        pred_c = torch.softmax(outputs_c, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_c.tolist(), pred_c, average='macro'))

        scores = np.average(scores, weights=[2, 1, 1])
        scores = torch.tensor(scores)

        logs = {'train_loss': loss, 'train_recall': scores}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        x, target_g, target_v, target_c, _ = batch
        outputs_g, outputs_v, outputs_c = self.forward(x)
        # Loss
        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c

        # Recall Score
        scores = []
        pred_g = torch.softmax(outputs_g, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_g.tolist(), pred_g, average='macro'))
        pred_v = torch.softmax(outputs_v, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_v.tolist(), pred_v, average='macro'))
        pred_c = torch.softmax(outputs_c, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_c.tolist(), pred_c, average='macro'))

        scores = np.average(scores, weights=[2, 1, 1])
        scores = torch.tensor(scores)

        return {'val_loss': loss, 'val_recall': scores}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_recall': avg_recall}

        return {'avg_val_loss': avg_loss, 'avg_val_recall': avg_recall, 'log': logs}

    def configure_optimizers(self):
        return [self.optimizer], []

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader_dict['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader_dict['val']

