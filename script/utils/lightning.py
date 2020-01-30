import os, pickle, gc
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import recall_score
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .Load_data import get_weights_dict
from .Dataset import BengariDataset
from .Augmentation import ImageTransform
from .Load_data import get_img


# データ読み込みや前処理もすべてまとめる
# こちらの方が学習速度が早い
class LightningSystem(pl.LightningModule):

    def __init__(self, net, data_dir, device, img_size=224, batch_size=128, test_size=0.1, lr=1e-3):
        super(LightningSystem, self).__init__()
        self.net = net
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        # Load Data
        # with open(os.path.join(self.data_dir, 'train.pkl'), 'rb') as f:
        #     data = pickle.load(f)
        # ids, imgs = data
        # del data
        # gc.collect()
        #
        # meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))

        ids, imgs, meta = get_img(data_dir)

        # Split Train, Valid Data  ################################################################
        train_ids, train_imgs = ids[int(len(ids) * test_size):], imgs[int(len(ids) * test_size):]
        val_ids, val_imgs = ids[:int(len(ids) * test_size)], imgs[:int(len(ids) * test_size)]

        del ids, imgs
        gc.collect()

        # Dataset  ################################################################
        self.train_dataset = BengariDataset(train_ids, train_imgs, meta,
                                            ImageTransform(self.img_size), phase='train')
        self.val_dataset = BengariDataset(val_ids, val_imgs, meta,
                                          ImageTransform(self.img_size), phase='val')

        # Criterion  ################################################################
        weights_dict = get_weights_dict(meta)

        self.criterion_g = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['g']).to(device))
        self.criterion_v = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['v']).to(device))
        self.criterion_c = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['c']).to(device))

        # Fine Tuning  ###############################################################

        # Setting Optimizer  ################################################################
        self.optimizer = optim.Adam(net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        # Set [optimizer], [schedular]
        return [self.optimizer], []

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

