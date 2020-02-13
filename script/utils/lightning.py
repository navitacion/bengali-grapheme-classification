import os, pickle, gc
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import recall_score, accuracy_score
import numpy as np

import pytorch_lightning as pl
from .Dataset import BengariDataset


# データ読み込みや前処理もすべてまとめる
# こちらの方が学習速度が早い
class LightningSystem(pl.LightningModule):

    def __init__(self, net, data_dir, optimizer, schedular, transform, batch_size=128, test_size=0.1):
        super(LightningSystem, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.schedular = schedular
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

        # Load Data
        with open(os.path.join(self.data_dir, 'train.pkl'), 'rb') as f:
            data = pickle.load(f)
        ids, imgs = data
        del data
        gc.collect()

        meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))

        # Remove Wrong Train Label
        wrong_train = ['Train_49823', 'Train_2819', 'Train_20689']
        meta = meta[~meta['image_id'].isin(wrong_train)]

        # ids, imgs, meta = get_img(data_dir)

        # Split Train, Valid Data  ################################################################
        train_ids, train_imgs = ids[int(len(ids) * test_size):], imgs[int(len(ids) * test_size):]
        val_ids, val_imgs = ids[:int(len(ids) * test_size)], imgs[:int(len(ids) * test_size)]

        del ids, imgs
        gc.collect()

        # Dataset  ################################################################
        self.train_dataset = BengariDataset(train_ids, train_imgs, meta,
                                            self.transform, phase='train')
        self.val_dataset = BengariDataset(val_ids, val_imgs, meta,
                                          self.transform, phase='val')

        # Criterion  ################################################################
        self.criterion_g = nn.CrossEntropyLoss()
        self.criterion_v = nn.CrossEntropyLoss()
        self.criterion_c = nn.CrossEntropyLoss()

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
        return [self.optimizer], [self.schedular]

    def training_step(self, batch, batch_idx):
        x, target_g, target_v, target_c, _ = batch
        outputs_g, outputs_v, outputs_c = self.forward(x)
        # Loss  ################################################################
        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c

        # Recall Score  ################################################################
        scores = []
        pred_g = torch.softmax(outputs_g, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_g.tolist(), pred_g, average='macro'))
        pred_v = torch.softmax(outputs_v, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_v.tolist(), pred_v, average='macro'))
        pred_c = torch.softmax(outputs_c, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_c.tolist(), pred_c, average='macro'))

        scores = np.average(scores, weights=[2, 1, 1])
        scores = torch.tensor(scores)

        # Accuracy  ################################################################
        acc = []
        acc.append(accuracy_score(target_g.tolist(), pred_g))
        acc.append(accuracy_score(target_v.tolist(), pred_v))
        acc.append(accuracy_score(target_c.tolist(), pred_c))

        acc = np.average(acc)
        acc = torch.tensor(acc)

        logs = {'train/loss': loss, 'train/recall': scores, 'train/acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        x, target_g, target_v, target_c, _ = batch
        outputs_g, outputs_v, outputs_c = self.forward(x)
        # Loss  ################################################################
        loss_g = self.criterion_g(outputs_g, target_g.long())
        loss_v = self.criterion_v(outputs_v, target_v.long())
        loss_c = self.criterion_c(outputs_c, target_c.long())

        loss = loss_g + loss_v + loss_c

        # Recall Score  ################################################################
        scores = []
        pred_g = torch.softmax(outputs_g, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_g.tolist(), pred_g, average='macro'))
        pred_v = torch.softmax(outputs_v, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_v.tolist(), pred_v, average='macro'))
        pred_c = torch.softmax(outputs_c, dim=1).argmax(dim=1).tolist()
        scores.append(recall_score(target_c.tolist(), pred_c, average='macro'))

        scores = np.average(scores, weights=[2, 1, 1])
        scores = torch.tensor(scores)

        # Accuracy  ################################################################
        acc = []
        acc.append(accuracy_score(target_g.tolist(), pred_g))
        acc.append(accuracy_score(target_v.tolist(), pred_v))
        acc.append(accuracy_score(target_c.tolist(), pred_c))

        acc = np.average(acc)
        acc = torch.tensor(acc)

        return {'val_loss': loss, 'val_recall': scores, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val/loss': avg_loss, 'val/recall': avg_recall, 'val/acc': avg_acc}

        return {'avg_val_loss': avg_loss, 'avg_val_recall': avg_recall, 'log': logs}
