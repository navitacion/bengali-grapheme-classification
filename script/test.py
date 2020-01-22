import gc, os, pickle
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from utils.Dataset import BengariDataset
from utils.Load_data import get_img, get_weights_dict
from models.Original import MyNet
from utils.Augmentation import ImageTransform
from utils.Trainer import train_model
from utils.logger import create_logger, get_logger
from utils.lightning import CoolSystem

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


data_dir = '../data/input'
# from pickle
with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
    data = pickle.load(f)
ids = data[0]
imgs = data[1]
del data
gc.collect()
meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))

print('Data Already')

# get class weights_dict
weights_dict = get_weights_dict(meta)

# Config
seed = 0
test_size = 0.1
batch_size = 128
num_epoch = 2
img_size = 224
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'mynet'
version = model_name + '_000'


# Split Train, Valid Data
train_ids, train_imgs = ids[int(len(ids) * test_size):], imgs[int(len(ids) * test_size):]
val_ids, val_imgs = ids[:int(len(ids) * test_size)], imgs[:int(len(ids) * test_size)]

del ids, imgs
gc.collect()

# Dataset
train_dataset = BengariDataset(train_ids, train_imgs, meta, ImageTransform(img_size), phase='train')
val_dataset = BengariDataset(val_ids, val_imgs, meta, ImageTransform(img_size), phase='val')

# Dataloader
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# dataloader_dict = {
#     'train': train_dataloader,
#     'val': val_dataloader
# }

for i in range(5):
    _img, target_g, target_v, target_c, _id = train_dataset.__getitem__(i)
    _img[_img < 0] = 0
    plt.imshow(_img.squeeze().numpy(), cmap='gray')
    plt.show()

print(_img)
print(_img.max())
print(_img.min())

