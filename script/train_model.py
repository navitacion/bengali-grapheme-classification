import gc, os, pickle
import pandas as pd

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
# Load Data
# from perquet
# ids, imgs, meta = get_img(data_dir)

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
img_size = 64
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
train_dataset = BengariDataset(train_ids, train_imgs, meta, ImageTransform(img_size))
val_dataset = BengariDataset(val_ids, val_imgs, meta, ImageTransform(img_size))

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

# Model
net = MyNet()
optimizer = optim.Adam(params=net.parameters(), lr=lr)


# # logging  ################################################################
# create_logger(version)
# get_logger(version).info('------- Config ------')
# get_logger(version).info(f'Random Seed: {seed}')
# get_logger(version).info(f'Batch Size: {batch_size}')
# get_logger(version).info(f'Optimizer: {optimizer.__class__.__name__}')
# get_logger(version).info(f'Learning Rate: {lr}')
# # get_logger(version).info(f'Update Params: {update_params_name}')
# get_logger(version).info('------- Train Start ------')
#
# net, best_loss, df_loss = train_model(net, dataloader_dict, weights_dict,
#                                       optimizer, device, num_epoch, model_name, version)


# Lightning
output_path = '../lightning'
model = CoolSystem(net, dataloader_dict, weights_dict, optimizer, device)

checkpoint_callback = ModelCheckpoint(filepath='./model', save_best_only=True)


trainer = Trainer(
    max_nb_epochs=num_epoch,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback
    # gpus=[0]
)

trainer.fit(model)
