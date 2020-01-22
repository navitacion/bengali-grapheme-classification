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
from utils.lightning import LightningSystem

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Config  ################################################################
data_dir = '../data/input'
use_pickle = True
seed = 0
test_size = 0.1
batch_size = 128
num_epoch = 4
img_size = 224
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'mynet'
version = model_name + '_000'

# Load Data  ################################################################
# from pickle
if use_pickle:
    with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
        data = pickle.load(f)
    ids = data[0]
    imgs = data[1]
    del data
    gc.collect()

else:
    # from perquet
    ids, imgs, meta = get_img(data_dir)

# MetaData
meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))
# get class weights_dict
weights_dict = get_weights_dict(meta)

print('Data Already')

# Split Train, Valid Data  ################################################################
train_ids, train_imgs = ids[int(len(ids) * test_size):], imgs[int(len(ids) * test_size):]
val_ids, val_imgs = ids[:int(len(ids) * test_size)], imgs[:int(len(ids) * test_size)]

del ids, imgs
gc.collect()

# Dataset  ################################################################
train_dataset = BengariDataset(train_ids, train_imgs, meta, ImageTransform(img_size), phase='train')
val_dataset = BengariDataset(val_ids, val_imgs, meta, ImageTransform(img_size), phase='val')

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

# Model  ################################################################
net = MyNet()
optimizer = optim.Adam(params=net.parameters(), lr=lr)

# Train - Lightning  ################################################################
output_path = '../lightning'
model = LightningSystem(net, dataloader_dict, weights_dict, optimizer, device)

checkpoint_callback = ModelCheckpoint(filepath='../model', save_weights_only=True, monitor='avg_val_loss')
earlystopping = EarlyStopping(monitor='avg_val_loss', min_delta=0.0, patience=2)


trainer = Trainer(
    max_nb_epochs=num_epoch,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=earlystopping
    # gpus=[0]
)

trainer.fit(model)
