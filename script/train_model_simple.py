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
from utils.Utils import seed_everything
from utils.lightning import LightningSystem, LightningSystem_2
from models.EfficientNet import Mymodel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from efficientnet_pytorch import EfficientNet


# Config  ################################################################
data_dir = '../data/input'
seed = 0
test_size = 0.1
batch_size = 128
num_epoch = 4
img_size = 224
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set Seed
seed_everything(seed)

# Model  ################################################################
net = Mymodel()

# Train - Lightning  ################################################################
output_path = '../lightning'
model = LightningSystem_2(net, data_dir, device, img_size, batch_size, test_size, lr)

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
