from models.senet import se_resnext50_32x4d
import torch
import os, pickle

data_dir = '../data/input'
# Load Data
with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
    data = pickle.load(f)
ids, imgs = data

print(imgs[0])
print(imgs[0].shape)