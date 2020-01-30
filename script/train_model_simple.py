import torch
from utils.Utils import seed_everything, freeze_until
from utils.lightning import LightningSystem
from models.EfficientNet import Mymodel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Config  ################################################################
data_dir = '../data/input'
seed = 0
test_size = 0.2
batch_size = 128
num_epoch = 10
img_size = 224
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set Seed
seed_everything(seed)

# Model  ################################################################
net = Mymodel()

# FineTuning
freeze_until(net, 'base._blocks.28._expand_conv.weight')

# Train - Lightning  ################################################################
output_path = '../lightning'
model = LightningSystem(net, data_dir, device, img_size, batch_size, test_size, lr)

checkpoint_callback = ModelCheckpoint(filepath='../model', save_weights_only=True, monitor='avg_val_loss')
earlystopping = EarlyStopping(monitor='avg_val_loss', min_delta=0.0, patience=2)


trainer = Trainer(
    max_epochs=num_epoch,
    min_epochs=5,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=earlystopping,
    gpus=[0]
)

if __name__ == '__main__':
    trainer.fit(model)
