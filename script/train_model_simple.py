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
batch_size = 64
num_epoch = 100
img_size = 224
lr = 1e-3
overfit_pct = 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set Seed
seed_everything(seed)

# Model  ################################################################
net = Mymodel()

# Sanity Check
# for name, param in net.named_parameters():
#     print(name)

# FineTuning
freeze_until(net, 'base._blocks.45._expand_conv.weight')

# Train - Lightning  ################################################################
output_path = '../lightning'
model = LightningSystem(net, data_dir, device, img_size, batch_size, test_size, lr)

# Load Pretrained Weights
weights = torch.load('../model/Efficientnet_b7_epoch_23.ckpt', map_location=device)
model.load_state_dict(weights['state_dict'])

checkpoint_callback = ModelCheckpoint(filepath='../lightning/ckpt', save_weights_only=True, monitor='avg_val_loss')
earlystopping = EarlyStopping(monitor='avg_val_loss', min_delta=0.0, patience=8)


trainer = Trainer(
    max_epochs=num_epoch,
    min_epochs=5,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=earlystopping,
    overfit_pct=overfit_pct,
    gpus=[0]
)

if __name__ == '__main__':
    trainer.fit(model)
