import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from utils.Utils import seed_everything, freeze_until
from utils.lightning import LightningSystem
from models.EfficientNet import Mymodel, Mymodel_2
from models.Resnet import Mymodel_resnet
from utils.Augmentation import ImageTransform, ImageTransform_M, ImageTransform_random_erase

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Config  ################################################################
data_dir = '../data/input'
seed = 0
test_size = 0.2
batch_size = 64
num_epoch = 150
img_size = 112
lr = 5e-4
overfit_pct = 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Set Seed
seed_everything(seed)

# Transform  ############################################################
transform = ImageTransform_random_erase(img_size)

# Model  ################################################################
net = Mymodel_2()

# Sanity Check
# for name, param in net.named_parameters():
#     print(name)

# FineTuning
# freeze_until(net, 'base._blocks.45._expand_conv.weight')

# Optimizer  ################################################################
optimizer = optim.Adam(net.parameters(), lr=lr)
# schedular = StepLR(optimizer, step_size=20, gamma=0.5)
# schedular = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4)
schedular = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3, threshold_mode='abs')

# Train - Lightning  ################################################################
output_path = '../lightning'
model = LightningSystem(net, data_dir, optimizer, schedular, transform, batch_size, test_size)

# Load Pretrained Weights
# weights = torch.load('../model/Efficientnet_b7_epoch_23.ckpt', map_location=device)
# model.load_state_dict(weights['state_dict'])

checkpoint_callback = ModelCheckpoint(filepath='../lightning/ckpt_2', save_weights_only=True, monitor='avg_val_loss')
earlystopping = EarlyStopping(monitor='avg_val_loss', min_delta=0.0, patience=20)


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
