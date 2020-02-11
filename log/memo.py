# Memo  ################################################################

# Version8
# Model: Resnet18
# Schedular: StepLR step_size=20, gamma=0.5
# batch_size = 128
# num_epoch = 100
# img_size = 112
# lr = 5e-4

# Version9
# Model: Efficientnet-b0
# Schedular: StepLR step_size=20, gamma=0.5
# batch_size = 128
# num_epoch = 100
# img_size = 112
# lr = 5e-4

# Version10
# Model: SeResNext(pretrained=True)
# Schedular: StepLR
# Fine Tuning: Full layer train
# lr=5e-4
# img_size 224
# Batch_size 32
# ckpt_3
# 途中でメモリエラー　並列で無理そう

# Version11
# Model: Efficientnet-b0
# Add Augmentation - RandomErase
# Schedular: StepLR step_size=20, gamma=0.5
# batch_size = 128
# num_epoch = 100
# img_size = 112
# lr = 5e-4


# Version12
# Model: Resnet18
# Add Augmentation - RandomErase
# Schedular: StepLR step_size=20, gamma=0.5
# batch_size = 128
# num_epoch = 100
# img_size = 112
# lr = 1e-3


# Version13
# Model: SeResnet50_32x4d
# Add Augmentation - RandomErase
# Schedular: StepLR step_size=20, gamma=0.5
# batch_size = 32
# num_epoch = 150
# img_size = 224
# lr = 1e-3
