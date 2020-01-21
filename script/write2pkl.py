import pickle, os, gc
from utils.Load_data import get_img, get_weights_dict
import matplotlib.pyplot as plt
from utils.Augmentation import ImageTransform

# Load Data
data_dir = '../data/input'
# ids, imgs, meta = get_img(data_dir)
# weights_dict = get_weights_dict(meta)
# print('Data Already')
#
# # Write to Pickle
# with open(os.path.join(data_dir, 'train.pkl'), 'wb') as f:
#     pickle.dump((ids, imgs), f, protocol=4)
#
# del ids, imgs, meta
# gc.collect()

# Load from Pickle
with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
    data = pickle.load(f)

print(data[0])
print(data[1][0].shape)

plt.imshow(data[1][0], cmap='gray')
plt.show()

img_ = ImageTransform(64)(data[1][0], 'train')

print(img_)
print(img_.size())
