a = [3, 5, 6]

import numpy as np
import pandas as pd
import os

data_dir = '../data/input'
meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Wrong Train Label
print(meta.shape)
wrong_train = ['Train_49823', 'Train_2819', 'Train_20689']
meta = meta[~meta['image_id'].isin(wrong_train)]
print(meta.shape)