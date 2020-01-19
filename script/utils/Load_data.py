import glob, os, gc
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_img(data_dir, phase='train', seed=0, frac=1.0):
    dir_list = glob.glob(os.path.join(data_dir, f'{phase}*.parquet'))
    HEIGHT = 137
    WIDTH = 236
    ids = None
    imgs = None

    # Load Image
    for i in tqdm(range(len(dir_list))):
        temp = pd.read_parquet(dir_list[i])
        # Shuffle
        temp = temp.sample(frac=frac, random_state=seed)
        _ids = temp.iloc[:, 0].values
        _imgs = temp.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

        del temp
        gc.collect()

        if i == 0:
            ids = _ids
            imgs = _imgs
        else:
            ids = np.concatenate([ids, _ids])
            imgs = np.concatenate([imgs, _imgs])

    # Load Metadata
    meta = pd.read_csv(os.path.join(data_dir, f'{phase}.csv'))

    return ids, imgs, meta


def get_weights_dict(meta):
    counter_g = dict(Counter(meta['grapheme_root'].tolist()))
    counter_g = sorted(counter_g.items(), key=lambda x: x[0])
    weights_g = [len(meta) / w for (_, w) in counter_g]

    counter_v = dict(Counter(meta['vowel_diacritic'].tolist()))
    counter_v = sorted(counter_v.items(), key=lambda x: x[0])
    weights_v = [len(meta) / w for (_, w) in counter_v]

    counter_c = dict(Counter(meta['consonant_diacritic'].tolist()))
    counter_c = sorted(counter_c.items(), key=lambda x: x[0])
    weights_c = [len(meta) / w for (_, w) in counter_c]

    weights_dict = {
        'g': weights_g,
        'v': weights_v,
        'c': weights_c
    }

    return weights_dict
