import pickle, os, gc
from utils.Load_data import get_img


def write2pkl(data_dir, output_dir):
    # Load Data
    ids, imgs, meta = get_img(data_dir)
    print('Data Already')

    # Write to Pickle
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump((ids, imgs), f, protocol=4)

    del ids, imgs, meta
    gc.collect()


if __name__ == '__main__':
    write2pkl(data_dir='../data/Raw', output_dir='../data/input')
