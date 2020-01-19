from utils.Load_data import get_img, get_weights_dict


data_dir = '../data/input'

ids, imgs, meta = get_img(data_dir)

print(meta.head())
