from torch.utils.data import Dataset, DataLoader


class BengariDataset(Dataset):

    def __init__(self, ids, imgs, meta, transform=None, phase='train'):
        self.ids = ids
        self.imgs = imgs
        self.meta = meta
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        _id = self.ids[idx]
        _img = self.imgs[idx]

        if self.transform is not None:
            _img = self.transform(_img, self.phase)

        if self.phase == 'train' or self.phase == 'val':
            target_g = self.meta[self.meta['image_id'] == _id]['grapheme_root'].values[0]
            target_v = self.meta[self.meta['image_id'] == _id]['vowel_diacritic'].values[0]
            target_c = self.meta[self.meta['image_id'] == _id]['consonant_diacritic'].values[0]

            return _img, target_g, target_v, target_c, _id

        else:
            return _img, _id
