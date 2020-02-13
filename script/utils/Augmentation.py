import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Global Var
HEIGHT = 137
WIDTH = 236

# Somehow the original input is inversed
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class Reverse:
    def __init__(self):
        pass

    def __call__(self, image):
        image = 255 - image.astype(np.uint8)
        image = (image * (255.0/image.max())).astype(np.uint8)

        return image


# Remove Noise
class MedianFilter:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, image):
        return cv2.medianBlur(image, self.kernel_size)


# https://www.kaggle.com/iafoss/image-preprocessing-128x128
class CropResize:
    def __init__(self, size, pad=16):
        self.size = size
        self.pad = pad

    def __call__(self, image):
        # crop a box around pixels large than the threshold
        # some images contain line at the sides
        image = image.astype(np.uint8)
        ymin, ymax, xmin, xmax = bbox(image[5:-5, 5:-5] > 80)

        # cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
        ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
        img = image[ymin:ymax, xmin:xmax]

        # remove lo intensity pixels as noise
        img[img < 28] = 0
        lx, ly = xmax - xmin, ymax - ymin
        l = max(lx, ly) + self.pad

        # make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l-ly) // 2,), ((l-lx) // 2,)], mode='constant')

        return cv2.resize(img, (self.size, self.size))


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel


class MorphologicalOpening:
    def __init__(self):
        pass

    def __call__(self, image):
        # Opening
        image = cv2.erode(image, get_random_kernel(), iterations=1)
        image = cv2.dilate(image, get_random_kernel(), iterations=1)

        return image


class MorphologicalClosing:
    def __init__(self):
        pass

    def __call__(self, image):
        # Closing
        image = cv2.dilate(image, get_random_kernel(), iterations=1)
        image = cv2.erode(image, get_random_kernel(), iterations=1)

        return image


class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, image):
        r = random.choice([0, 1, -1, 999])
        if r != 999:
            image = cv2.flip(image, r)
        else:
            pass
        return image


class RandomRotate:
    def __init__(self):
        pass

    def __call__(self, image):
        r = random.choice([0, 1, 2, 3])

        # 時計回り
        if r == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # 反時計回り
        elif r == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 180°
        elif r == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        else:
            pass

        return image


class Rotate:
    def __init__(self):
        pass

    def __call__(self, image):
        angle = random.choice(np.arange(0, 90, 10).tolist())
        h, w = image.shape
        center = (int(w / 2), int(h / 2))
        trans = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(image, trans, (w, h))

        return image


# https://www.kumilog.net/entry/numpy-data-augmentation
class RandomErase:
    def __init__(self, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
        self.p = p
        self.s = s
        self.r = r

    def __call__(self, image):

        # マスクするかしないか
        if np.random.rand() > self.p:
            return image
        _image = np.copy(image)

        # マスクする画素値をランダムで決める
        mask_value = np.random.randint(0, 256)

        h, w = _image.shape
        # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
        mask_area = np.random.randint(h * w * self.s[0], h * w * self.s[1])

        # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
        mask_aspect_ratio = np.random.rand() * self.r[1] + self.r[0]

        # マスクのサイズとアスペクト比からマスクの高さと幅を決める
        # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        _image[top:bottom, left:right].fill(mask_value)
        return _image


class Gray2RGB:
    def __init__(self):
        pass

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


class PostProcess:
    def __init__(self):
        pass

    def __call__(self, image):
        image[image < 0] = 0

        return image


class ImageTransform:
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                RandomFlip(),
                Rotate(),
                RandomRotate(),
                RandomErase(),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'val': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'test': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)



class ImageTransform_M:
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                RandomFlip(),
                RandomRotate(),
                MorphologicalOpening(),
                MorphologicalClosing(),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'val': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'test': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class ImageTransform_random_erase:
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                RandomFlip(),
                RandomRotate(),
                RandomErase(),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'val': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'test': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                # Gray2RGB(),
                transforms.ToTensor(),
                PostProcess()
            ]),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
