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
                RandomRotate(),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'val': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                transforms.ToTensor(),
                PostProcess()
            ]),
            'test': transforms.Compose([
                Reverse(),
                MedianFilter(),
                CropResize(resize),
                transforms.ToTensor(),
                PostProcess()
            ]),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
