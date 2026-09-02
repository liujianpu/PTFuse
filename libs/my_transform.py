import random
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose(T.Compose):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1 ,img2):
        for t in self.transforms:
            img1 ,img2 = t(img1 ,img2)
        return img1 ,img2


class Resize(T.Resize):
    def __init__(self, size):
        self.size = size

    def __call__(self, img1 ,img2):
        img1 = F.resize(img1, (self.size , self.size ))
        img2 = F.resize(img2, (self.size , self.size ))

        return img1 ,img2


class Resize_16(T.Resize):
    def __init__(self):
        pass

    def __call__(self, img1 ,img2):
        width, height = img1.size

        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        img1 = F.resize(img1, (new_height, new_width))
        img2 = F.resize(img2, (new_height, new_width))

        return img1 ,img2
    
class Resize_32(T.Resize):
    def __init__(self):
        pass

    def __call__(self, img1 ,img2):
        width, height = img1.size

        new_width = (width // 32) * 32
        new_height = (height // 32) * 32

        img1 = F.resize(img1, (new_height, new_width))
        img2 = F.resize(img2, (new_height, new_width))

        return img1 ,img2
    
class Resize_128(T.Resize):
    def __init__(self):
        pass

    def __call__(self, img1 ,img2):
        width, height = img1.size

        new_width = (width // 128) * 128
        new_height = (height // 128) * 128

        img1 = F.resize(img1, (new_height, new_width))
        img2 = F.resize(img2, (new_height, new_width))

        return img1 ,img2

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img1 ,img2):
        if random.random() < self.flip_prob:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        return img1 ,img2


class RandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img1 ,img2):
        if random.random() < self.flip_prob:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)
        return img1 ,img2


class RandomCrop(T.RandomCrop):
    def __init__(self, size):
        self.size = size

    def __call__(self, img1 ,img2):
        crop_params = T.RandomCrop.get_params(img1, (self.size, self.size))
        img1 = F.crop(img1, *crop_params)
        img2 = F.crop(img2, *crop_params)
        return img1 ,img2

class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        self.size = size

    def __call__(self, img1 ,img2):
        img1 = F.center_crop(img1, self.size)
        img2 = F.center_crop(img2, self.size)
        return img1 ,img2


class ToTensor(T.ToTensor):
    def __call__(self, img1 ,img2):
        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        return img1 ,img2
    

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img