from __future__ import print_function, division

import numpy as np
from skimage import transform
from torch.distributions.one_hot_categorical import *


class OneHotLabel(object):

    def  __init__(self,num_cls):
        assert isinstance(num_cls,int)
        self.num_cls = num_cls

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        assert isinstance(label,torch.Tensor)
        shape = label.shape
        label = label.reshape(shape[0],shape[1])

        onehot = torch.LongTensor(self.num_cls,shape[0],shape[1]).zero_()
        for i in range(shape[0]):
            for j in range(shape[1]):
                onehot[label[i][j], i, j] = 1
        return {'image': image, 'label': onehot}



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, factor):
        assert isinstance(factor, float)
        self.factor = factor

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        image = transform.rescale(image,self.factor)
        label = transform.rescale(label,self.factor)
        return {'image': image, 'label': label}



class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        new_h, new_w = self.output_size

        #skimage transform resize,The order of interpolation.
        # The order has to be in the range 0-5:
        # * 0: Nearest-neighbor * 1: Bi-linear (default) * 2: Bi-quadratic * 3: Bi-cubic * 4: Bi-quartic * 5: Bi-quintic
        image = transform.resize(image, (new_h, new_w),order=1)
        label = transform.resize(label, (new_h, new_w),order=0)


        return {'image': image, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        assert (new_h<h) and (new_w<w),'random crop size should be smaller than image size'

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]


        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}