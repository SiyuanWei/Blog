from __future__ import print_function, division

from torch.utils.data import Dataset
from skimage import io
from input.preprocess import *
from input.DatasetConfig import CITYSCAPES


def iscar(label):
    if label.name is 'car':
        return 1
    else:
        return 0

id_to_carid={label.id:iscar(label) for label in CITYSCAPES.labels}


class BaseDataset(Dataset):

    def __init__(self, img_files, label_files, transform=None, num_classes=2, is_transform=False, ignore_lable=255, label_map=None):
        self.img_files = img_files
        self.label_files = label_files
        self.num_classes = num_classes
        self.is_transform = is_transform
        self.ignore_label = ignore_lable
        self.transform=transform
        if label_map:
            self.idmap_func = np.vectorize(label_map.get)
        else:
            self.idmap_func = None


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image,label = self.img_files[idx],self.label_files[idx]
        image,label = io.imread(image),io.imread(label)

        if self.idmap_func:
            label = self.idmap_func(label)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader
