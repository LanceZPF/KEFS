import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=64, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.imgs = np.load(data_dir)['imgs']
        print("load!!!!")
        self.size = self.imgs.shape[0]
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation
            
    def transform(self, img, hflip=False):
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        img = tfs.functional.to_tensor(img)
        img = img * 2 - 1
        
        return img

    def __getitem__(self, index):
        img = self.imgs[index % self.size]
        img = Image.fromarray(img)
        
        return self.transform(img, hflip = False)

    def __len__(self):
        if self.size < 500:
            return 50 * self.size
        else:
            return self.size

    def name(self):
        return 'ImageDataset'


def get_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=64, crop=None):

    dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        drop_last = True,
    )
    return loader