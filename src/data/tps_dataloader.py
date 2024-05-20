# TPS dataset
import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
from src.data.tps_sampler import TPSRandomSampler
import cv2

def _get_smooth_step(n, b):
    x = torch.linspace(-1, 1, n)
    y = 0.5 + 0.5 * torch.tanh(x / b)
    
    return y

def _get_smooth_mask(h, w, margin, step, b = 0.4):
    step_up = _get_smooth_step(step, b)
    step_down = _get_smooth_step(step, -b)
    def create_strip(size):
        return torch.cat(
              [torch.zeros(margin, dtype=torch.float32),
               step_up,
               torch.ones(size - 2 * margin - 2 * step, dtype=torch.float32),
               step_down,
               torch.zeros(margin, dtype=torch.float32)], axis=0)
    mask_x = create_strip(w)
    mask_y = create_strip(h)
    mask2d = mask_y[:, None] * mask_x[None]
    
    return mask2d

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
        
        self.image_sizes = [image_size, image_size]

        rotsd = [0.0, 5.0]
        scalesd=[0.0, 0.1] 
        transsd=[0.1, 0.1]
        warpsd=[0.001, 0.005, 0.001, 0.01]

        self._target_sampler = TPSRandomSampler(
            self.image_sizes[1], self.image_sizes[0], rotsd=rotsd[0], scalesd=scalesd[0],
            transsd=transsd[0], warpsd=warpsd[:2], pad=False)
        self._source_sampler = TPSRandomSampler(
            self.image_sizes[1], self.image_sizes[0], rotsd=rotsd[1], scalesd=scalesd[1],
            transsd=transsd[1], warpsd=warpsd[2:], pad=False)
            
    def get_img_mask_pair(self, img):
        
        crop_percent = 0.8

        height, width = self.image_sizes

        final_sz = self.image_size
        resize_sz = np.round(final_sz / crop_percent).astype(np.int32)
        margin = np.round((resize_sz - final_sz) / 2.0).astype(np.int32)

        f_scale_y = resize_sz / final_sz

        image = cv2.resize(img, None, fx=f_scale_y, fy=f_scale_y, interpolation = cv2.INTER_LINEAR)
        image = image[margin:margin + final_sz, margin:margin + final_sz]

        mask = _get_smooth_mask(height, width, 10, 20)[:, :, None]

        return image, mask
    
    def _apply_tps(self, img):
        # input an image
        image, mask  = self.get_img_mask_pair(img)
        
        def target_warp(images):
            return self._target_sampler.forward_py(images)
        def source_warp(images):
            return self._source_sampler.forward_py(images)

        cat_image = np.concatenate([mask[np.newaxis,...], image[np.newaxis,...]], 3)
        shape = cat_image.shape

        future_image = target_warp(cat_image)
        image = source_warp(future_image)

        future_mask = future_image[0,..., 0:1]
        future_image = future_image[0,..., 1:].astype(np.uint8)
        mask = image[0,..., 0:1]
        image = image[0,..., 1:].astype(np.uint8)

        inputs = {'image': image, 'mask': mask, 'future_image': future_image, 'future_mask':future_mask}
        
        return inputs

    def transform(self, img, hflip=False):
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        img = tfs.functional.to_tensor(img)
        img = img * 2 - 1
        
        return img
    
    def __getitem__(self, index):
        img = self.imgs[index % self.size]
        
        inputs = self._apply_tps(img)
        
        img = inputs["future_image"] # use the future_image one and not use the mask for recon
        img = Image.fromarray(img)
        
        return self.transform(img, hflip = False)

    def __len__(self):
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