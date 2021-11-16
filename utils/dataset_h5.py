from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', transforms=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        # agregado
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_nd, normalize=True, debug=False):
        
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        # if(debug):
        #     print('pil_img.mode', pil_img.mode)
        
        # img_nd = np.array(pil_img)
        
        if(debug):
            print('img_nd.shape', img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
           
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        
        #if normalize and img_trans.max() > 1:
        if img_trans.max() > 1:
            img_trans = img_trans / 255
            
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0]).convert('L')
        mask_h5 = h5py.File(mask_file[0], 'r')
        mask = np.asarray(mask_h5['dm'])

        img = Image.open(img_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        assert img.size[::-1] == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'
        
        #img = self.preprocess(img)
        mask = self.preprocess(mask, normalize=False)
        
        
        if self.transforms is not None:
            img = self.transforms(img)
        #     mask = self.transforms(mask)
        
        return {
            'image': img, # torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
