from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py
import random

class OfdaDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', transforms=None, mask_h5=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        # agregados
        self.transforms = transforms
        self.mask_h5 = mask_h5

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, debug=False):
        
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        if(debug):
            print('pil_img.mode', pil_img.mode)

        img_nd = np.array(pil_img)
        
        if(debug):
            print('img_nd.shape', img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
           
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        
        #if img_trans.max() > 1:
        #    img_trans = img_trans / 255
            
        return img_trans
    
    @staticmethod
    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    
    def __set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        if self.mask_h5:
            mask_h5_data = h5py.File(mask_file[0], 'r')
            mask = np.asarray(mask_h5_data['dm'])
            #mask_size = mask.shape[::-1]
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_file[0]) #.convert('L')
            #mask_size = mask.size
        
        img = Image.open(img_file[0])
        
        '''
        ############# padding ############
        diffX, diffY = 256 - mask.size[0], 256 - mask.size[1]
        mask = BasicDataset.add_margin(mask, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, 0)
        
        diffX, diffY = 256 - img.size[0], 256 - img.size[1]
        #img = BasicDataset.add_margin(img, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, 0) #synthetic
        img = BasicDataset.add_margin(img, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, (255,255,255)) #ofda
        ##################################
        '''

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        #mask = self.preprocess(mask, self.scale)
        
        if self.transforms is not None:
            seed = random.randint(0, 2**32)
            self.__set_seed(seed)
            img = self.transforms[0](img)
            self.__set_seed(seed)
            mask = self.transforms[1](mask)
        
        return {
            'image': img, #torch.from_numpy(img).type(torch.FloatTensor),
            #'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'mask': mask,
            'path': img_file[0]
        }


#class CarvanaDataset(BasicDataset):
#    def __init__(self, imgs_dir, masks_dir, scale=1):
#        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
