from welford import Welford
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2

if __name__ == '__main__':
    
    w = Welford()
    imgs_dir = '../data_dm_overlapping/imgs/'
    
    for img_name in listdir(imgs_dir):
        img_path = join(imgs_dir, img_name)
        
        if(not img_name.endswith('.png') or not isfile(img_path)):
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                w.add(img[i,j,:]/255)

    print('Mean: ', w.mean)
    print('Std: ', np.sqrt(w.var_s))