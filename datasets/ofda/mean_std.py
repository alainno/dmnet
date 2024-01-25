import numpy as np
from welford import Welford
import os
import cv2

# Initialize Welford object
w = Welford()

path = "./test/samples/"

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    if(filename.endswith('.png') and os.path.isfile(img_path)):
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                w.add(img[x,y,:])
    
print(w.mean)
print(np.sqrt(w.var_s))