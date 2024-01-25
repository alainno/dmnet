import os
import cv2
import matplotlib.pyplot as plt

#
cur_dir = os.path.dirname(os.path.realpath(__file__))
imgs_path = os.path.join(cur_dir, 'test/samples')

# fig = plt.figure(figsize=(4*5,4))
fig = plt.figure(figsize=(4*5,4))
fig.subplots_adjust(wspace=0.1)
k = 0

filenames = os.listdir(imgs_path)
filenames.sort()

for filename in filenames:
    img_path = os.path.join(imgs_path,filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path, 0)
        # img = 255 - img
        plt.subplot(1,5,k+1), plt.axis('off'), plt.imshow(img, cmap='gray')
        k+=1
    if k==4:
        break
        
plt.savefig('samples.png', bbox_inches='tight',pad_inches=0)
#plt.show()