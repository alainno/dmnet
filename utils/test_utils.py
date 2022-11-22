import numpy as np
import math
import cv2
from .pruning import *
import os
import torch
from torchvision import transforms
from PIL import Image
from .net_utils import *
import pandas as pd

def get_std(diametros):
    n = sum(diametros.values())
    mean = sum([k*v for k,v in diametros.items()]) / n
    var = sum([(k-mean)**2 for k,v in diametros.items() for i in range(int(v))]) / n
    std = math.sqrt(var)
    return std

def get_mean_diameter(diameter_count):
    """
    Promediar el diámetro:
    Recibe un diccionario con los diámetros detectados y y la cantidad respectiva.
    Retorna el promedio del diámetro
    """
    suma = 0
    contador = 0
    for k,v in diameter_count.items():
        suma += k*v
        contador += v
    return (suma / contador)

def get_mae(values, col_index_1, col_index_2):
    mae = np.sum(np.abs(values[:,col_index_1] - values[:,col_index_2]))
    return mae

def get_mse(values, col_index_1, col_index_2):
    mse = np.sum(np.abs(np.power(values[:,col_index_1] - values[:,col_index_2], 2)))
    return mse


# def get_binarized_images(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename), 0)
#         if img is not None:
#             binarized = binarize(img)
# #             cv2.imwrite(os.path.join(folder, 'bin_'+filename), bin)
#             binarized = cv2.cvtColor(binarized,cv2.COLOR_GRAY2RGB)
#             binarized = binarized.astype('f')
#             images.append(binarized)
#     return images

# def binarize(img):
#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     blurred = cv2.GaussianBlur(opening, (3, 3), 0)
#     _,otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     bin_img = cv2.bitwise_not(otsu)
#     bin_img_pruned = pruning(bin_img, 10)
#     bin_img_pruned = bin_img_pruned.astype('uint8')
#     bin_img_pruned = cv2.normalize(bin_img_pruned, None, 0, 255, cv2.NORM_MINMAX)
#     return bin_img_pruned


# def predict_from_list(image_list, net, device=None):
    
#     tf = transforms.Compose([
#                     #transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
#                 ])
    
#     tensor_list = [tf(torch.from_numpy(image)).permute(2,0,1).unsqueeze(0) for image in image_list]
#     tensor_batch = torch.cat(tensor_list)
    

#     #tensor_batch = tf(tensor_batch)
    
#     with torch.no_grad():
#         imgs = tensor_batch if device is None else tensor_batch.to(device)
#         output = net(imgs)
#     predictions = output.cpu().numpy()
#     dm_list = [predictions[i].squeeze() for i in range(predictions.shape[0])]
#     return dm_list

def batch_from_folder(folder_path):

    tensor_list = []

    tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.114],std=[0.237])
                    # transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                ])
    
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename), 0)
        if img is not None:
            pil_img = Image.open(os.path.join(folder_path,filename))
            
            diffX = 256 - pil_img.size[0]            
            diffY = 256 - pil_img.size[1]
            pil_img = add_margin(pil_img, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, 0)
            
            tensor = tf(pil_img)
            tensor = tensor.expand(3, -1, -1)
            tensor_list.append(tensor)    
    
    tensor_batch = torch.cat([tensor.unsqueeze(0) for tensor in tensor_list])
    return tensor_batch



def gt_ofda_diameter_count(img_path):
    
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    
    df = pd.read_csv(f'data/ofda/gt/{img_id}.csv',header=None)

    diameter_count = {}

    for i,row in df.iterrows():
        diameter_count[row[1]] = diameter_count.get(row[1], 0) + row[0]
    
    return diameter_count



def batch_from_list(img_list):

    tensor_list = []

    tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                ])
    
    for img in img_list:
        pil_img = Image.fromarray(img)

        diffX = 256 - pil_img.size[0]            
        diffY = 256 - pil_img.size[1]
        pil_img = add_margin(pil_img, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, 0)

        tensor = tf(pil_img)
        tensor = tensor.expand(3, -1, -1)
        tensor_list.append(tensor)  
    
    tensor_batch = torch.cat([tensor.unsqueeze(0) for tensor in tensor_list])
    return tensor_batch