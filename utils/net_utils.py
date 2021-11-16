import cv2
from .postprocessing import PostProcess
from skimage import morphology
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from unet import UNet
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def get_diameter_count(distance_map):
    '''
    Obtenemos los diametros, usando de guia el esqueleto de cada segmento procesado con dynamic watershed a partir del mapa de distancia
    '''
    diametros = {}
    #distance_map = unet_dm(img_file)
    #distance_map = cv2.imread(sample_prediction, 0)
    dm_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #dm_normalized = dm_normalized.astype(np.uint8)
    #segmentos = PostProcess(dm_normalized, 40, 16)
    segmentos = PostProcess(dm_normalized, 40, 32)
    #distance_map = (distance_map / 255) / 0.01

    # recorrer todas los segmentos detectadaços:
    for i in range(0, segmentos.max()):
        # obtenemos el segmento actual:
        segmento = (segmentos==i+1)
        # obtenemos el skeleton del segmento:
        seg_skeleton = morphology.skeletonize(segmento, method='lee')
        # obtenemos sus diametros desde el mapa de distancia:
        seg_diametros = np.floor(distance_map[seg_skeleton>0]*2)
        # contamos los diametros:
        unique, counts = np.unique(seg_diametros, return_counts=True)
        seg_diametros_count = dict(zip(unique, counts))
        # juntamos los diametros:
        for k,v in seg_diametros_count.items():
            diametros[k] = diametros.get(k,0) + v

    return diametros

def calc_diameter_mean(diameter_count):
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


def unet_dm(img_file, model_file='MODEL.pth'):
    '''
    Función de predicción
    '''
    img = Image.open(img_file)
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                        ])
    img = tf(img)
    img = img.unsqueeze(0)
    #img.cuda()
    
    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    #net.cuda()
    net.load_state_dict(torch.load(model_file))
    net.eval()
    
    with torch.no_grad():
        output = net(img)
        
    dm = output.squeeze().cpu().numpy()
    return dm


def matchGt(paths, dataset_file):
    '''
    Retorna el valor del gt (promedio de diametro de una imagen)
    '''
    df = pd.read_pickle(dataset_file)
    gts = df.iloc[pd.Index(df['index']).get_indexer([int(os.path.splitext(os.path.basename(path))[0]) for path in paths])]
    return gts['gt'].tolist()


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def predict_dm(net, img_path, device=None):
    '''
    Predecir el mapa de distancia a partir de una imagen sintetica con ruido
    '''
    img = Image.open(img_path)
    
    if not isinstance(img.getpixel((0,0)), tuple):
        rgb_img = Image.new('RGB', img.size)
        rgb_img.paste(img)
        img = rgb_img
        
    ##################################
    # Padding
    print(img.size)
    diffY = 256 - img.size[1]
    diffX = 256 - img.size[0] 
    img = add_margin(img, diffY // 2, diffX - diffX//2, diffY - diffY//2, diffX // 2, (0,0,0))
#     print(img.size)
    ##########################       
    
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                        ])
    img = tf(img)
    img = img.unsqueeze(0)
    
    
    with torch.no_grad():
        if device is not None:
            img = img.to(device)
        output = net(img)
    dm = output.squeeze().cpu().numpy()
    return dm

def get_diameters(net, img_path_list, device=None):
    
    diameter_means = []
    
    for img_path in img_path_list:    
        pred_dm = predict_dm(net, img_path, device)
        diameter_count = get_diameter_count(pred_dm)
        diameter_mean = calc_diameter_mean(diameter_count)
        diameter_means.append(diameter_mean)
        
    return diameter_means

def get_diameters_from_dm_list(dm_list):
    diameter_means = []
    for distance_map in dm_list:
        diameter_count = get_diameter_count(distance_map)
        diameter_mean = calc_diameter_mean(diameter_count)
        diameter_means.append(diameter_mean)        
    return diameter_means

def to_multiple_gpu(net):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device=device)

def get_densities(diameter_count):
    suma = sum(diameter_count.values())
    densities = [v/suma for v in diameter_count.values()]
    return list(diameter_count.keys()),densities

def predict_dm_list(net, batch, device=None):
    with torch.no_grad():
        imgs = batch['image'] if device is None else batch['image'].to(device)
        output = net(imgs)
    predictions = output.cpu().numpy()
    dm_list = [predictions[i].squeeze() for i in range(predictions.shape[0])]
    return dm_list


def predict_dm_tensores(net, batch, device=None):
    with torch.no_grad():
        imgs = batch['image'] if device is None else batch['image'].to(device)
        output = net(imgs)
    return output.cpu()

def dm_tensores_to_dm_list(tensores):
    predictions = tensores.numpy()
    dm_list = [predictions[i].squeeze() for i in range(predictions.shape[0])]
    return dm_list