import cv2
from postprocessing import PostProcess
from skimage import morphology
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from unet import UNet
import os
import pandas as pd

def get_diameter_count(distance_map):
    '''
    Obtenemos los diametros, usando de guia el esqueleo de cada segmento procesado con dynamic watershed a partir del mapa de distancia
    '''
    diametros = {}
    #distance_map = unet_dm(img_file)
    #distance_map = cv2.imread(sample_prediction, 0)
    dm_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #dm_normalized = dm_normalized.astype(np.uint8)
    segmentos = PostProcess(dm_normalized, 40, 16)
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
    #base = os.path.basename(path)
    #index = os.path.splitext(base)[0]
    
    df = pd.read_pickle(dataset_file)
       
    #df = pd.DataFrame({'index':[0,1,2,3,4],'gt':[11,12,13,14,15]})

    #paths = ["002", "005"]

    #gts = df.loc[df["index"].isin([int(os.path.splitext(os.path.basename(path))[0]) for path in paths])]
    
    gts = df.iloc[pd.Index(df['index']).get_indexer([int(os.path.splitext(os.path.basename(path))[0]) for path in paths])]
    
    '''
    try:
        gt = df.loc[df['index'] == index]['gt'].item()
    except:
        gt = None
    
    return gt
    '''
    return gts['gt'].tolist()


def predict_dm(net, img_path):
    '''
    Predecir el mapa de distancia a partir de una imagen sintetica con ruido
    '''
    img = Image.open(img_path)
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                        ])
    img = tf(img)
    img = img.unsqueeze(0)    
    with torch.no_grad():
        output = net(img)
    dm = output.squeeze().cpu().numpy()
    return dm

def get_diameters(net, img_path_list):
    
    diameter_means = []
    
    for img_path in img_path_list:    
        pred_dm = predict_dm(net, img_path)
        diameter_count = get_diameter_count(pred_dm)
        diameter_mean = calc_diameter_mean(diameter_count)
        diameter_means.append(diameter_mean)
        
    return diameter_means