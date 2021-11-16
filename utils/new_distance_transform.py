# from utils import getSkeletonIntersection, removeCrosses
import numpy as np
import math
import cv2
from skimage import morphology

def new_distance_transform(sample_img):
    """ Implementación del algoritmo de transformación de distancia de Ziabari, 2008 """
    # Declaramos la variable de salida:
    diametros = {}
    # Binarización de la micrografia:
    rgb_img = cv2.imread(sample_img)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _,bin_img = cv2.threshold(gray_img,16,255,cv2.THRESH_BINARY)
    # Exqueletizamos la imagen binaria:
    skeleton = morphology.skeletonize(bin_img, method='lee')  
    # Obtenemos su mapa de distancia:
    distance_map = cv2.distanceTransform(bin_img, cv2.DIST_C, cv2.DIST_MASK_3)
    # Obtenemos las intersecciones del skeleton:
    crosses = getSkeletonIntersection(skeleton)
    # Removemos las intersecciones segun el skeleton y el mapa de distancia:
    uncrossed = removeCrosses(skeleton, distance_map, crosses)
    # computamos los diametros con el mapa de distancia y el skeleton sin intersecciones:
    diametros_dm = np.floor(distance_map[uncrossed>0]*2)
    unique, counts = np.unique(diametros_dm, return_counts=True)
    diametros = dict(zip(unique, counts))

    return diametros


def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0]];
    image = skeleton.copy();
    # image = image/255;
    intersections = list();
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbours1 = neighbours(x, y, image);
                valid = True;
                if neighbours1 in validIntersection:
                    intersections.append((y, x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections

def removeCrosses(skeleton, distance_map, crosses):
    """ Eliminar intersecciones """
    uncrossed = skeleton.copy()
    # recorrer puntos de interseccion:
    for cross in crosses:
        x = cross[0]
        y = cross[1]
        width = math.ceil(distance_map[y,x])
        x_start = x - width if x - width >= 0 else 0
        y_start = y - width if y - width >= 0 else 0
        # por cada punto eliminar su alrededor segun valor del mapa de distancia
        uncrossed[y_start:y+width+1,x_start:x+width+1] = False

    return uncrossed