from skimage import morphology
import cv2
import numpy as np
import math

def fiber_ec(sample_img, retornar_dibujo=False):
    '''
    Implementación del algoritmo de FIBER EC (Quispe, 2017)
    '''
    # Binarizamos la imagen y obtenemos su esqueleto:
    rgb_img = cv2.imread(sample_img)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _,bin_img = cv2.threshold(gray_img,16,255,cv2.THRESH_BINARY)
    skeleton = morphology.skeletonize(bin_img, method='lee')
    skeleton = skeleton.astype(np.uint8)
    skeleton[skeleton==1] = 255
    # Detectamos las líneas con el algoritmo de hough:
    linesP = cv2.HoughLinesP(skeleton, 1, np.pi / 180, 25, None, 10, 1)
    # Definimos las variables de salida
    diametros = {}
    target = rgb_img.copy()
    # Si hay líneas detectadas:
    if linesP is not None:
        for i in range(0, len(linesP)):
            # Se obtienen el punto de origen y el punto final de la línea detectada:
            l = linesP[i][0]
            # Dibujamos la línea en color rojo:
            cv2.line(target, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv2.LINE_8)
            # Se obtiene el ángulo de la línea detectada y el eje horizontal:
            p1 = Point(l[0], l[1])
            p2 = Point(l[2], l[3])
            angle = math.atan2(p1.y-p2.y,p1.x-p2.x)
            # Contar líneas paralelas:
            hay_linea = True
            distancia = 0 # Distancia incremental entre la línea central y las líneas paralelas
            diametro = 1 # Contador de líneas
            while hay_linea:
                distancia += 1
                # Se obtienen línea paralela con distancia positiva:
                new_p1_positive = getNewPoint(p1, angle, distancia)
                new_p2_positive = getNewPoint(p2, angle, distancia)
                point_list_positive = getPointList(new_p1_positive, new_p2_positive)
                # Se obtienen línea paralela con distancia negativa:
                new_p1_negative = getNewPoint(p1, angle, -distancia)
                new_p2_negative = getNewPoint(p2, angle, -distancia)
                point_list_negative = getPointList(new_p1_negative, new_p2_negative)
                # Verificar si las líneas paralelas estan completas:
                linea_completa_positive = esLineaCompleta(point_list_positive, bin_img)
                linea_completa_negative = esLineaCompleta(point_list_negative, bin_img)
                # Dibujar al menos una línea paralela e incrementar diametro:
                if linea_completa_positive:
                    cv2.line(target, (new_p1_positive.x, new_p1_positive.y), (new_p2_positive.x, new_p2_positive.y), (0,255,0), 1, cv2.LINE_AA)
                    diametro += 1
                if linea_completa_negative:
                    cv2.line(target, (new_p1_negative.x, new_p1_negative.y), (new_p2_negative.x, new_p2_negative.y), (0,255,0), 1, cv2.LINE_AA)
                    diametro += 1
                # Dejamos de contar si falta una línea paralela
                if not linea_completa_positive or not linea_completa_negative:
                    hay_linea = False
            # Agregamos y/o contamos el diametro obtenido:
            diametros[diametro] = diametros.get(diametro, 0) + 1

    if retornar_dibujo:
        return diametros, target
    else:
        return diametros
    
    
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

def getNewPoint(origen, angle, distancia):
    new_x = int(round(origen.x + distancia*math.cos(angle+math.pi/2)))
    new_y = int(round(origen.y + distancia*math.sin(angle+math.pi/2)))
    return Point(new_x, new_y)

def getPointList(p1, p2):
    point_list = []

    p1 = np.array([p1.y,p1.x])
    p2 = np.array([p2.y,p2.x])

    p = p1
    d = p2-p1
    N = np.max(np.abs(d))
    s = d/N

    point_list.append(Point(int(round(p[1])),int(round(p[0]))))
    for ii in range(0,N):
        p = p+s
        point_list.append(Point(int(round(p[1])),int(round(p[0]))))

    return point_list

def esLineaCompleta(point_list, bin_img):
    completa = True
    w = bin_img.shape[1]-1
    h = bin_img.shape[0]-1
    for point in point_list:
        completa = (point.x >= 0
                    and point.x <= w
                    and point.y >= 0
                    and point.y <= h
                    and bin_img[point.y, point.x] > 0)
        if not completa:
            break
    return completa