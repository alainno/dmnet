# Mahotas: Mahotas is a computer vision and image processing library for Python.
import mahotas as mh
import numpy as np
import cv2

############
# Prunning #
############

# def branchedPoints(skel):
#     branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
#     branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
#     branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
#     branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
#     branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
#     branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
#     branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
#     branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
#     branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
#     br1=mh.morph.hitmiss(skel,branch1)
#     br2=mh.morph.hitmiss(skel,branch2)
#     br3=mh.morph.hitmiss(skel,branch3)
#     br4=mh.morph.hitmiss(skel,branch4)
#     br5=mh.morph.hitmiss(skel,branch5)
#     br6=mh.morph.hitmiss(skel,branch6)
#     br7=mh.morph.hitmiss(skel,branch7)
#     br8=mh.morph.hitmiss(skel,branch8)
#     br9=mh.morph.hitmiss(skel,branch9)
#     return br1+br2+br3+br4+br5+br6+br7+br8+br9

def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    # ep1=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
    # ep2=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint2)
    # ep3=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint3)
    # ep4=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint4)
    # ep5=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint5)
    # ep6=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint6)
    # ep7=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint7)
    # ep8=cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint8)
    ep1=mh.morph.hitmiss(skel, endpoint1)
    ep2=mh.morph.hitmiss(skel, endpoint2)
    ep3=mh.morph.hitmiss(skel, endpoint3)
    ep4=mh.morph.hitmiss(skel, endpoint4)
    ep5=mh.morph.hitmiss(skel, endpoint5)
    ep6=mh.morph.hitmiss(skel, endpoint6)
    ep7=mh.morph.hitmiss(skel, endpoint7)
    ep8=mh.morph.hitmiss(skel, endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep

def pruning(skeleton, size):
    '''remove iteratively end points "size" 
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton