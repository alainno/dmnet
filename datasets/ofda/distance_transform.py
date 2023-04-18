import cv2
import unittest
import glob
import os

class DistanceMap:
    def __init__(self, src, dst):
        self.src_path = src
        self.dst_path = dst
        
    def save_distance_transform(self):
        bin_img = cv2.imread(self.src_path, 0)
        dm = cv2.distanceTransform(bin_img,cv2.DIST_L2,3)
        dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.imwrite(self.dst_path, dm)
    
    def distance_transform_batch(self, src_dir, dst_dir):
        for filename in glob.glob(src_dir + "*.png"):
            if not self.save_distance_transform(filename, dst_dir + os.path.basename(filename)):
                return False
        return True


class TestDistanceMap(unittest.TestCase):
    def runTest(self):
        dm = DistanceMap()
        self.assertEqual(dm.save_distance_transform("ofda_segmentation/0157.png", "gt_dm/0157.png"), True, "generación incorrecta")
        
    def test_distance_transform_batch(self):
        dm = DistanceMap()
        self.assertEqual(dm.distance_transform_batch("ofda_segmentation/", "gt_dm/"), True, "generación incorrecta")
            

        
if __name__ == "__main__":
    unittest.main()