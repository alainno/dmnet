import cv2
import unittest

class DistanceMap:
    def __init__(self, src, dst):
        self.src_path = src
        self.dst_path = dst
        
    def generate(self):
        bin_img = cv2.imread(self.src_path, 0)
        dm = cv2.distanceTransform(bin_img,cv2.DIST_L2,3)
        dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.imwrite(self.dst_path, dm)

class TestDistanceMap(unittest.TestCase):
    def runTest(self):
        dm = DistanceMap("ofda_segmentation/0157.png", "gt_dm/0157.png")
        self.assertEqual(dm.generate(), True, "generación incorrecta")
        
        
    #def test_distance_transform_batch(self, src_dir, dst_dir):
        
        
        
if __name__ == "__main__":
    unittest.main()