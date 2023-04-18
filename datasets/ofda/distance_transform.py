import cv2
import unittest
import glob
import os
import h5py

class DistanceMap:
#    def __init__(self, src, dst):
#        self.src_path = src
#        self.dst_path = dst
        
    def save_distance_transform(self, src, dst, as_h5=True):
        bin_img = cv2.imread(src, 0)
        dm = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
        if as_h5:
            output_path = dst.replace('.png','.h5')
            with h5py.File(output_path, 'w') as hf:
                hf['dm'] = self.dm_mask
        else:
            dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.imwrite(dst, dm)
    
    def distance_transform_batch(self, src_dir, dst_dir, as_h5=True):
        for filename in glob.glob(src_dir + "*.png"):
            if not self.save_distance_transform(filename, dst_dir + os.path.basename(filename), as_h5):
                return False
        return True


class TestDistanceMap(unittest.TestCase):
#    def runTest(self):
#        dm = DistanceMap()
#        self.assertEqual(dm.save_distance_transform("ofda_segmentation/0157.png", "gt_dm/0157.png"), True, "generación incorrecta")
        
    def test_distance_transform_batch(self):
        dm = DistanceMap()
        self.assertEqual(dm.distance_transform_batch("ofda_segmentation/", "gt_dm/"), True, "generación incorrecta")
            

        
if __name__ == "__main__":
    unittest.main()