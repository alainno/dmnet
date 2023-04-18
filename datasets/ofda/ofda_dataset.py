import unittest
import cv2
import h5py
import os
import glob


class OFDASegmented:
    def __init__(self, src_dir):
        self.src_dir = src_dir
        
    def save_distance_transform(self, src, dst, as_h5=True):
        bin_img = cv2.imread(src, 0)
        dm_img = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
        if as_h5:
            with h5py.File(dst, 'w') as hf:
                hf['dm'] = dm_img
                return True
        else:
            dm_img = cv2.normalize(dm_img, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.imwrite(dst, dm_img)
    
    def export_as_distance_maps(self, dst_dir, as_h5=True):
        for filename in glob.glob(self.src_dir + "*.png"):
            if not self.save_distance_transform(filename, os.path.join(dst_dir, os.path.basename(filename).replace('.png','.h5')), as_h5):
                return False
        return True
        

class TestOFDASegmented(unittest.TestCase):
    def test_export_as_h5(self):
        src_dir = "ofda_segmentation/"
        dst_dir = "gt_dm/"
        dataset = OFDASegmented(src_dir)
        #dataset.distance_transform()
        self.assertEqual(dataset.export_as_distance_maps(dst_dir), True, "exportación satisfactoria")
        
        
if __name__ == "__main__":
    unittest.main()