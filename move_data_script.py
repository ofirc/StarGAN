import os
import shutil

'''
This script separate CelebA data set to two groups:
1. Train set - first 180000 images.
2. Test set - all other images.

Eventually, we didn't use that script because StarGAN has done it for us (Kept test set).
'''


CelebA_IN_DIR = r'data/CelebA_nocrop/images'
CelebA_OUT_DIR = r'data/CelebA_held_out_set'

if not os.path.exists(CelebA_OUT_DIR):
    os.makedirs(CelebA_OUT_DIR)

images = os.listdir(CelebA_IN_DIR)

images.sort()

for img in images[180000:]:
    src = os.path.join(CelebA_IN_DIR, img)
    shutil.move(src, CelebA_OUT_DIR)
