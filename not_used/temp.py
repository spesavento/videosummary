#!/usr/bin/env python

# python -m pip install -U scikit-image
# (also Cython, numpy, scipi, matplotlib, networkx, pillow, imageio, tifffile, PyWavelets)
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np # may not be needed
import cv2
import glob
import numpy as np
import os
from os.path import isfile, join


print(str(1).zfill(5))
print(str(135).zfill(5))
print(str(1450).zfill(5))