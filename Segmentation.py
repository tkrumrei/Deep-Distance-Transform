'''
Schritte:
- Seeds finden (lokale Minima und lokale Maxima Hintergrund)
- Seed based watershed Algorithm aus Paper
'''
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from PIL import Image

def find_local_minima(image):
    local_minima = (image == minimum_filter(image, footprint=np.ones((5, 5))))
    return np.argwhere(local_minima)

def find_local_maxima(image, neighborhood_size=5):
    local_maxima = (image == maximum_filter(image, footprint=np.ones((neighborhood_size, neighborhood_size))))
    return np.argwhere(local_maxima)

test = np.load("D:/Datasets/Cellpose/train/DistanceTransform/000_DT.npy")

Smin = find_local_minima(test)

img = Image.fromarray(test.astype(np.uint8))

img.show()

