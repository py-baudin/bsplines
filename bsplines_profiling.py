import numpy as np
from scipy import ndimage
import bsplines
import time


# make data
size1 = 25
size2 = 100
nd = 2
data = np.random.uniform(size=(size1,)*nd)
locations = np.indices((size2,)*nd)/(size2 - 1) * (size1 - 1)
# locations = np.random.uniform(0, size1 - 1, size=(nd,) + (size2,)*nd)

# pre-run on subset
# subset = np.zeros(nd, dtype=int)
# bsplines.interpolate(data, locations[:,subset], degree=3, extension='reflect')


# scipy
tic1 = time.time()
for i in range(100):
    res1 = ndimage.map_coordinates(data, locations, order=3, mode='reflect')

tic2 = time.time()
print(f'scipy: {tic2 - tic1:.3}s')

# bsplines
tic1 = time.time()
for i in range(100):
    res2 = bsplines.interpolate(data, locations, degree=3, extension='reflect')

tic2 = time.time()
print(f'bsplines: {tic2 - tic1:.3}s')

assert np.allclose(res1, res2, atol=1e-7)
