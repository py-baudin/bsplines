import numpy as np
from scipy import ndimage
import bsplines
import time

# make data
nrep = 10
size1 = 25
size2 = 1000
data = np.random.uniform(size=(size1, size1))
locations = np.indices((size2, size2)) / (size2 - 1) * (size1 - 1)
grid = [np.linspace(0, size1 - 1, size2) for _ in range(2)]

# scipy
coeffs = ndimage.spline_filter(data, order=3, mode="reflect")
tic1 = time.time()
for i in range(nrep):
    res1 = ndimage.map_coordinates(
        coeffs, locations, order=3, mode="reflect", prefilter=False
    )
tic2 = time.time()
print(f"scipy: {tic2 - tic1:.3}s")


spl = bsplines.bspline(data, degree=3, extension="reflect")

# bsplines
tic1 = time.time()
for i in range(nrep):
    res2 = spl(locations)
tic2 = time.time()
print(f"bsplines: {tic2 - tic1:.3}s")

# bsplines on grid
tic1 = time.time()
for i in range(nrep):
    res3 = spl(*grid)
tic2 = time.time()
print(f"bsplines (grid): {tic2 - tic1:.3}s")


assert np.allclose(res1, res2, atol=1e-7)
assert np.allclose(res1, res3, atol=1e-7)
