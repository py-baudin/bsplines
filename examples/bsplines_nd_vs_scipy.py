import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import bsplines

points = np.indices((7, 7))
data = np.random.uniform(size=(7, 7))
coords = np.indices([30, 30]) / 30 * 6
degree = 3

extensions = ["reflect", "mirror", "nearest", "grid-wrap"]

fig, axes = plt.subplots(nrows=3, ncols=4, num="2d-vs-scipy", figsize=(10, 6))
for i in range(4):
    ext = extensions[i]
    _scipy = ndimage.map_coordinates(data, coords, order=degree, mode=ext)
    _bsplines = bsplines.interpolate(data, coords, degree=degree, extension=ext)
    plt.sca(axes[0, i])
    plt.imshow(_scipy, interpolation="nearest")
    plt.axis("off")
    plt.title(f"scipy - {ext}") if i == 0 else plt.title(ext)
    plt.sca(axes[1, i])
    plt.imshow(_bsplines, interpolation="nearest")
    plt.axis("off")
    plt.title(f"bsplines - {ext}") if i == 0 else plt.title(ext)
    plt.sca(axes[2, i])
    plt.imshow(_scipy - _bsplines, interpolation="nearest", cmap="plasma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.title(f"diff - {ext}") if i == 0 else plt.title(ext)


plt.suptitle(f"`scipy.ndimage` vs `bsplines`")
plt.tight_layout()
plt.show()
