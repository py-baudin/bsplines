import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import bsplines

npt = 7
nco = 30
points = np.indices([npt, npt])
data = np.random.uniform(size=[npt, npt])
spl1 = bsplines.bspline(data, degree=3, extension="reflect")
spl2 = bsplines.bspline(data, degree=(2, 3), extension="reflect")

# grid coordinates
grid = [np.linspace(0, npt - 1, nco)] * 2
# random coordinates
coords = np.random.uniform(0, npt - 1, (2, 100))
k = (nco - 1) / (npt - 1)

fig, axes = plt.subplots(ncols=2, num="bsplines-grid")
plt.sca(axes[0])
plt.imshow(spl1(*grid).T, interpolation="nearest")
plt.scatter(
    coords[0] * k,
    coords[1] * k,
    c=spl1(coords),
    vmin=data.min(),
    vmax=data.max(),
    edgecolors="k",
)
plt.scatter(
    points[0] * k,
    points[1] * k,
    c=data,
    vmin=data.min(),
    vmax=data.max(),
    edgecolors="r",
    marker="s",
)
plt.axis("off")
plt.title(f"2d interpolation with k={spl1.degree}")

plt.sca(axes[1])
plt.imshow(spl2(*grid).T, interpolation="nearest")
plt.scatter(
    coords[0] * k,
    coords[1] * k,
    c=spl2(coords),
    vmin=data.min(),
    vmax=data.max(),
    edgecolors="k",
)
plt.scatter(
    points[0] * k,
    points[1] * k,
    c=data,
    vmin=data.min(),
    vmax=data.max(),
    edgecolors="r",
    marker="s",
)
plt.axis("off")
plt.title(f"2d interpolation with k={spl2.degree}")


plt.tight_layout()
plt.show()
