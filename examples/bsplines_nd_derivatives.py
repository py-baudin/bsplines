import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import bsplines

degree = 3
# nx, ny = (25, 26)
nx, ny = (10, 10)
sx, sy = (100, 101)
# sx, sy = (nx, ny)

rgen = np.random.RandomState(5)
# rgen = np.random
rd = rgen.uniform(-0.5 / nx, 0.5 / nx, 4)


def f(x, y):
    phi = rd[0] * x**2 * y + rd[1] * y**2 + rd[2] * x + rd[3]
    return np.cos(phi)


def df(x, y):
    phi = rd[0] * x**2 * y + rd[1] * y**2 + rd[2] * x + rd[3]
    dx = -(2 * rd[0] * x * y + rd[2]) * np.sin(phi)
    dy = -(rd[0] * x**2 + 2 * rd[1] * y) * np.sin(phi)
    return np.stack([dx, dy])


points = np.indices([nx, ny])
data = f(*points)
coords = [np.linspace(0, nx - 1, sx), np.linspace(0, ny - 1, sy)]
locs = np.linspace(0, sx - 1, nx)[points[0]], np.linspace(0, sy - 1, ny)[points[1]]
grid = np.meshgrid(*coords, indexing="ij")
ext = "anti-symmetric"
spl = bsplines.BSpline.prefilter(data, degree=degree, extension=ext)
pred = spl(*coords).T
gt = f(*grid).T

vmin, vmax = data.min(), data.max()

fig, axes = plt.subplots(nrows=2, ncols=3, num="diff-2d", figsize=(7, 6))
plt.sca(axes[0, 0])
plt.imshow(gt, vmin=vmin, vmax=vmax)
plt.axis("off")
plt.title(f"Ground truth")

plt.sca(axes[1, 0])
plt.imshow(pred, vmin=vmin, vmax=vmax)
plt.scatter(*locs, c=data, edgecolors="k")
plt.axis("off")
plt.title(f"2D interpolation (n={degree})")

dxy = df(*points)
dgt = df(*grid)
dsplx = spl.derivative(1, axis=0)(*coords)
dsply = spl.derivative(1, axis=1)(*coords)


plt.sca(axes[0, 1])
plt.imshow(dgt[0].T, vmin=vmin, vmax=vmax)
plt.axis("off")
plt.title(f"x derivative")

plt.sca(axes[1, 1])
plt.imshow(dsplx.T, vmin=vmin, vmax=vmax)
h = plt.scatter(*locs, c=dxy[0], edgecolors="k", vmin=vmin, vmax=vmax)
plt.axis("off")
plt.title("x derivative")

plt.sca(axes[0, 2])
plt.imshow(dgt[1].T, vmin=vmin, vmax=vmax)
plt.axis("off")
plt.title(f"y derivative")

plt.sca(axes[1, 2])
plt.imshow(dsply.T, vmin=vmin, vmax=vmax)
plt.scatter(*locs, c=dxy[1], edgecolors="k", vmin=vmin, vmax=vmax)
plt.axis("off")
plt.title("y derivative")

plt.suptitle("2D B-Spline interpolation and derivatives ")
plt.tight_layout()
plt.show()
