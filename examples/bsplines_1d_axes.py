import numpy as np
from matplotlib import pyplot as plt
import bsplines

degree = 3
nx = 10
nd = 10
rgen = np.random


# model
alpha = rgen.uniform(0, 2, nd)
eta = rgen.uniform(0, 4 * np.pi, nd)
phi = rgen.uniform(0, 2 * np.pi, nd)
gamma = rgen.uniform(0, 2, nd)
w = rgen.uniform(0, 1)


def f(x):
    return alpha * np.cos(eta * x[..., np.newaxis] + phi) + gamma * w**3


# generate dictionary
grid = np.linspace(0, 1, nx)
dct = f(grid)

# interpolate
spl = bsplines.BSpline.prefilter(dct, axes=0, bounds=(0, 1))
coords = np.linspace(*spl.bounds[0], 300)
intp = spl(coords)

# plot dictionary
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_title(f"Dictionary elements (n={nx})")
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_title(f"Interpolated elements (k={spl.degree})")
ax2.shareview(ax1)
colors = plt.cm.plasma(np.linspace(0, 1, nx))
for i, v in enumerate(grid):
    ax1.plot(np.arange(nd), dct[i], zs=v, zdir="y", alpha=0.7, color=colors[i])
    ax2.plot(np.arange(nd), dct[i], zs=v, zdir="y", alpha=0.7, color=colors[i])
ax1.set_xlabel("index")
ax1.set_ylabel("parameter value")
for i in range(nd):
    ax2.plot(coords, intp[:, i], ":", zs=i, zdir="x", alpha=0.7, color="k")
ax2.set_xlabel("index")
ax2.set_ylabel("parameter value")
plt.show()
