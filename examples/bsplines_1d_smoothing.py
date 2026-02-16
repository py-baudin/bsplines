import numpy as np
from matplotlib import pyplot as plt
import bsplines

# sample random points
npoint = 10
data = np.random.uniform(-1, 1, size=(npoint, 2))
data[-1] = data[0]

# connect the points
spl = bsplines.bspline(data, degree=3, axes=0, extension='periodic')

# generate a noisy curve
ncoord = 300
coords = np.linspace(0, len(data) - 1, ncoord)
gt = spl(coords)
noisy = gt + np.random.normal(0, 1e-2, (ncoord, 2))
noisy[-1] = noisy[0]


# smoothing spline
navg = 10
degree = 6
spl = bsplines.bspline(noisy, degree=degree, navg=navg, extension='periodic')
locs = np.linspace(*spl.bounds[0], num=1000)
smooth = spl(locs)
nodes = spl(np.arange(*spl.bounds[0]))

# plot
plt.close('all')
plt.figure(0)
plt.plot(gt[:, 0], gt[:, 1], label='ground truth', alpha=0.5)
plt.plot(noisy[:, 0], noisy[:, 1], label='noisy')
plt.plot(smooth[:, 0], smooth[:, 1], label='BSpline approximation')
plt.plot(nodes[:, 0], nodes[:, 1], '+', color='k', label='BSpline nodes')
plt.title(f'BSpline approximation (num. avg={navg}, k={degree})')
plt.legend(loc='lower right')
plt.xticks([])
plt.yticks([])
plt.show()




