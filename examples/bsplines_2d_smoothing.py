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

# 10-average spline
s = 10
degree = 6
spl = bsplines.bspline(noisy, degree=10, axes=[0], s=s, extension='periodic')
smooth = spl(np.linspace(0, spl.shape[0] - 2, 1000))
nodes = spl(np.arange(spl.shape[0] - 1))
# s = 10
# degree = 6
# coeffs = 0
# for i in range(s):
#     select = np.roll(noisy, i, axis=0)[::s]
#     # select = np.r_[select, select[0:1]]
#     spl = bsplines.bspline(select, axes=[0], degree=degree, extension='periodic')
#     coeffs += spl.coeffs
# spl = bsplines.BSpline(degree, coeffs/s, axes=[0])
# locs = np.linspace(0, len(select) - 1, 1000)
# smooth2 = spl(locs)

# plot
plt.close('all')
plt.figure(0)
plt.plot(gt[:, 0], gt[:, 1], label='ground truth')
plt.plot(noisy[:, 0], noisy[:, 1], label='noisy')
plt.plot(smooth[:, 0], smooth[:, 1], label='BSpline approximation')
plt.plot(nodes[:, 0], nodes[:, 1], '+', color='k')
plt.title(f'BSpline approximation (s={s}, k={degree})')
plt.legend(loc='lower right')
plt.show()




