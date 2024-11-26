import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import bsplines

points = np.indices([7, 7])
data = np.random.uniform(size=[7, 7])
coords = np.linspace(0, 6, 30)[np.indices([30, 30])]

fig, axes = plt.subplots(nrows=2, ncols=2, num='2d-vs-scipy')
for i in range(4):
    _bsplines = bsplines.interpolate(data, coords, degree=i, extension='reflect')
    plt.sca(axes.flat[i])
    plt.imshow(_bsplines.T, interpolation='nearest')
    plt.scatter(*np.linspace(0, 29, 7)[points], c=data, vmin=data.min(), vmax=data.max(), edgecolors='k')
    plt.axis('off')
    plt.title(f'Degree {i}')

plt.suptitle('2d interpolation with B-Splines')
plt.tight_layout()
plt.show()
