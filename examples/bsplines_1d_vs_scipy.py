import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import bsplines

size = 5
points = np.arange(size)
data = np.random.uniform(size=size)
data[-1] = data[0] # for periodic condition
coords = np.sort(np.r_[np.random.uniform(0, size - 1, 300), points])

degree = 3
fig, axes = plt.subplots(nrows=2, ncols=2, num='1d-vs-scipy', sharex=True, sharey=True)
for i, ext in enumerate([None, 'natural', 'clamped', 'periodic']):
    plt.sca(axes.flat[i])
    plt.plot(points, data, 'ko', label='data')
    _scipy = interpolate.make_interp_spline(points, data, k=degree, bc_type=ext)(coords)
    plt.plot(coords, _scipy, label=f'scipy k={degree}')
    if ext is not None:
        _bsplines = bsplines.interpolate(data, coords, degree=degree, extension=ext)
        plt.plot(coords, _bsplines, ':', label=f'bsplines k={degree}')
    plt.legend(loc='lower right')
    plt.grid(axis='x')
    plt.title(f'Boundary condition: {ext}')
plt.suptitle(f'`scipy.interpolate` vs `bsplines`')

fig, axes = plt.subplots(nrows=2, ncols=2, num='1d-vs-scipy-derivatives', sharex=True, sharey=True)
for i, ext in enumerate([None, 'natural', 'clamped', 'periodic']):
    plt.sca(axes.flat[i])
    _scipy = interpolate.make_interp_spline(points, data, k=degree, bc_type=ext).derivative(1)(coords)
    plt.plot(coords, _scipy, label=f'scipy k={degree}')
    if ext is not None:
        _bsplines = bsplines.interpolate(data, coords, degree=degree, extension=ext, order=1)
        plt.plot(coords, _bsplines, ':', label=f'bsplines k={degree}')
    plt.legend(loc='lower right')
    plt.grid(axis='x')
    plt.title(f'Boundary condition: {ext}')
plt.suptitle(f'1st derivatives')

fig, axes = plt.subplots(nrows=2, ncols=2, num='1d-vs-scipy-derivatives2', sharex=True, sharey=True)
for i, ext in enumerate([None, 'natural', 'clamped', 'periodic']):
    plt.sca(axes.flat[i])
    _scipy = interpolate.make_interp_spline(points, data, k=degree, bc_type=ext).derivative(2)(coords)
    plt.plot(coords, _scipy, label=f'scipy k={degree}')
    if ext is not None:
        _bsplines = bsplines.interpolate(data, coords, degree=degree, extension=ext, order=2)
        plt.plot(coords, _bsplines, ':', label=f'bsplines k={degree}')
    plt.legend(loc='lower right')
    plt.grid(axis='x')
    plt.title(f'Boundary condition: {ext}')
plt.suptitle(f'2nd derivatives')

plt.tight_layout()
plt.show()

