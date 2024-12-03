import numpy as np
from matplotlib import pyplot as plt
import bsplines


npoint = 10
data = np.random.uniform(-1, 1, size=npoint)

ncoord = 100
coords = np.linspace(0, npoint - 1, ncoord)

degree = 4
spl = bsplines.BSpline.prefilter(data, degree=degree)
interp = spl(coords)
diff1 = spl.derivative(1)(coords)
diff2 = spl.derivative(2)(coords)
diff3 = spl.derivative(3)(coords)

ylim = max(
    np.max(np.abs(interp) + npoint / ncoord * np.abs(diff1) * 2),
    np.max(np.abs(diff1) + npoint / ncoord * np.abs(diff2) * 2),
    np.max(np.abs(diff2) + npoint / ncoord * np.abs(diff3) * 2),
)

fig, axes = plt.subplots(nrows=3, num='diff-1d')
plt.sca(axes.flat[0])
plt.plot(coords, interp, 'k-', label=f'interpolation (degree={degree})', alpha=0.7)
plt.grid(axis='x')
plt.xlim(-0.5, npoint-0.5)
plt.ylim(-ylim, ylim)
plt.quiver(coords, interp, np.ones_like(diff1), diff1, diff1, angles='xy', units='xy', scale_units='xy', width=3e-2, cmap='plasma')
plt.plot(np.arange(npoint), data, 'o', color='red', label='data')
plt.legend(loc='upper right')

plt.sca(axes.flat[1])
plt.axhline(0, linestyle=':', color='gray')
plt.plot(coords, diff1, 'k-', label=f'first derivative', alpha=0.7)
plt.grid(axis='x')
plt.xlim(-0.5, npoint-0.5)
plt.ylim(-ylim, ylim)
plt.quiver(coords, diff1, np.ones_like(diff2), diff2, diff2, angles='xy', units='xy', scale_units='xy', width=3e-2, cmap='plasma')
plt.legend(loc='upper right')

plt.sca(axes.flat[2])
plt.axhline(0, linestyle=':', color='gray')
plt.plot(coords, diff2, 'k-', label=f'second derivative', alpha=0.7)
plt.grid(axis='x')
plt.xlim(-0.5, npoint-0.5)
plt.ylim(-ylim, ylim)
plt.quiver(coords, diff2, np.ones_like(diff3), diff3, diff3, angles='xy', units='xy', scale_units='xy', width=3e-2, cmap='plasma')
plt.legend(loc='upper right')

plt.suptitle('1D B-Splines: n-th derivatives')
plt.tight_layout()
plt.show()




