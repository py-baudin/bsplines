import numpy as np
from matplotlib import pyplot as plt
import bsplines


npoint = 10
data = np.random.uniform(-1, 1, size=npoint)

ncoord = 50
coords = np.sort(np.r_[np.random.uniform(0, npoint - 1, ncoord), np.arange(npoint)])

degree = 3
interp = bsplines.interpolate(data, coords, degree=degree)
diff1 = bsplines.interpolate(data, coords, degree=degree, order=1)
diff2 = bsplines.interpolate(data, coords, degree=degree, order=2)

ylim = max(
    np.max(np.abs(interp) + npoint / ncoord * np.abs(diff1) * 2),
    np.max(np.abs(diff1) + npoint / ncoord * np.abs(diff2) * 2),
)

fig, axes = plt.subplots(nrows=2, num='diff-1d')
plt.sca(axes.flat[0])
plt.plot(coords, interp, 'k-o', label=f'interpolation (order=3)', alpha=0.7)
plt.grid(axis='x')
plt.xlim(-0.5, npoint-0.5)
plt.ylim(-ylim, ylim)
plt.quiver(coords, interp, np.ones_like(diff1), diff1, diff1, angles='xy', units='xy', scale_units='xy', width=3e-2, cmap='plasma')
plt.plot(np.arange(npoint), data, 'o', color='red', label='data')
plt.legend(loc='upper right')

plt.sca(axes.flat[1])
plt.axhline(0, linestyle=':', color='gray')
plt.plot(coords, diff1, 'k-o', label=f'first derivative', alpha=0.7)
plt.grid(axis='x')
plt.xlim(-0.5, npoint-0.5)
plt.ylim(-ylim, ylim)
plt.quiver(coords, diff1, np.ones_like(diff2), diff2, diff2, angles='xy', units='xy', scale_units='xy', width=3e-2, cmap='plasma')
plt.legend(loc='upper right')



plt.tight_layout()
plt.show()




