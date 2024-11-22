import numpy as np
from matplotlib import pyplot as plt
import bsplines


npoint = 10
data = np.random.uniform(size=npoint)

ncoord = 100
coords = np.linspace(0, npoint - 1, ncoord)

plt.figure('interp-1d')

for order in [0, 1, 2, 3]:
    interp = bsplines.interpolate(data, coords, degree=order)
    plt.plot(coords, interp, label=f'order {order}', alpha=0.7)
plt.plot(np.arange(npoint), data, 'o', label='data')
plt.grid(axis='x')
plt.xlabel('coordinates')
plt.ylabel('data')
plt.legend(loc='upper right')
plt.title('1d interpolation with B-Splines')

plt.tight_layout()
plt.show()




