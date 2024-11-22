import numpy as np
from scipy import ndimage, interpolate
import timeit
import bsplines
from importlib import reload
reload(bsplines)



#
#
# # 1d small
# num = 1000
# size = 20
# points = np.arange(size)
# signals = np.random.uniform(size=(num, size))
# locations = np.sort(np.random.uniform(0, size - 1, (num, 1000)))
#
# def _scipy_1(signal, loc, order=3, bc='natural'):
#     return interpolate.make_interp_spline(points, signal, k=order, bc_type=bc)(loc)
#
# def _bsplines_1(signal, loc, order=3, bc='natural'):
#     return bsplines.interpolate(signal, loc, order=order, extension=bc)
#
# print("1d small")
# opts = {'number': 10000, 'repeat': 7}
# res1 = timeit.repeat('(_scipy_1(signals[i], locations[i]) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res1):.2f}us per loop')
# res2 = timeit.repeat('(_bsplines_1(signals[i], locations[i]) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res2):.2f}us per loop')
#
#
# # 1d large
# num = 1000
# size = 100000
# points = np.arange(size)
# signals = np.random.uniform(size=(num, size))
# locations = np.sort(np.random.uniform(0, size - 1, (num, 1000)))
#
# def _scipy_2(signal, loc, order=3, bc='natural'):
#     return interpolate.make_interp_spline(points, signal, k=order, bc_type=bc)(loc)
#
# def _bsplines_2(signal, loc, order=3, bc='natural'):
#     return bsplines.interpolate(signal, loc, order=order, extension=bc)
#
# print("1d large")
# opts = {'number': 10000, 'repeat': 7}
# res1 = timeit.repeat('(_scipy_2(signals[i], locations[i]) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res1):.2f}us per loop')
# res2 = timeit.repeat('(_bsplines_2(signals[i], locations[i]) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res2):.2f}us per loop')
#
#
# # 2d small
# num = 1000
# size = (25, 25)
# signals = np.random.uniform(size=(num,) + size)
# locations = np.indices([100, 100]) / 99 * 24
#
# def _scipy_3(signal, loc, order=3):
#     return ndimage.map_coordinates(signal, loc, order=order, mode='reflect')
#
# def _bsplines_3(signal, loc, order=3):
#     return bsplines.interpolate(signal, loc, order=order, extension='reflect')
#
#
# print("2d small")
# opts = {'number': 10000, 'repeat': 7}
# res = timeit.repeat('(_scipy_3(signals[i], locations) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res):.2f}us per loop')
# res = timeit.repeat('(_bsplines_3(signals[i], locations) for i in range(num))', globals=globals(), **opts)
# print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res):.2f}us per loop')


# 2d large
num = 1000
size = (1000, 1000)
signals = np.random.uniform(size=(num,) + size)
locations = np.indices([1000, 1000]) / 999 * 999

def _ndimage_4(signal, loc, order=3):
    return ndimage.map_coordinates(signal, loc, order=order, mode='reflect')

def _bsplines_4(signal, loc, order=3):
    return bsplines.interpolate(signal, loc, order=order, extension='reflect')


print("2d large")
opts = {'number': 100, 'repeat': 7}
res = timeit.repeat('(_ndimage_4(signals[i], locations) for i in range(num))', globals=globals(), **opts)
print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res):.2f}us per loop')
res = timeit.repeat('(_bsplines_4(signals[i], locations) for i in range(num))', globals=globals(), **opts)
print(f'{opts["number"]} loops, best of {opts["repeat"]}: {1e6* np.min(res):.2f}us per loop')
