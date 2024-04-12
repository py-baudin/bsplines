import numpy as np
from scipy import ndimage, interpolate
import bsplines

# tmp
from importlib import reload; reload(bsplines)
from matplotlib import pyplot as plt

def test_bsplines_1d():# 1d
    size = 20
    points = np.arange(size)
    signal = np.random.uniform(size=size)
    locations = np.sort(np.random.uniform(0, size - 1, 1000))

    # degree 1
    _scipy = interpolate.make_interp_spline(points - 0.5, signal, k=0)(locations)
    _bsplines = bsplines.interpolate(signal, locations, degree=0)
    assert np.allclose(_scipy, _bsplines)

    # degree 1
    _scipy = interpolate.make_interp_spline(points, signal, k=1, bc_type=None)(locations)
    _bsplines = bsplines.interpolate(signal, locations, degree=1, extension='clamped')
    assert np.allclose(_scipy, _bsplines)

    # degree 3
    for ext in ['clamped', 'natural']:
        _scipy = interpolate.make_interp_spline(points, signal, k=3, bc_type=ext)(locations)
        _bsplines = bsplines.interpolate(signal, locations, degree=3, extension=ext)
        assert np.allclose(_scipy, _bsplines)


    # derivatives
    spl = interpolate.make_interp_spline(points, signal, k=3, bc_type='natural')
    _scipy = spl.derivative(1)(locations)
    _bsplines = bsplines.interpolate(signal, locations, degree=3, extension='natural', order=1)
    assert np.allclose(_scipy, _bsplines, atol=1e-7)

    _scipy = spl.derivative(2)(locations)
    _bsplines = bsplines.interpolate(signal, locations, degree=3, extension='natural', order=2)
    
    plt.figure(); plt.plot(locations, _scipy); plt.plot(locations, _bsplines); plt.show()
    assert np.allclose(_scipy, _bsplines, atol=1e-7)



def test_bspline_nd():
    size = (25, 25)
    points = np.indices(size)
    signal = np.random.uniform(size=size)
    locations = np.indices([100, 100]) / 100 * 24

    for degree in range(4):
        for mode in ['reflect', 'mirror', 'nearest', 'grid-wrap']:
            _scipy = ndimage.map_coordinates(signal, locations, order=degree, mode=mode)
            _bsplines = bsplines.interpolate(signal, locations, degree=degree, extension=mode)
            assert np.allclose(_scipy, _bsplines)

    # derivatives
    x = np.array([1, 0])[:, np.newaxis, np.newaxis] * 1e-8
    _scipy0 = ndimage.map_coordinates(signal, locations, order=3, mode='reflect')
    _scipyx = (ndimage.map_coordinates(signal, locations + x, order=3, mode='reflect') - _scipy0) * 1e8
    y = np.array([0, 1])[:, np.newaxis, np.newaxis] * 1e-8
    _scipyy = (ndimage.map_coordinates(signal, locations + y, order=3, mode='reflect') - _scipy0) * 1e8
    _bsplines = bsplines.interpolate(signal, locations, degree=3, extension='reflect', order=1)

    plt.figure(); plt.imshow(_scipyx); plt.figure(); plt.imshow(_bsplines[0]); plt.show()
    assert np.allclose(_scipyx, _bsplines[0], atol=1e-7)
    assert np.allclose(_scipyy, _bsplines[1], atol=1e-7)

