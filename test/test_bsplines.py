import numpy as np
import bsplines

def test_derivatives():
    size = 10
    yvals = np.random.uniform(size=size)
    xcoords = np.sort(np.random.uniform(0, size - 1, 100))

    # compare derivatives at interpolation points
    eps = 1e-8
    icoords = np.arange(size - 1)
    spl1 = bsplines.interpolate(yvals, icoords, degree=3)
    spl2 = bsplines.interpolate(yvals, icoords + eps, degree=3)
    fdspl = (spl2 - spl1) * 1/eps
    dspl = bsplines.interpolate(yvals, icoords, degree=3, order=1)

    assert np.allclose(fdspl, dspl)
