# n-dimensional B-Splines interpolation

A pure python/numpy implementation of n-dimensional B-Splines.

This is currently *not* faster than `scipy.interpolate.make_interp_spline` or `scipy.ndimage.map_coordinates`.
It might be more flexible in some situations, and less in others.

- (for now) assumes regularly spaced knots (e.g. image pixels)
- extensions:
    - `'constant'`: use constant value (default=0)
    - `'nearest'`: use nearest value
    - `'half-symmetric'` (`'mirror'`): use center of boundary point as symmetry axis 
    - `'whole-symmetric'` (`'reflect'`): use extremity of boundary point as symmetry axis
    - `'anti-symmetric'` (`'natural'`): use center of boundary point as anti-symmetry axis
    - `'true-periodic'` (`'grid-wrap'`): use points from the opposite side
    - `'periodic'` (`'wrap'`): use points from the opposite side (first and last must match)
  
```python
# basic usage: interpolate data at given coordinates
res = bsplines.interpolate(data, coords, degree=3, extension='nearest')

Note that `coords[i]` is assumed to belong to [0, data.shape[i] - 1].

# or simply compute BSpline coefficients
spl = bsplines.bspline(data, degree=3, extension='nearest')
# or equivalently
spl = BSpline.prefilter(data, degree=3, ext='nearest')

# interpolate at coords: (ndim x npoint)
intp = spl(coords)
# interpolate at grid points (cx, cy, ...) (much faster)
intp = spl(cx, cy)

# compute nth-order derivative BSpline in selected axis
spl_dn = spl.derivative(n, axis=0)
# compute jacobian matrix (stacked 1st derivatives)
jac = spl.jacobian(coords)
```

Based on:
> Briand T, Monasse P
  Theory and Practice of Image B-Spline Interpolation.
  Image Processing On Line 2018; 8:99–141.
