# n-dimensional B-Splines interpolation

## Description

A pure `python`/`numpy` implementation of n-dimensional B-Splines.

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
  

## Usage 

```python
# arguments: 
# `data`: (n1, n2, ...) ndim-ndarray
# `coords`: (ndim, npoint) 2d-ndarray, or ndim-list/tuple (nx, ny, ...) of 1d-ndarray
# `degree`: int or ndim-tuple of ints (for different degrees in different dimensions)
# `extension`: str (see above)

# Note that `coords[i]` is assumed to belong to [0, data.shape[i] - 1].
# if `coords` is a 2d ndarray, nd-mode is assummed (see below)
# if `coords` is a list/tuple, grid mode is assumed (see below)

# basic usage: interpolate data at given coordinates
res = bsplines.interpolate(data, coords, degree=3, extension='nearest')

# or simply compute b-spline coefficients, returning a `BSpline` object
spl = bsplines.bspline(data, degree=3, extension='nearest')
# or equivalently
spl = BSpline.prefilter(data, degree=3, ext='nearest')

# nd-mode: interpolate at coords: (ndim x npoint)
intp = spl(coords)
# grid-mode: interpolate at grid points (cx, cy, ...) (much faster, if applicable)
intp = spl(cx, cy)

# compute nth-order derivative BSpline in selected axis
spl_dn = spl.derivative(n, axis=0)
# compute jacobian matrix (1st derivatives at `coords` stacked on last axis)
jac = spl.jacobian(coords)
```

## Examples

1D interpolation and derivatives (`examples/bsplines_1d_derivatives.py`):

[diff-1d](docs/diff-1d.png)

2D interpolation (`examples/bsplines_nd_interp.py.py`)

[2d-splines](docs/bsplines-2d.png)


## Bibliography

Based on
> Briand T, Monasse P, 
  "Theory and Practice of Image B-Spline Interpolation".
  Image Processing On Line 2018; 8:99–141.

and other ressources for the derivatives, grid-mode, etc.

