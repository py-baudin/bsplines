"""
> Briand T, Monasse P
  Theory and Practice of Image B-Spline Interpolation.
  Image Processing On Line 2018; 8:99–141.

2 methods
- direct interpolation for n=0, and n=1
- 2-step (prefiltering, filtering) for n>1

to check:
- direct interpolation with [18]

"""
import math
import itertools
import numpy as np
NAX = np.newaxis

"""
TODO:
- fix extension length N
- implement transmitted boundary for prefilter
- use gamma prime and n!beta
- profile/optimize
    - tabulate coefficients for degrees 2,3
    - split transform: 1) evaluate BSplines(coords) and 2) transform(coeffs, BSplines)
    - inplace functions (prefilter/evaluate)
    - vectorized/cupy or parallel/numba
    - derivative w/r f: precompute Jacobian of pre-filter
    - optimize for degrees 0,1
"""


MAP_EXTENSIONS = {
    # half symmetric (first-derivative = 0)
    'half-symmetric': 'half-symmetric',
    'half': 'half-symmetric',
    'mirror': 'half-symmetric', # scipy.ndimage alias
    'clamped': 'half-symmetric', # scipy.interpolate alias
    # anti-symmetric (second derivative = 0)
    'anti-symmetric': 'anti-symmetric',
    'anti': 'anti-symmetric',
    'natural': 'anti-symmetric', # scipy.interpolate alias
    # whole symmetric
    'whole-symmetric': 'whole-symmetric',
    'whole': 'whole-symmetric',
    'reflect': 'whole-symmetric', # scipy.ndimage alias
    # constant
    'constant': 'constant',
    'nearest': 'constant',
    # true-periodic
    'true-periodic': 'true-periodic', 
    # periodic (expects 1st and last points to match)
    'periodic': 'periodic', # scipy.interpolate
    'grid-wrap': 'periodic', # scipy.ndimage alias
}


def interpolate(data, coords, *, degree=3, maxdim=None, order=0, extension='natural', **kwargs):
    """ interpolation function """
    data = np.asarray(data)
    kwargs['extension'] = extension

    if data.ndim == 1:
        bspline = BSpline.setup(data, degree=degree, **kwargs)
    else:
        bspline = BSplineND.setup(data, degree=degree, ndim=maxdim, **kwargs)

    if order > 0:
        bspline = bspline.derivative(order)
    return bspline(coords)


class BSpline:

    @classmethod
    def setup(cls, *args, **kwargs):
        obj = cls()
        obj.prefilter(*args, **kwargs)
        return obj


    def __init__(self, coeffs=None, *, degree=3, offset=None):
        self.degree = degree
        self.coeffs = coeffs
        self.offset = offset

    def prefilter(self, data, *, degree=3, **kwargs):
        coeffs = prefilter_larger(data, degree, **kwargs)
        self.coeffs = coeffs
        self.degree = degree

    def __call__(self, coords):
        if self.offset is not None:
            coords = coords + self.offset
        return indirect_transform_1d(coords, self.degree, self.coeffs)

    def derivative(self, order=1):
        coeffs = np.diff(self.coeffs, n=order)
        degree = self.degree - order
        offset = -0.5 if bool(order % 2) else None
        return BSpline(coeffs, degree=degree, offset=offset)


class BSplineND:
    """ N-dimensional B-Spline """

    @classmethod
    def setup(cls, *args, **kwargs):
        obj = cls()
        obj.prefilter(*args, **kwargs)
        return obj

    def __init__(self, coeffs=None, *, degree=3, ndim=None):
        self.degree = degree
        self.ndim = ndim
        self.coeffs = coeffs

    def prefilter(self, data, *, degree=3, ndim=None, **kwargs):
        data = np.asarray(data)
        ndim = data.ndim if ndim is None else int(ndim)
        epsilon = ... # todo
        # coefficients
        coeffs = data
        for axis in range(ndim):
            coeffs = prefilter_larger(coeffs, degree, axis=axis, **kwargs)
        self.coeffs = coeffs
        self.degree = degree
        self.ndim = ndim

    def __call__(self, coords):
        return indirect_transform_nd(coords, self.degree, self.coeffs, maxdim=self.ndim)

    def derivative(self, coords, order=1, axes=None):
        derivs = []
        for axis in range(self.ndim):
            if axes and not axis in axes:
                continue
            deriv = indirect_transform_nd(coords, self.degree, self.coeffs, maxdim=self.ndim, order=order, axis=axis)
            derivs.append(deriv)
        return derivs


def jacobian(size, coords, *, degree=3, extension='constant', epsilon=1e-6):
    """ evaluate the BSpline Jacobian at the given coordinates """
    coords = np.asarray(coords)
    shape = (size,) if isinstance(size, int) else tuple(size)
    ndim = len(shape)
    jac = np.zeros(coords.shape[1:] + shape)

    n = np.array(degree) * np.ones(ndim, dtype=int)
    n_ = n // 2

    x0 = np.ceil(coords.T - (n + 1) / 2).T.astype(int)
    indices = [np.arange(n[i] + 1) for i in range(ndim)]

    # center = tuple(np.array(shape) // 2)
    points = np.eye(np.prod(shape)).reshape(shape+ shape)
    coeffs = points
    for i in range(ndim):
        coeffs = prefilter_larger(coeffs, degree, extension=extension, epsilon=epsilon, axis=i)

    newx = (Ellipsis,) + (np.newaxis,)*ndim
    funcs = {
        (i, k): evaluate_bspline(coords[i] - (x0[i] + k), n[i])[newx]
        for i in range(ndim) for k in indices[i]
    }


    # b-spline functions
    for k in itertools.product(*indices):
        loc = tuple(x0[i] + k[i] + n_[i] for i in range(ndim))
        jac += coeffs[loc] * np.prod([funcs[(i, k[i])] for i in range(ndim)], axis=0)

    return jac


#
# B-spline functions

def indirect_transform_1d(x, n, c, *, order=0, out=None):
    x = np.asarray(x)
    c = np.asarray(c)
    if out is None:
        out = np.zeros_like(x, dtype=c.dtype)

    if order > 0:
        #c1, c2 = (c[1:], c[:-1]) if n % 2 == 0 else (c, c)
        # c1, c2 = (c, c)
        # breakpoint()
        c_ = c[1:] - c[:-1]
        x = x + 0.5 * (-1)**n
        return indirect_transform_1d(x, n - 1, c_, order=order - 1)
        # out += indirect_transform_1d(x + 0.5, n - 1, c, order=order - 1)
        # out -= indirect_transform_1d(x - 0.5, n - 1, c, order=order - 1)
        # return out 
    else:
        n_ = int(n / 2)
        x0 = np.maximum(np.ceil(x - (n + 1) / 2).astype(int), -n_)
        for k in range(n + 1):
            out += c[x0 + k + n_] * evaluate_bspline(x - (x0 + k), n)
    return out

#
# more vectorized
def indirect_transform_nd(coords, degree, coeffs, *, maxdim=None, order=0, axis=0, out=None):
    coords = np.asarray(coords)
    coeffs = np.asarray(coeffs)

    ndim = coeffs.ndim if maxdim is None else int(maxdim)

    shape = coords.shape[1:] + coeffs.shape[ndim:]
    if out is None:
        out = np.zeros(shape, dtype=coeffs.dtype)
    elif out.shape != shape:
        raise ValueError(f'Invalid output shape: {out.shape} (expected: {shape})')

    n = int(degree) # integer
    n_ = n // 2

    x0 = np.ceil(coords - (n + 1) / 2).astype(int)

    # b-spline functions
    coords = np.stack([coords[:ndim] - (x0 + k) for k in range(n + 1)], axis=-1)
    funcs = evaluate_bspline(coords, n, order=order)

    idx = np.indices([n + 1] * ndim).reshape(ndim, -1)
    # multiply b-spline functions over all axes
    weights = np.prod(funcs[np.arange(ndim)[:, NAX], ..., idx], axis=0)

    nax = (Ellipsis,) + (NAX,) * ndim
    loc = tuple(x0[i] + idx[i][nax] + n_ for i in range(ndim))
    # sum over all coefficients
    out = (coeffs[loc] * weights).sum(axis=0)
    return out

#
# base version
def _indirect_transform_nd(coords, degree, coeffs, *, maxdim=None, order=0, axis=0, out=None):
    coords = np.asarray(coords)
    coeffs = np.asarray(coeffs)

    maxdim = coeffs.ndim if maxdim is None else int(maxdim)

    shape = coords.shape[1:] + coeffs.shape[maxdim:]
    if out is None:
        out = np.zeros(shape, dtype=coeffs.dtype)
    elif out.shape != shape:
        raise ValueError(f'Invalid output shape: {out.shape} (expected: {shape})')

    n = degree
    n_ = n // 2

    x0 = np.ceil(coords - (n + 1) / 2).astype(int)
    indices = [np.arange(n + 1) for i in range(maxdim)]

    # b-spline functions
    newx = tuple([Ellipsis] + [NAX] * (coeffs.ndim - maxdim))
    funcs = {
        (i, k): evaluate_bspline(coords[i] - (x0[i] + k), n, order=order * (i==axis))[newx]
        for i in range(maxdim) for k in indices[i]
    }

    for k in itertools.product(*indices):
        loc = tuple(x0[i] + k[i] + n_ for i in range(maxdim))
        out += coeffs[loc] * np.prod([funcs[(i, k[i])] for i in range(maxdim)], axis=0)
    return out


# vectorized
def evaluate_bspline(x, n):
    """ evaluate bspline basis function (or its derivatives) """
    x = np.abs(x)

    if n == 0:
        return 1.0 * (x <= 0.5)

    # shape = x.shape
    # x, indices = np.unique(x, return_inverse=True)
    out = np.zeros(x.shape, dtype=float)

    n_ = int(n / 2)
    C, D = get_bspline_coefficients(n)

    # full domain
    radius = (n + 1) / 2
    fn = math.factorial(n)
    support = x < radius
    x = x[support]
    _out = out[support]

    k = np.ceil((n + 1) / 2 - x).astype(int) - 1

    for j in range(n_):
        # general case
        mask = (k == j)
        out_j = _out[mask]
        Cj = C[j]
        y = (n + 1) / 2 - x[mask] - j
        out_j[:] = Cj[-1]
        for i in range(1, n + 1):
            out_j *= y
            out_j += Cj[n - i]
        _out[mask] = out_j

    # k == n_
    mask = (k == n_)
    x_i = x[mask]
    out_i = _out[mask]
    out_i[:] = D[-1]
    for i in range(1, n + 1):
        out_i *= x_i
        if (i % 2) or i == n:
            out_i += D[n - i]
    _out[mask] = out_i
    out[support] = _out / fn
    return out

#
# #
# # numba version
# import numba as nb
# #
# def _indirect_transform_nd(coords, degree, coeffs, *, maxdim=None, order=0, axis=0, out=None):
#     coords = np.asarray(coords)
#     coeffs = np.asarray(coeffs)
#
#     ndim = coeffs.ndim if maxdim is None else int(maxdim)
#
#     shape = coords.shape[1:] + coeffs.shape[ndim:]
#     if out is None:
#         out = np.zeros(shape, dtype=coeffs.dtype)
#     elif out.shape != shape:
#         raise ValueError(f'Invalid output shape: {out.shape} (expected: {shape})')
#
#     C, D = get_bspline_coefficients(degree)
#     fn = math.factorial(degree)
#     coeffs = np.ascontiguousarray(coeffs)
#     coords = coords.reshape(coords.shape[0], -1).T
#     indices = np.indices([degree + 1] * ndim).reshape(ndim, -1).T
#     out = _jit_indirect_transform(coords, degree, indices, coeffs, C, D, out)
#     return out.reshape(shape) / fn ** ndim
#
#
#
# @nb.njit(parallel=True, cache=True, fastmath=True)
# def _jit_indirect_transform(coords, n, indices, coeffs, C, D, out):
#     npoint = len(coords)
#     nindex = len(indices)
#     ndim  = coords.shape[1]
#     n_ = n // 2
#     for i in nb.prange(npoint):
#         value = 0.0
#         for ik in nb.prange(nindex):
#             idx = 0
#             weight = 1.0
#             for axis in nb.prange(ndim):
#                 x = coords[i, axis]
#                 k = indices[ik, axis]
#                 x0 = math.ceil(x - (n + 1) / 2)
#                 _idx = (int(x0) + k + n_) * coeffs.strides[axis]
#                 if _idx < 0:
#                     break
#                 idx += _idx
#                 loc = x - x0 - k
#                 weight *= jit_evaluate_bspline(loc, n, C, D)
#             if _idx < 0:
#                 continue
#             value += coeffs.flat[idx//coeffs.itemsize] * weight
#         out.flat[i] = value
#     return out
#
#
# @nb.njit(nogil=True, cache=True, fastmath=True)
# def jit_evaluate_bspline(x, n, C, D):
#     x = abs(x)
#
#     if n == 0:
#         return 1.0 * (x <= 0.5)
#
#     radius = (n + 1) / 2
#     if x >= radius:
#         return 0.0
#
#     k = math.ceil(radius - x) - 1
#     n_ = int(n / 2)
#
#     if k < n_:
#         out = C[k, -1]
#         y = radius - x - k
#         for i in range(1, n + 1):
#             out *= y
#             out += C[k, n - i]
#         return out
#
#     else: # k == n_
#         out = D[-1]
#         for i in range(1, n + 1):
#             out *= x
#             if (i % 2) or i == n:
#                 out += D[n - i]
#         return out
#


def prefilter_larger(f, n, axis=0, extension='constant', epsilon=1e-6):
    """ prefiltering algorithm """
    if n < 2:
        return f

    f = np.asarray(f)
    K = f.shape[axis]
    n_ = n // 2

    if axis != 0:
        f = np.moveaxis(f, axis, 0)

    # get poles, normalization and trucation index
    z = get_poles(n)
    gamma = get_normalization_coefficient(n)

    N = compute_truncation_indices(n, epsilon=epsilon)
    L = compute_extension_length(N)

    # prefilter
    c = apply_extension(f, L[0], extension, axis=0)

    for i in range(n_):
        c = exponential_filter(c, z[i], N[i])


    # renormalize
    c *= gamma

    if axis != 0:
        c = np.moveaxis(c, 0, axis)
    return c


def exponential_filter(s, alpha, N, out=None):
    """
        s: signal
        alpha: filter coefficient
        N: truncation index (default: L)

    """
    s = np.asarray(s)
    K = s.shape[0]
    
    if out is None:
        out = np.zeros((K - 2*N,) + s.shape[1:], dtype=s.dtype)

    # apply causal filter
    out[0] = np.einsum('i,i...->...', alpha**np.arange(N + 1), s[N:None:-1])
    for i in range(1, K - 2*N):
        out[i] = s[i + N] + alpha * out[i - 1]

    # apply anti-causal filter
    ls_end = np.einsum('i,i...->...', alpha**np.arange(N + 1), s[-N - 1:])
    out[-1] = alpha / (alpha**2 - 1) * (out[-1] + ls_end - s[-N - 1])
    for i in range(K - 2 * N - 2, -1, -1):
        out[i] = alpha * (out[i + 1] - out[i])

    return out



def apply_extension(f, L, extension, axis=0):
    """ extend f with one extension among:
        'whole-symmetric', 'half-symmetric', 'periodic', 'constant'
    """
    f = np.asarray(f)
    if f.shape[axis] < L + 1: 
        # extend f recursively
        while f.shape[axis] < L + 1:
            L += 1 - f.shape[axis]
            f = apply_extension(f, f.shape[axis] - 1, extension, axis=axis)

    if not extension in MAP_EXTENSIONS:
        raise ValueError(f'Invalid extension type: {extension}')
    ext = MAP_EXTENSIONS[extension]
    
    slices0 = [slice(None)] * f.ndim
    slices1 = list(slices0)
    if ext == 'constant':
        f0 = np.take(f, [0], axis=axis)
        f1 = np.take(f, [-1], axis=axis)
        f = np.concatenate([f0]*L + [f] +  [f1] * L, axis=axis)
    elif ext == "whole-symmetric":
        slices0[axis] = slice(L - 1, None, -1)
        slices1[axis] = slice(-1, -L - 1, -1)
        f = np.concatenate([f[tuple(slices0)], f, f[tuple(slices1)]], axis=axis)
    elif ext == "half-symmetric":
        slices0[axis] = slice(L, 0, -1)
        slices1[axis] = slice(-2, -L - 2, -1)
        f = np.concatenate([f[tuple(slices0)], f, f[tuple(slices1)]], axis=axis)
    elif ext == "true-periodic":
        slices0[axis] = slice(-L, None)
        slices1[axis] = slice(L)
        f = np.concatenate([f[tuple(slices0)], f, f[tuple(slices1)]], axis=axis)
    elif ext == "periodic":
        if not np.allclose(f.take(-1, axis=axis), f.take(0, axis=axis)):
            raise ValueError('First and last points do not match while periodic extension expected')
        slices0[axis] = slice(-L - 1, -1)
        slices1[axis] = slice(1, L + 1)
        f = np.concatenate([f[tuple(slices0)], f, f[tuple(slices1)]], axis=axis)
    elif ext == "anti-symmetric":
        f0 = f[tuple(slice(None) if i != axis else slice(0, 1) for i in range(f.ndim))]
        f1 = f[tuple(slice(None) if i != axis else slice(-1, None) for i in range(f.ndim))]
        slices0[axis] = slice(L, 0, -1)
        slices1[axis] = slice(-2, -L - 2, -1)
        f = np.concatenate([2*f0 - f[tuple(slices0)], f, 2*f1 - f[tuple(slices1)]], axis=axis)
    return f


def exponential_filter_bc(s, alpha, N=None, *, cond=..., out=None):
    """ Exponential filter with boundary condictions

        s: signal
        alpha: filter coefficient
        L: extension length (output indices correspond to slice [L:-L])
        N: truncation index (default: L)
    """
    s = np.asarray(s)
    K = s.shape[0]
    if out is None:
        out = np.zeros_like(s)

    # apply causal filter
    out[0] = s[1]/(1 - alpha) # tmp
    for i in range(1, K):
        out[i] = s[i] + alpha * out[i - 1]

    # apply anti-causal filter
    l_end = out[-2] / (1 - alpha) # tmp
    out[-1] = alpha / (alpha**2 - 1) * (out[-1] + l_end - s[-1])
    for i in range(K - 2, -1, -1):
        out[i] = alpha * (out[i + 1] - out[i])

    return out


def compute_extension_length(N):
    n_ = len(N)
    L = [n_]
    for j in range(n_):
        L.append(L[-1] + N[-j - 1])
    return L[::-1]


def compute_truncation_indices(n, ndim=1, epsilon=1e-6):
    """ compute truncation indices """
    n_ = int(n/2)
    z = get_poles(n)
    logz = np.log(np.abs(z))
    mu = [0] # mu_1
    for k in range(1, n_):
        muk = 1 / (1 + 1 / (logz[k] * np.sum(1 / logz[:k])))
        mu.append(muk)

    rho = np.prod((1 + z)/(1 - z))**2
    eps = epsilon #rho * epsilon / ndim

    N = []
    for i in range(n_):
        _muprod = np.prod(mu[i + 1:])
        _log = math.log(eps * rho * (1 - z[i]) * (1 - mu[i]) * _muprod)
        _N = int(_log / logz[i]) + 1
        N.append(_N)

    return N




def get_poles(n):
    global POLES
    if not n in POLES:
        POLES[n] = compute_poles(n)
    return POLES[n]

def get_normalization_coefficient(n):
    global GAMMA
    if not n in GAMMA:
        GAMMA[n] = compute_normalization_constant(n)
    return GAMMA[n]

def get_bspline_coefficients(n):
    global COEFFS
    if not n in COEFFS:
        COEFFS[n] = compute_bspline_coefficients(n)
    return COEFFS[n]


# pre-computed values

POLES = {}
GAMMA = {}
COEFFS = {}


#
# offline


def compute_bspline_coefficients(n):
    """ normalized B-spline coefficients """
    n_ = int(n / 2)

    # binomials
    bin = [math.comb(n, i) for i in range(0, n + 1)] # n
    bin_ = [math.comb(n + 1, j) for j in range(0, n_ + 1)] # n + 1

    # C
    C = np.zeros((n_, n + 1))
    for k in range(n_):
        for j in range(n):
            # j < n
            for i in range(k):
                C[k, j] += (-1)**i * bin_[i] * (k - i)**(n - j)
            C[k, j] *= bin[j]
        for i in range(k + 1):
            # j == n
            C[k, n] += (-1)**i * bin_[i]

    # D
    D = np.zeros(n + 1)
    for j in range(n + 1):
        for i in range(n_ + 1):
            D[j] += (-1)**i * bin_[i] * ((n + 1) / 2 - i)**(n - j)
        D[j] *= (-1)**j * bin[j]

    return C, D



def compute_normalization_constant(n):
    """ normalization constant gamma """
    return 2**n * math.factorial(n) if (n%2 == 0) else math.factorial(n)

def compute_poles(n):
    """ poles of pre-filters """
    b = compute_polynomial_coefficients(n)
    coeffs = np.concatenate([b[:0:-1], b])
    poly = np.polynomial.Polynomial(coeffs)
    roots = poly.roots()
    return roots[np.abs(roots)<1]

def compute_polynomial_coefficients(n):
    """ polynomial coefficients of the pre-filters """
    n_ = int(n / 2)
    b = np.zeros(n_ + 1)
    d = np.zeros(n_ + 1)
    b[0] = 1
    d[0] = 1/2

    # degree 1...n_-1 coefficients
    for m in range(1, n):
        m_ = int(m / 2)
        dk = d[0]
        for k in range(m_ + 1):
            dk_1 = dk
            dk = d[k]
            d[k] = 1 / m * (((m + 2) / 2 + k) * b[min(k + 1, n_)] + (m / 2 - k) * b[k])
            b[k] = 1 / m * (((m + 1) / 2 + k) * dk + ((m + 1) / 2 - k) * dk_1)

    # degree n_ coefficients
    for k in range(n_ + 1):
        b[k] = 1 / n * (((n + 1) / 2 + k) * d[k] + ((n + 1) / 2 - k) * d[max(k - 1, 0)])

    return b
