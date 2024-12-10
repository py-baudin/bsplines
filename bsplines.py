"""
n-dimensional B-Splines interpolation

- assumes regularly spaced knots (e.g. image pixels)
- extensions: 
    'constant': use constant value (default=0)
    'nearest': use nearest value
    'half-symmetric' ('mirror'): use center of boundary point as symmetry axis 
    'whole-symmetric' ('reflect'): use extremity of boundary point as symmetry axis
    'anti-symmetric' ('natural'): use center of boundary point as anti-symmetry axis
    'true-periodic' ('grid-wrap'): use points from the opposite side
    'periodic' ('wrap'): use points from the opposite side (first and last must match)
```

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
"""

import math
import numpy as np

try:
    import numba as nb
except ImportError:
    nb = None
NAX = np.newaxis

"""
TODO:
- implement transmitted boundary for prefilter
- use gamma prime and n!beta
- profile/optimize
    - tabulate coefficients for degrees 2,3
    - split transform: 1) evaluate BSplines(coords) and 2) transform(coeffs, BSplines)
    - inplace functions (prefilter/evaluate)
    - vectorized/cupy or parallel/numba
    - derivative w/r f: precompute Jacobian of pre-filter
    - optimize for degrees 0,1

to check:
- direct interpolation with [18]
"""


MAP_EXTENSIONS = {
    # half symmetric (first-derivative = 0)
    "half-symmetric": "half-symmetric",
    "half": "half-symmetric",
    "mirror": "half-symmetric",  # scipy.ndimage alias
    "clamped": "half-symmetric",  # scipy.interpolate alias
    # anti-symmetric (second derivative = 0)
    "anti-symmetric": "anti-symmetric",
    "anti": "anti-symmetric",
    "natural": "anti-symmetric",  # scipy.interpolate alias
    # whole symmetric
    "whole-symmetric": "whole-symmetric",
    "whole": "whole-symmetric",
    "reflect": "whole-symmetric",  # scipy.ndimage alias
    # constant
    "constant": "constant",
    # nearest
    "nearest": "nearest",
    # true-periodic
    "true-periodic": "true-periodic",
    "grid-wrap": "true-periodic",  # scipy.ndimage alias
    # periodic (expects 1st and last points to match)
    "periodic": "periodic",  # scipy.interpolate alias
    "wrap": "periodic",  # scipy.ndimage alias
}


def interpolate(data, coords, degree=3, extension="nearest", **kwargs):
    """interpolation function"""
    bspline = BSpline.prefilter(data, degree=degree, extension=extension, **kwargs)
    if isinstance(coords, (list, tuple)):
        # grid mode
        return bspline(*coords)
    # else: nd mode
    return bspline(coords)


def bspline(data, degree=3, extension="nearest", **kwargs):
    """retrun BSpline object"""
    return BSpline.prefilter(data, degree=degree, extension=extension, **kwargs)


class BSpline:
    """N-dimensional B-Spline"""

    @property
    def degree(self):
        """degree(s) of the B-Spline interpolation"""
        if self.ndim == 1:
            return self._degree[0]
        return self._degree

    @property
    def axes(self):
        """axes of interpolation"""
        return self._axes

    @property
    def ndim(self):
        """number of axes of interpolation"""
        return len(self.axes)

    @property
    def shape(self):
        """shape of the interpolation knot array"""
        return tuple(
            self.coeffs.shape[axis] + min(0, 1 - deg)
            for deg, axis in zip(self._degree, self.axes)
        )

    @property
    def bounds(self):
        """expected bounds of the coordinate arrays"""
        if self._bounds is not None:
            return self._bounds
        return [(0, s - 1) for s in self.shape]

    @classmethod
    def prefilter(cls, data, *, degree=3, bounds=None, axes=None, extension="nearest"):
        """compute B-Spline coefficients and return BSpline object"""
        data = np.asarray(data)
        if axes is None:
            axes = tuple(range(data.ndim))
        else:
            axes = (axes,) if isinstance(axes, int) else tuple(map(int, axes))
        # epsilon = ... # todo
        degree = tuple(degree * np.ones(len(axes), dtype=int))
        coeffs = data
        for i, axis in enumerate(axes):
            coeffs = prefilter_larger(degree[i], coeffs, extension, axis=axis)
        return BSpline(degree, coeffs, axes=axes, bounds=bounds)

    def __init__(self, degree, coeffs, *, axes=None, offset=None, bounds=None):
        """Initialize BSpline object from coefficients"""
        self.coeffs = np.asarray(coeffs)
        ndim = self.coeffs.ndim if axes is None else len(axes)
        self._axes = tuple(range(ndim)) if axes is None else tuple(axes)
        self._degree = tuple(int(v) for v in degree * np.ones(ndim, dtype=int))
        self._offset = None if offset is None else np.asarray(offset)
        if bounds and len(bounds) == 2 and all(isinstance(b, int) for b in bounds):
            bounds = [tuple(bounds)]
        self._bounds = None if bounds is None else list(map(tuple, bounds))

    def __call__(self, *coords):
        """Evaluate B-Spline at given coordinates"""
        ndim = self.ndim
        grid_mode = len(coords) > 1

        if grid_mode:
            # grid mode
            if ndim != len(coords):
                raise ValueError(f"Expected {ndim} coordinates, got: {len(coords)}")
            coords = [np.array(arr) for arr in coords]
            if any(arr.ndim > 1 for arr in coords):
                raise ValueError(
                    f"Expected 1d arrays in grid mode, got: {[arr.shape for arr in coords]}"
                )
        else:
            # 1d or nd mode
            coords = np.atleast_2d(np.copy(coords[0]))
            if coords.shape[0] != ndim:
                raise ValueError(
                    f"First coordinates dimentions must be: {ndim}, got: {coords.shape[0]}"
                )

        for i in range(self.ndim):
            bounds = self.bounds[i]
            if np.any((bounds[0] > coords[i]) | (bounds[1] < coords[i])):
                raise ValueError(f"Out of bound coordinates in axis {i}")

            if self._bounds is not None or self._offset is not None:
                a, b, f = 1.0, 0.0, 0.0
                if self._bounds is not None:
                    shape = self.shape
                    a = (shape[i] - 1) / np.diff(bounds)
                    b = bounds[0]
                if self._offset is not None:
                    f = self._offset[i]
                coords[i] = (coords[i] - b) * a + f

        if ndim == 1:
            return indirect_transform_1d(coords[0], self._degree[0], self.coeffs)
        elif grid_mode:
            return indirect_transform_nd_grid(coords, self._degree, self.coeffs)
        else:
            return indirect_transform_nd(coords, self._degree, self.coeffs)

    def derivative(self, n=1, *, axis=0):
        """Return the `n-th' order derivative as BSpline"""
        coeffs = np.diff(self.coeffs, n=n, axis=self.axes[axis])
        d = np.asarray(self._degree)
        n = n * np.eye(d.size, dtype=int)[axis]
        offset = (n % 2) * (-1) ** d * 0.5
        return BSpline(
            tuple(d - n),
            coeffs,
            offset=tuple(offset),
            axes=self.axes,
            bounds=self._bounds,
        )

    def jacobian(self, *coords, axes=None):
        """return Jacobian matrix for selected axes"""
        jac = []
        for axis in axes or self.axes:
            der = self.derivative(n=1, axis=axis)(*coords)
            jac.append(der)
        return np.stack(jac, axis=-1)


#
# B-spline functions


def indirect_transform_1d(x, n, c, *, out=None):
    x = np.asarray(x)
    c = np.asarray(c)
    if out is None:
        out = 0
    n_ = n // 2
    x0 = np.maximum(np.ceil(x - (n + 1) / 2).astype(int), -n_)
    nax = (...,) + (np.newaxis,) * (c.ndim - 1)
    for k in range(n + 1):
        out += c[x0 + k + n_] * evaluate_bspline(x - (x0 + k), n)[nax]
    return out


def indirect_transform_nd_grid(x, n, c):
    c = np.asarray(c)
    x = [np.asarray(arr) for arr in x]
    ndim = len(x)
    n = (n,) * ndim if isinstance(n, int) else tuple(map(int, n))

    out = c
    for axis in range(ndim):
        c_ = np.moveaxis(out, axis, 0)
        out = indirect_transform_1d(x[axis], n[axis], c_)
        out = np.moveaxis(out, 0, axis)
    return out


def indirect_transform_nd(x, n, c):
    c = np.asarray(c)
    x = np.asarray(x)
    ndim = x.shape[0]
    nxtr = c.ndim - ndim  # non interpolated axes
    n = n * np.ones(ndim, dtype=int)

    n_ = n // 2
    x0 = np.maximum(np.ceil(x.T - (n + 1) / 2).astype(int), -n_).T

    # multiply b-spline functions over all axes
    ks = np.indices(n + 1).reshape(ndim, -1)
    nax = (Ellipsis,) + (NAX,) * (x.ndim - 1)
    b = c[*((ks + n_[:, NAX])[nax] + x0[:, NAX])]
    for i in range(ndim):
        xi = (x[i] - x0[i]) - np.arange(n[i] + 1)[nax]
        b *= evaluate_bspline(xi, n[i])[(ks[i], Ellipsis) + (NAX,) * nxtr]
    out = b.sum(axis=0)
    return out


# same as above, but using more loops and less vectorization
def indirect_transform_nd_loop(x, n, c, *, out=None):
    x = np.asarray(x)
    c = np.asarray(c)
    ndim = x.shape[0]
    # nxtr = c.ndim - ndim # non interpolated axes
    n = (n * np.ones(ndim, dtype=int))[0]

    if out is None:
        out = np.zeros(x.shape[1:], dtype=c.dtype)
    n_ = n // 2
    x0 = np.maximum(np.ceil(x - (n + 1) / 2).astype(int), -n_)

    # b-spline functions
    bs = {
        (i, k): evaluate_bspline(x[i] - (x0[i] + k), n)
        for i in range(ndim)
        for k in np.arange(n + 1)
    }

    # multiply b-spline functions over all axes
    for ks in np.indices([n + 1] * ndim).reshape(ndim, -1).T:
        loc = tuple(x0[i] + ks[i] + n_ for i in range(ndim))
        out += c[loc] * np.prod([bs[(i, ks[i])] for i in range(ndim)], axis=0)
    return out


def evaluate_bspline(x, n):
    """evaluate bspline basis function (or its derivatives)"""
    x = np.fabs(x)

    if n == 0:
        return 1.0 * (x <= 0.5)

    out = np.zeros(x.shape, dtype=float)

    n_ = int(n / 2)
    C, D = get_bspline_coefficients(n)

    # full domain
    radius = (n + 1) / 2
    support = x < radius
    x = x[support]
    _out = out[support]

    k = np.ceil((n + 1) / 2 - x - 1).astype(int)

    for j in range(n_):
        # general case
        mask = k == j
        out_j = _out[mask]
        Cj = C[j]
        y = (n + 1) / 2 - x[mask] - j
        out_j[:] = Cj[-1]
        for i in range(1, n + 1):
            out_j *= y
            out_j += Cj[n - i]
        _out[mask] = out_j

    # k == n_
    mask = k == n_
    x_i = x[mask]
    out_i = _out[mask]
    out_i[:] = D[-1]
    for i in range(1, n + 1):
        out_i *= x_i
        if ((n - i) % 2 == 0) or i == n:
            out_i += D[n - i]
    _out[mask] = out_i

    fn = math.factorial(n)
    out[support] = _out / fn
    return out


if nb:

    FACTORIAL = np.array(
        [
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            39916800,
            479001600,
            6227020800,
            87178291200,
            1307674368000,
            20922789888000,
            355687428096000,
            6402373705728000,
            121645100408832000,
            2432902008176640000,
        ],
        dtype="int64",
    )

    def evaluate_bspline(x, n):
        C, D = get_bspline_coefficients(n)
        out = np.empty(x.shape, dtype=C.dtype)
        return jit_evaluate_bspline(x, n, C, D, out)

    @nb.njit(cache=True, parallel=True)  # (nogil=True, fastmath=True,
    def jit_evaluate_bspline(x, n, C, D, out):
        for i in nb.prange(x.size):
            out.flat[i] = jit_evaluate_bspline_1(x.flat[i], n, C, D)
        return out

    @nb.njit(cache=True, fastmath=True)  # nogil=True, fastmath=True, cache=True)
    def jit_evaluate_bspline_1(x, n, C, D):
        x = np.fabs(x)
        if n == 0:
            return 1.0 * (x <= 0.5)

        radius = (n + 1) / 2
        if x >= radius:
            return 0.0
        k = np.uint16(np.ceil(radius - x)) - 1
        n_ = n // 2
        fn = FACTORIAL[n]

        if k < n_:
            out = C[k, -1]
            y = radius - x - k
            for i in range(1, n + 1):
                out *= y
                out += C[k, n - i]
            return out / fn

        else:  # k == n_
            out = D[-1]
            for i in range(1, n + 1):
                out *= x
                if ((n - i) % 2 == 0) or i == n:
                    out += D[n - i]
            return out / fn

    # def indirect_transform_nd(x, n, c, *, out=None):
    #     global COEFFS
    #     x = np.asarray(x)
    #     c = np.asarray(c)
    #     n = tuple(n * np.ones(len(x), dtype=int))
    #     n1 = tuple(i + 1 for i in n)
    #     CD = [get_bspline_coefficients(i) for i in n]
    #     if out is None:
    #         out = np.empty(x.shape[1:])
    #     return jit_indirect_transform_nd(x, n, n1, c, CD, out)

    # @nb.njit(parallel=True, cache=True, fastmath=True)
    # def jit_indirect_transform_nd(x, n, n1, c, CD, out):
    #     ndim = x.shape[0]
    #     out[:] = 0
    #     n_ = np.array(n) // 2
    #     x0 = np.empty(x.shape, dtype=np.int8)
    #     x0[:] = np.maximum(np.ceil(x.T - np.array(n1) / 2), -n_).T
    #     # b-spline functions
    #     ks = np.indices(n1).reshape(ndim, -1).T
    #     for i in range(out.size):
    #         intp = 0
    #         for k in ks:
    #             b = 1
    #             stride, idx = 1, 0
    #             for d in range(ndim):
    #                 ax = ndim - d - 1
    #                 C, D = CD[ax]
    #                 b *= jit_evaluate_bspline_1(x[ax].flat[i] - x0[ax].flat[i] - k[ax], n[ax], C, D)
    #                 idx += (x0[ax].flat[i] + n_[ax] + k[ax]) * stride
    #                 stride *= c.shape[ax]
    #             intp += b * c.flat[idx]
    #         out.flat[i] = intp
    #     return out


def prefilter_larger(n, f, extension, *, axis=0, epsilon=1e-6):
    """prefiltering algorithm"""
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


def exponential_filter(s, alpha, N, *, out=None):
    """
    s: signal
    alpha: filter coefficient
    N: truncation index (default: L)

    """
    s = np.asarray(s)
    K = s.shape[0]

    if out is None:
        out = np.zeros((K - 2 * N,) + s.shape[1:], dtype=s.dtype)

    # apply causal filter
    out[0] = np.einsum("i,i...->...", alpha ** np.arange(N + 1), s[N:None:-1])
    for i in range(1, K - 2 * N):
        out[i] = s[i + N] + alpha * out[i - 1]

    # apply anti-causal filter
    ls_end = np.einsum("i,i...->...", alpha ** np.arange(N + 1), s[-N - 1 :])
    out[-1] = alpha / (alpha**2 - 1) * (out[-1] + ls_end - s[-N - 1])
    for i in range(K - 2 * N - 2, -1, -1):
        out[i] = alpha * (out[i + 1] - out[i])

    return out


def apply_extension(f, L, extension, *, axis=0, value=0):
    """extend f with one extension among:
    'whole-symmetric', 'half-symmetric', 'periodic', 'constant'
    """
    f = np.asarray(f)
    if f.shape[axis] < L + 1:
        # extend f recursively
        while f.shape[axis] < L + 1:
            L += 1 - f.shape[axis]
            f = apply_extension(f, f.shape[axis] - 1, extension, axis=axis)

    if not extension in MAP_EXTENSIONS:
        raise ValueError(f"Invalid extension type: {extension}")
    ext = MAP_EXTENSIONS[extension]

    slices0 = [slice(None)] * f.ndim
    slices1 = list(slices0)
    if ext == "constant":
        f0 = value + 0 * np.take(f, [0], axis=axis)
        f = np.concatenate([f0] * L + [f] + [f0] * L, axis=axis)
    elif ext == "nearest":
        f0 = np.take(f, [0], axis=axis)
        f1 = np.take(f, [-1], axis=axis)
        f = np.concatenate([f0] * L + [f] + [f1] * L, axis=axis)
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
            raise ValueError(
                "First and last points do not match while periodic extension expected"
            )
        slices0[axis] = slice(-L - 1, -1)
        slices1[axis] = slice(1, L + 1)
        f = np.concatenate([f[tuple(slices0)], f, f[tuple(slices1)]], axis=axis)
    elif ext == "anti-symmetric":
        f0 = f[tuple(slice(None) if i != axis else slice(0, 1) for i in range(f.ndim))]
        f1 = f[
            tuple(slice(None) if i != axis else slice(-1, None) for i in range(f.ndim))
        ]
        slices0[axis] = slice(L, 0, -1)
        slices1[axis] = slice(-2, -L - 2, -1)
        f = np.concatenate(
            [2 * f0 - f[tuple(slices0)], f, 2 * f1 - f[tuple(slices1)]], axis=axis
        )
    return f


# def exponential_filter_bc(s, alpha, N=None, *, cond=..., out=None):
#     """ Exponential filter with boundary condictions

#         s: signal
#         alpha: filter coefficient
#         L: extension length (output indices correspond to slice [L:-L])
#         N: truncation index (default: L)
#     """
#     s = np.asarray(s)
#     K = s.shape[0]
#     if out is None:
#         out = np.zeros_like(s)

#     # apply causal filter
#     out[0] = s[1]/(1 - alpha) # tmp
#     for i in range(1, K):
#         out[i] = s[i] + alpha * out[i - 1]

#     # apply anti-causal filter
#     l_end = out[-2] / (1 - alpha) # tmp
#     out[-1] = alpha / (alpha**2 - 1) * (out[-1] + l_end - s[-1])
#     for i in range(K - 2, -1, -1):
#         out[i] = alpha * (out[i + 1] - out[i])

#     return out


def compute_extension_length(N):
    n_ = len(N)
    L = [n_]
    for j in range(n_):
        L.append(L[-1] + N[-j - 1])
    return L[::-1]


def compute_truncation_indices(n, ndim=1, epsilon=1e-6):
    """compute truncation indices"""
    n_ = int(n / 2)
    z = get_poles(n)
    logz = np.log(np.abs(z))
    mu = [0]  # mu_1
    for k in range(1, n_):
        muk = 1 / (1 + 1 / (logz[k] * np.sum(1 / logz[:k])))
        mu.append(muk)

    rho = np.prod((1 + z) / (1 - z)) ** 2
    eps = epsilon  # rho * epsilon / ndim

    N = []
    for i in range(n_):
        _muprod = np.prod(mu[i + 1 :])
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
    """normalized B-spline coefficients"""
    n_ = int(n / 2)

    # binomials
    bin = [math.comb(n, i) for i in range(0, n + 1)]  # n
    bin_ = [math.comb(n + 1, j) for j in range(0, n_ + 1)]  # n + 1

    # C
    C = np.zeros((n_, n + 1))
    for k in range(n_):
        for j in range(n):
            # j < n
            for i in range(k):
                C[k, j] += (-1) ** i * bin_[i] * (k - i) ** (n - j)
            C[k, j] *= bin[j]
        for i in range(k + 1):
            # j == n
            C[k, n] += (-1) ** i * bin_[i]

    # D
    D = np.zeros(n + 1)
    for j in range(n + 1):
        for i in range(n_ + 1):
            D[j] += (-1) ** i * bin_[i] * ((n + 1) / 2 - i) ** (n - j)
        D[j] *= (-1) ** j * bin[j]

    return C, D


def compute_normalization_constant(n):
    """normalization constant gamma"""
    return 2**n * math.factorial(n) if (n % 2 == 0) else math.factorial(n)


def compute_poles(n):
    """poles of pre-filters"""
    b = compute_polynomial_coefficients(n)
    coeffs = np.concatenate([b[:0:-1], b])
    poly = np.polynomial.Polynomial(coeffs)
    roots = poly.roots()
    return roots[np.abs(roots) < 1]


def compute_polynomial_coefficients(n):
    """polynomial coefficients of the pre-filters"""
    n_ = int(n / 2)
    b = np.zeros(n_ + 1)
    d = np.zeros(n_ + 1)
    b[0] = 1
    d[0] = 1 / 2

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
