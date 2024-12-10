import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import bsplines

NAX = np.newaxis

degree = 3
nx, ny = (20, 20)
nd = 20
nobs = 100
rgen = np.random  # np.random.RandomState(0)
SNR = 20


# model
alpha = rgen.uniform(0, 2, nd)
eta = rgen.uniform(0, 2 * np.pi, nd)
phi = rgen.uniform(0, 2 * np.pi, nd)
gamma = rgen.uniform(0, 2, nd)


def f(params):
    x, w = params
    return alpha * np.cos(eta * x[..., NAX] + phi) + gamma * w[..., NAX] ** 3


def jf(params):
    x, w = params
    angle = eta * x[..., NAX] + phi
    dfx = -eta * alpha * np.sin(angle)
    dfw = 3 * gamma * w[..., NAX] ** 2
    return np.stack([dfx, dfw], axis=-1)


# generate dictionary
grid = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
dct = f(grid)

# interpolator
spl = bsplines.BSpline.prefilter(dct, axes=(0, 1), bounds=[(0, 1), (0, 1)])

# generate observations
gt = tuple(np.random.uniform(0, 1, (2, nobs)))
obs = f(gt)

noise = np.random.normal(size=(nobs, nd))
obs += 1 / SNR * np.linalg.norm(obs, axis=1)[:, NAX] * noise


def costfun(f, jf):
    def parse(args):
        return tuple(np.reshape(args, (2, -1)))

    def cost(args):
        fev = f(parse(args))
        res = fev - obs
        return np.sum(res**2) / nobs

    def jac(args):
        args = parse(args)
        fev = f(args)
        jfev = jf(args)
        res = fev - obs
        grad = 2 * np.einsum("omp,om->po", jfev, res)
        return grad.ravel() / nobs

    cost.jac = jac
    cost.parse = parse
    return cost


# fit
bounds = [(0, 1)] * nobs * 2
init = [0.5] * nobs * 2

cf = costfun(f, jf)
res = optimize.minimize(
    cf, init, jac=cf.jac, bounds=bounds, options={"disp": True, "maxiter": 500}
)
sol = cf.parse(res.x)
pred = f(sol)
res = np.linalg.norm(obs - pred, axis=1) / np.linalg.norm(obs, axis=1)

# fit with spline model
cf2 = costfun(spl, spl.jacobian)
res2 = optimize.minimize(
    cf2, init, jac=cf2.jac, bounds=bounds, options={"disp": True, "maxiter": 500}
)
sol2 = cf.parse(res2.x)
pred2 = f(sol2)
res2 = np.linalg.norm(obs - pred2, axis=1) / np.linalg.norm(obs, axis=1)


#
# plot

pdiff = (gt[0] - sol[0], gt[0] - sol2[0], gt[1] - sol[1], gt[1] - sol2[1])
resids = (res, res2)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]

fig, axes = plt.subplot_mosaic("AAB", num="results")
plt.sca(axes["A"])
h = plt.boxplot(pdiff, positions=(0.8, 1.2, 1.8, 2.2), widths=0.35, patch_artist=True)
for i, patch in enumerate(h["boxes"]):
    patch.set_facecolor(colors[i % 2])
    patch.set_alpha(0.5)
plt.ylabel("Error (a.u.)")
plt.title("Estimation error")
plt.legend(h["boxes"][:2], ["original", "interpolated"], loc="upper left")
plt.xticks(ticks=[1, 2], labels=["parameter 1", "parameter 2"])

plt.sca(axes["B"])
h = plt.boxplot(resids, widths=0.8, patch_artist=True)
for i, patch in enumerate(h["boxes"]):
    patch.set_facecolor(colors[i % 2])
    patch.set_alpha(0.5)
plt.xlim(0.5, 2.5)
plt.xticks([], [])
plt.ylabel("L1-residual (a.u.)")
plt.title("Residuals")

plt.suptitle(f"Original vs interpolated model (n={spl.shape})")
plt.tight_layout()
plt.show()
