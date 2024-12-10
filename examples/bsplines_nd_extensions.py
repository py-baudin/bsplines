import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import bsplines

im0 = np.random.randint(0, 2, [7, 7])
extensions = [
    None,
    "constant",
    "nearest",
    "half-symmetric",
    "whole-symmetric",
    "true-periodic",
]

fig, axes = plt.subplots(nrows=2, ncols=3, num="2d-extensions", figsize=(7, 6))
for i in range(6):
    im = im0
    value = 0
    ext = extensions[i]
    if ext is None:
        value = 0.5
    for axis in [0, 1]:
        im = bsplines.apply_extension(
            im, 5, axis=axis, extension=ext or "constant", value=value
        )
    plt.sca(axes.flat[i])
    plt.imshow(im.T, interpolation="nearest", cmap="gray")
    plt.axis("off")
    if ext is None:
        plt.title("original image")
    else:
        plt.title(ext)

plt.suptitle("Extensions")
plt.tight_layout()
plt.show()
