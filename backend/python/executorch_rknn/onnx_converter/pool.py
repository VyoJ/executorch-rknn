import numpy as np

from .builder import converts


@converts("aten.max_pool2d_with_indices.default")
def max_pool2d(b, n):
    x, kernel, stride, padding, dilation, ceil = (list(n.args) + [None] * 6)[:6]
    padding = list(padding or [0, 0])
    b.emit("MaxPool", [b.name(x)], n, kernel_shape=list(kernel), strides=list(stride or kernel),
           pads=padding * 2, dilations=list(dilation or [1, 1]), ceil_mode=int(bool(ceil)))


def _check_mean(n):
    x, dims = n.args[:2]
    rank = len(x.meta["val"].shape)
    if rank != 4 or sorted(d % 4 for d in dims) != [2, 3]:
        # RV1106 has no ReduceMean, on the NPU or in its CPU fallback.
        return f"mean over dims {list(dims)} of a rank-{rank} tensor"
    return None


@converts("aten.mean.dim", check=_check_mean)
def mean(b, n):
    x, _, keepdim = (list(n.args) + [False])[:3]
    if keepdim:
        b.emit("GlobalAveragePool", [b.name(x)], n)
    else:
        pooled = b.emit_tmp("GlobalAveragePool", [b.name(x)], f"{n.name}__pooled")
        b.emit("Reshape", [pooled, b.const(list(n.meta["val"].shape), np.int64)], n)
