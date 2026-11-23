import numpy as np

from .builder import converts


@converts("aten.view_copy.default")
def view(b, n):
    b.emit("Reshape", [b.name(n.args[0]), b.const(list(n.args[1]), np.int64)], n)


@converts("aten.permute_copy.default")
def permute(b, n):
    b.emit("Transpose", [b.name(n.args[0])], n, perm=list(n.args[1]))
