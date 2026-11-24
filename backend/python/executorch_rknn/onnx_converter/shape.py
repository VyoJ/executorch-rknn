import numpy as np

from .builder import converts


def _check_batch_1(n):
    # A 4-D reshape or transpose with a batch above 1 (MobileViT's patch
    # unfold, [1, 144, 8, 8] -> [576, 2, 4, 2]) splits into thousands of NPU
    # tasks on RV1106 and can time out in rknn_run.
    for t in (n.args[0].meta["val"], n.meta["val"]):
        if t.dim() == 4 and t.shape[0] != 1:
            return f"4-D {n.target.__name__.split('.')[0]} with batch {t.shape[0]}"
    return None


@converts("aten.view_copy.default", check=_check_batch_1)
def view(b, n):
    b.emit("Reshape", [b.name(n.args[0]), b.const(list(n.args[1]), np.int64)], n)


@converts("aten.permute_copy.default", check=_check_batch_1)
def permute(b, n):
    b.emit("Transpose", [b.name(n.args[0])], n, perm=list(n.args[1]))


@converts("aten.cat.default")
def cat(b, n):
    tensors = n.args[0]
    dim = n.args[1] if len(n.args) > 1 else 0
    b.emit("Concat", [b.name(t) for t in tensors], n, axis=dim % len(n.meta["val"].shape))


def _check_clone(n):
    dim_order = n.kwargs.get("dim_order")
    if dim_order is not None and list(dim_order) != list(range(len(dim_order))):
        return f"clone to dim order {list(dim_order)}"
    return None


@converts("dim_order_ops._clone_dim_order.default", check=_check_clone)
def clone(b, n):
    # A contiguous copy is a no-op in ONNX: the node is an alias of its input.
    b.names[n.name] = b.name(n.args[0])
