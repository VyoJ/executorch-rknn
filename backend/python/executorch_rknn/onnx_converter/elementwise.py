from .builder import converts


@converts("aten.relu.default")
def relu(b, n):
    b.emit("Relu", [b.name(n.args[0])], n)


@converts("aten.hardtanh.default")
def hardtanh(b, n):
    x, lo, hi = (list(n.args) + [-1.0, 1.0])[:3]
    b.emit("Clip", [b.name(x), b.const(lo), b.const(hi)], n)


def _check_add(n):
    if n.kwargs.get("alpha", 1) != 1:
        return "add with alpha != 1"
    if not all(hasattr(a, "meta") for a in n.args[:2]):
        return "add with a scalar operand"
    return None


@converts("aten.add.Tensor", check=_check_add)
def add(b, n):
    b.emit("Add", [b.name(a) for a in n.args[:2]], n)


@converts("aten.sigmoid.default")
def sigmoid(b, n):
    b.emit("Sigmoid", [b.name(n.args[0])], n)


def _check_mul(n):
    if not all(hasattr(a, "meta") for a in n.args[:2]):
        return "mul with a scalar operand"
    return None


@converts("aten.mul.Tensor", check=_check_mul)
def mul(b, n):
    b.emit("Mul", [b.name(a) for a in n.args[:2]], n)
