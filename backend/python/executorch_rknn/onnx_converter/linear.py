from .builder import converts


@converts("aten.addmm.default")
def addmm(b, n):
    bias, mat1, mat2 = n.args
    b.emit("Gemm", [b.name(mat1), b.name(mat2), b.name(bias)], n,
           alpha=float(n.kwargs.get("alpha", 1)), beta=float(n.kwargs.get("beta", 1)))
