from .builder import converts


def _check_conv(n):
    return "transposed convolution" if n.args[6] else None


@converts("aten.convolution.default", check=_check_conv)
def convolution(b, n):
    x, w, bias, stride, padding, dilation, _, _, groups = n.args
    inputs = [b.name(x), b.name(w)] + ([b.name(bias)] if bias is not None else [])
    # kernel_shape is optional in ONNX, but without it RV1106 computes wrong
    # results (the simulator does not).
    b.emit("Conv", inputs, n, kernel_shape=list(w.meta["val"].shape[2:]), strides=list(stride),
           pads=list(padding) * 2, dilations=list(dilation), group=groups)


@converts("aten._native_batch_norm_legit_no_training.default")
def batch_norm(b, n):
    x, weight, bias, mean, var, _, eps = n.args
    b.emit("BatchNormalization", [b.name(a) for a in (x, weight, bias, mean, var)], n, epsilon=eps)
