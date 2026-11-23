"""PT2E's quantize/dequantize nodes as ONNX QuantizeLinear/DequantizeLinear.

RKNN reads these as a QAT model and keeps their scales and zero points.
"""

import numpy as np
import torch

from .builder import converts

_DTYPES = {torch.int8: np.int8, torch.uint8: np.uint8}


def _check_per_tensor(n):
    x, scale, zp, qmin, qmax, dtype = n.args[:6]
    if dtype not in _DTYPES:
        return f"{dtype} quantisation (RV1106 runs int8 only)"
    if not isinstance(scale, float):
        return "scale given as a tensor"
    return None


def _check_per_channel(n):
    x, scales, zps, axis, qmin, qmax, dtype = n.args[:7]
    if dtype not in _DTYPES:
        return f"{dtype} quantisation (RV1106 runs int8 only)"
    if any(a.op != "placeholder" for a in (x, scales) + ((zps,) if zps is not None else ())):
        return "per-channel parameters computed at run time"
    return None


def _per_tensor(op, b, n):
    x, scale, zp, _, _, dtype = n.args[:6]
    b.emit(op, [b.name(x), b.const(scale), b.const(zp, _DTYPES[dtype])], n)


@converts("quantized_decomposed.quantize_per_tensor.default", check=_check_per_tensor)
def quantize_per_tensor(b, n):
    _per_tensor("QuantizeLinear", b, n)


@converts("quantized_decomposed.dequantize_per_tensor.default", check=_check_per_tensor)
def dequantize_per_tensor(b, n):
    _per_tensor("DequantizeLinear", b, n)


@converts("quantized_decomposed.dequantize_per_channel.default", check=_check_per_channel)
def dequantize_per_channel(b, n):
    x, scales, zps, axis, _, _, dtype = n.args[:7]
    scale_values = b.value(scales).astype(np.float32)
    zp_values = (b.value(zps) if zps is not None else np.zeros_like(scale_values)).astype(_DTYPES[dtype])
    b.emit("DequantizeLinear", [b.name(x), b.const(scale_values), b.const(zp_values, _DTYPES[dtype])],
           n, axis=axis)
