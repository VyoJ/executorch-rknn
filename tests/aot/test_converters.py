"""Each converter against onnxruntime, through the same path preprocess uses."""

import numpy as np
import pytest
import torch
from conftest import edge_program, run_onnx
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch_rknn.onnx_converter import convert, unsupported_reason
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return x + self.conv(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


X = (1, 8, 16, 16)
CASES = {
    "conv": (nn.Conv2d(8, 16, 3), X),
    "conv_stride_pad_no_bias": (nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False), X),
    "conv_depthwise": (nn.Conv2d(8, 8, 3, padding=1, groups=8), X),
    "conv_dilated_1x5": (nn.Conv2d(8, 4, (1, 5), padding=(0, 4), dilation=(1, 2)), X),
    "batch_norm": (nn.BatchNorm2d(8), X),
    "relu": (nn.ReLU(), X),
    "relu6": (nn.ReLU6(), X),
    "max_pool": (nn.MaxPool2d(3, stride=2, padding=1), X),
    "adaptive_avg_pool": (nn.AdaptiveAvgPool2d(1), X),
    "mean_keepdim_false": (Lambda(lambda x: x.mean((2, 3))), X),
    "permute": (Lambda(lambda x: x.permute(0, 2, 3, 1)), X),
    "residual_add": (Residual(), X),
    "flatten_linear": (Classifier(), (1, 8, 4, 4)),
    "silu": (nn.SiLU(), X),
    "mul": (Lambda(lambda x: x * torch.sigmoid(x)), X),
    "cat": (Lambda(lambda x: torch.cat([x, x * x], 1)), X),
    "clone": (Lambda(lambda x: x.permute(0, 2, 3, 1).contiguous() + 1.0), X),
}


@pytest.mark.parametrize("name", CASES)
def test_float(name):
    module, shape = CASES[name]
    if isinstance(module, nn.BatchNorm2d):
        module.running_mean.uniform_(-1, 1)
        module.running_var.uniform_(0.5, 2)
    module.eval()
    x = torch.randn(shape)
    with torch.no_grad():
        expected = module(x).numpy()
    model = convert(edge_program(module, x))
    (got,) = run_onnx(model, x.numpy())
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_conv_has_kernel_shape():
    model = convert(edge_program(nn.Conv2d(8, 4, 3).eval(), torch.randn(X)))
    conv = next(n for n in model.graph.node if n.op_type == "Conv")
    assert [a.ints for a in conv.attribute if a.name == "kernel_shape"] == [[3, 3]]


def test_no_reduce_mean():
    model = convert(edge_program(nn.AdaptiveAvgPool2d(1).eval(), torch.randn(X)))
    assert [n.op_type for n in model.graph.node] == ["GlobalAveragePool"]


def test_mean_over_channels_is_unsupported():
    ep = edge_program(Lambda(lambda x: x.mean(1)), torch.randn(X))
    node = next(n for n in ep.graph.nodes if n.op == "call_function")
    assert "mean over dims" in unsupported_reason(node)


def test_quantized():
    module = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU6(), Residual(), nn.AdaptiveAvgPool2d(1),
                           nn.Flatten(), nn.Linear(8, 10)).eval()
    x = torch.randn(1, 3, 16, 16)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(module, (x,)).module(), quantizer)
    for _ in range(8):
        prepared(torch.randn(1, 3, 16, 16))
    quantized = convert_pt2e(prepared)
    with torch.no_grad():
        expected = quantized(x).numpy()
    model = convert(edge_program(quantized, x))
    ops = {n.op_type for n in model.graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops
    (got,) = run_onnx(model, x.numpy())
    # Float and integer arithmetic may round a value near a quantisation step
    # differently; allow a couple of output steps.
    step = float(next(a for a in quantized.graph.nodes if "dequantize_per_tensor" in str(a.target)).args[1])
    np.testing.assert_allclose(got, expected, atol=max(3 * step, 0.05))


def test_batched_4d_reshape_is_unsupported():
    ep = edge_program(Lambda(lambda x: x.reshape(16, 2, 4, 8)), torch.randn(1, 8, 16, 8))
    (view,) = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert "batch 16" in unsupported_reason(view)
