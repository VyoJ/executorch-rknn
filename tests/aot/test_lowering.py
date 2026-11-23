"""RknnPartitioner and RknnBackend through to_edge_transform_and_lower."""

import pytest
import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch_rknn import RknnOptions, RknnPartitioner, blob
from export_mobilenet_v2 import delegate_payloads
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def small_net():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(8, 4)).eval()


def quantize(module, x):
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(module, (x,)).module(), quantizer)
    for _ in range(4):
        prepared(torch.randn_like(x))
    return convert_pt2e(prepared)


def lower(module, x, options=None):
    return to_edge_transform_and_lower(torch.export.export(module, (x,)),
                                       partitioner=[RknnPartitioner(options)])


def test_whole_graph_becomes_one_delegate(tmp_path):
    x = torch.randn(1, 3, 16, 16)
    edge = lower(quantize(small_net(), x), x, RknnOptions(dump_dir=str(tmp_path)))
    payload = blob.unpack(delegate_payloads(edge.to_executorch().buffer))
    assert payload.inputs == [blob.TensorInfo("float32", (1, 3, 16, 16))]
    assert payload.outputs == [blob.TensorInfo("float32", (1, 4))]
    assert payload.rknn == (tmp_path / "model.rknn").read_bytes()


def test_unsupported_ops_are_listed():
    class Net(nn.Module):
        def forward(self, x):
            return torch.sin(x).mean(1)

    with pytest.raises(NotImplementedError) as e:
        lower(Net(), torch.randn(1, 3, 8, 8))
    assert "aten.sin.default" in str(e.value)
    assert "mean over dims [1]" in str(e.value)


def test_float_graph_needs_quantisation():
    with pytest.raises(ValueError, match="int8 only"):
        lower(small_net(), torch.randn(1, 3, 16, 16))
