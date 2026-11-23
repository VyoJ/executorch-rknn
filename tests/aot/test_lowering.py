"""RknnPartitioner and RknnBackend through to_edge_transform_and_lower."""

import operator

import pytest
import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch_rknn import RknnOptions, RknnPartitioner, blob
from executorch_rknn.onnx_converter.builder import target_name
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


def lower(module, x, options=None, **partitioner_args):
    return to_edge_transform_and_lower(torch.export.export(module, (x,)),
                                       partitioner=[RknnPartitioner(options, **partitioner_args)])


def test_whole_graph_becomes_one_delegate(tmp_path):
    x = torch.randn(1, 3, 16, 16)
    edge = lower(quantize(small_net(), x), x, RknnOptions(dump_dir=str(tmp_path)))
    payload = blob.unpack(delegate_payloads(edge.to_executorch().buffer))
    assert payload.inputs == [blob.TensorInfo("float32", (1, 3, 16, 16))]
    assert payload.outputs == [blob.TensorInfo("float32", (1, 4))]
    assert payload.rknn == (tmp_path / "model.rknn").read_bytes()


def delegate_calls(edge):
    graph = edge.exported_program().graph_module.graph
    calls = [n for n in graph.nodes if n.op == "call_function" and "call_delegate" in str(n.target)]
    cpu = sorted(target_name(n).split(".")[1] for n in graph.nodes
                 if n.op == "call_function" and n not in calls and n.target is not operator.getitem)
    return len(calls), cpu


class Gated(nn.Module):
    """Two convs whose product (unsupported: aten.mul) feeds a third."""

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Conv2d(3, 8, 1), nn.Conv2d(3, 8, 1), nn.Conv2d(8, 4, 3)

    def forward(self, x):
        return self.c(self.a(x) * self.b(x))


def test_unsupported_op_in_the_middle_splits_the_graph():
    x = torch.randn(1, 3, 8, 8)
    edge = lower(quantize(Gated().eval(), x), x)
    assert delegate_calls(edge) == (2, ["mul"])


def test_unsupported_op_at_the_end_stays_on_cpu():
    x = torch.randn(1, 3, 16, 16)
    model = nn.Sequential(small_net(), nn.Softmax(-1)).eval()
    assert delegate_calls(lower(quantize(model, x), x)) == (1, ["_softmax"])


def test_delegates_exchange_only_float():
    x = torch.randn(1, 3, 8, 8)
    edge = lower(quantize(Gated().eval(), x), x)
    for n in edge.exported_program().graph_module.graph.nodes:
        if n.target is operator.getitem:
            assert n.meta["val"].dtype == torch.float32


def test_small_partitions_stay_on_cpu():
    class Net(nn.Module):
        def forward(self, x):
            return torch.relu(torch.sin(x))

    edge = lower(Net(), torch.randn(1, 3, 8, 8), min_partition_ops=2)
    assert delegate_calls(edge) == (0, ["relu", "sin"])


def test_nothing_supported_means_no_delegate():
    class Net(nn.Module):
        def forward(self, x):
            return torch.sin(x).mean(1)

    assert delegate_calls(lower(Net(), torch.randn(1, 3, 8, 8))) == (0, ["mean", "sin"])


def test_float_graph_needs_quantisation():
    with pytest.raises(ValueError, match="int8 only"):
        lower(small_net(), torch.randn(1, 3, 16, 16))
