"""Graph passes in transform_passes(), on the edge program and through to an
executed .pte."""

import math

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
from executorch_rknn import transform_passes
from executorch_rknn.onnx_converter.builder import target_name
from torch import nn


class Attention(nn.Module):
    """Scaled dot-product scores the way MobileViT computes them."""

    def forward(self, q, k):
        return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)


def ops(module, *inputs):
    edge = to_edge_transform_and_lower(torch.export.export(module, inputs), transform_passes=transform_passes())
    graph = edge.exported_program().graph_module.graph
    names = [target_name(n) for n in graph.nodes if n.op == "call_function"]
    return names, edge, graph


def test_noop_expand_is_removed():
    q, k = torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)
    names, _, _ = ops(Attention(), q, k)
    assert "aten.expand_copy.default" not in names


def test_broadcasting_expand_stays():
    class Broadcast(nn.Module):
        def forward(self, q, k):
            return q @ k.transpose(-1, -2)

    names, _, _ = ops(Broadcast(), torch.randn(2, 4, 16, 8), torch.randn(1, 4, 16, 8))
    assert "aten.expand_copy.default" in names


def test_scale_moves_before_bmm():
    q, k = torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)
    _, edge, graph = ops(Attention(), q, k)
    bmm = next(n for n in graph.nodes if n.op == "call_function" and target_name(n) == "aten.bmm.default")
    div = next(n for n in graph.nodes if n.op == "call_function" and target_name(n) == "aten.div.Tensor")
    assert bmm.args[0] is div
    module = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
    torch.testing.assert_close(module.forward((q, k))[0], Attention()(q, k), rtol=1e-5, atol=1e-6)
