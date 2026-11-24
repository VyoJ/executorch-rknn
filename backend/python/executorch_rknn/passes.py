"""Graph passes RknnPartitioner relies on. Run them as transform_passes of
to_edge_transform_and_lower, which applies them before partitioning (a
partitioner may not change the graph)."""

from torch.fx.passes.infra.pass_base import PassBase, PassResult

from .onnx_converter.builder import target_name

QUANTIZE = "quantized_decomposed.quantize_per_tensor.default"


class DuplicateQuantizePass(PassBase):
    """Gives each dequantize its own copy of the quantize feeding it.

    One quantize can feed linears that end up in different delegates (q, k
    and v after a LayerNorm, say). Each delegate must receive its input
    through its own quantize/dequantize pair, or RKNN sees a float input to
    an int8 operator."""

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for q in [n for n in graph.nodes if n.op == "call_function" and target_name(n) == QUANTIZE]:
            for dq in list(q.users)[1:]:
                with graph.inserting_after(q):
                    copy = graph.node_copy(q)
                dq.replace_input_with(q, copy)
                modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


def transform_passes():
    return [DuplicateQuantizePass()]
