"""Graph passes for lowering to the board. Run them as transform_passes of
to_edge_transform_and_lower, which applies them before partitioning (a
partitioner may not change the graph). DuplicateQuantizePass is required by
RknnPartitioner; the others make what stays on the CPU cheaper."""

from torch.fx.passes.infra.pass_base import PassBase, PassResult

from .onnx_converter.builder import QUANTIZE, target_name

EXPAND = "aten.expand_copy.default"
VIEW = "aten.view_copy.default"
BMM = "aten.bmm.default"
DIV = "aten.div.Tensor"


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


class RemoveNoopExpandPass(PassBase):
    """Removes expand_copy nodes whose output has their input's shape.

    torch.matmul on 4-D tensors decomposes into expand, view and bmm; when
    nothing broadcasts, each expand is a full copy of its input."""

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for n in [n for n in graph.nodes if n.op == "call_function" and target_name(n) == EXPAND]:
            if tuple(n.meta["val"].shape) == tuple(n.args[0].meta["val"].shape):
                n.replace_all_uses_with(n.args[0])
                graph.erase_node(n)
                modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


class ScaleBeforeBmmPass(PassBase):
    """Rewrites div(bmm(a, b), c), with a view allowed after the bmm and c a
    single-element constant, as bmm(div(a, c), b).

    Attention divides its N x N scores by sqrt(d); dividing the N x d queries
    instead does N / d times less work. The result differs only by rounding."""

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for div in [n for n in graph.nodes if n.op == "call_function" and target_name(n) == DIV]:
            x, c = div.args[:2]
            if div.kwargs or not hasattr(c, "meta") or c.op != "placeholder" or c.meta["val"].numel() != 1:
                continue
            view = x if target_name(x) == VIEW else None
            bmm = view.args[0] if view is not None else x
            if target_name(bmm) != BMM or len(bmm.users) != 1 or (view is not None and len(view.users) != 1):
                continue
            a = bmm.args[0]
            if bmm.args[1] is a:
                continue
            with graph.inserting_before(bmm):
                scaled = graph.call_function(div.target, (a, c))
            scaled.meta = {**a.meta, "val": a.meta["val"]}
            bmm.replace_input_with(a, scaled)
            div.replace_all_uses_with(x)
            graph.erase_node(div)
            modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)


def transform_passes():
    return [DuplicateQuantizePass(), RemoveNoopExpandPass(), ScaleBeforeBmmPass()]
