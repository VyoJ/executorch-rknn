from collections import Counter

from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data

from .compile_specs import RknnOptions
from .onnx_converter import unsupported_reason
from .preprocess import RknnBackend


class RknnPartitioner(Partitioner):
    """Version 1: delegates the whole graph, or raises listing what it cannot convert."""

    def __init__(self, options=None):
        super().__init__()
        self.options = options or RknnOptions()
        self.delegation_spec = DelegationSpec(RknnBackend.__name__, self.options.compile_specs())

    def partition(self, exported_program):
        nodes = [n for n in exported_program.graph_module.graph.nodes if n.op == "call_function"]
        reasons = Counter(r for r in map(unsupported_reason, nodes) if r is not None)
        if reasons:
            listing = "\n".join(f"  {count} x {reason}" for reason, count in sorted(reasons.items()))
            raise NotImplementedError(f"RknnPartitioner cannot delegate the whole graph:\n{listing}")
        for n in nodes:
            n.meta["delegation_tag"] = "rknn"
        tag_constant_data(exported_program)
        return PartitionResult(exported_program, {"rknn": self.delegation_spec})
