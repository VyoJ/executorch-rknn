import dataclasses
import logging
import operator
from collections import Counter

from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from .compile_specs import RknnOptions
from .onnx_converter import unsupported_reason
from .onnx_converter.builder import target_name
from .preprocess import RknnBackend

log = logging.getLogger(__name__)

_QDQ = {
    "quantized_decomposed.quantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_channel.default",
}
# Nodes that do no work on the NPU worth a context of their own.
_TRIVIAL = _QDQ | {"aten.view_copy.default", "aten.permute_copy.default"}


class _Supported(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        if node.op != "call_function":
            return False
        if node.target is operator.getitem:
            return self.is_node_supported(submodules, node.args[0])
        return unsupported_reason(node) is None


def _is_float(node):
    val = node.meta.get("val")
    return val is not None and getattr(val, "dtype", None) is not None and val.dtype.is_floating_point


def _crosses_non_float(node, members):
    """True if node, in the partition, exchanges a non-float tensor with a node outside it."""
    ins = [a for a in node.all_input_nodes if a.op != "placeholder" and a not in members]
    outs = [u for u in node.users if u not in members]
    return (any(not _is_float(a) for a in ins)) or (outs and not _is_float(node))


def _float_boundaries(members):
    """Move quantize/dequantize nodes out of the partition until only float
    tensors cross its boundary: the delegate's inputs and outputs are float."""
    members = set(members)
    changed = True
    while changed:
        changed = False
        for n in list(members):
            if n.target is operator.getitem:
                continue
            if target_name(n) in _QDQ and _crosses_non_float(n, members):
                members.discard(n)
                changed = True
    return members


def _compute_ops(members):
    return sum(1 for n in members if n.target is not operator.getitem and target_name(n) not in _TRIVIAL)


class RknnPartitioner(Partitioner):
    """Delegates the largest connected groups of operators RKNN can run; the
    rest stays on the CPU. Each group becomes its own .rknn and RKNN context.

    min_partition_ops: groups with fewer operators than this (not counting
    quantize, dequantize, view and permute) stay on the CPU.
    """

    def __init__(self, options=None, min_partition_ops=1):
        super().__init__()
        self.options = options or RknnOptions()
        self.min_partition_ops = min_partition_ops

    def partition(self, exported_program):
        gm = exported_program.graph_module
        proposed = CapabilityBasedPartitioner(gm, _Supported(), allows_single_node_partition=True).propose_partitions()
        groups = []
        for p in proposed:
            members = _float_boundaries(p.nodes)
            if _compute_ops(members) >= max(self.min_partition_ops, 1):
                groups.append(members)

        delegated = set().union(*groups) if groups else set()
        cpu = Counter(target_name(n) for n in gm.graph.nodes
                      if n.op == "call_function" and n not in delegated and n.target is not operator.getitem)
        if cpu:
            log.info("RknnPartitioner: %d partition(s); on the CPU: %s", len(groups),
                     ", ".join(f"{c} x {name}" for name, c in sorted(cpu.items())))

        tags = {}
        for i, members in enumerate(groups):
            tag = f"rknn_{i}"
            options = self.options
            if options.dump_dir and len(groups) > 1:
                options = dataclasses.replace(options, dump_dir=f"{options.dump_dir}/{tag}")
            tags[tag] = DelegationSpec(RknnBackend.__name__, options.compile_specs())
            for n in members:
                n.meta["delegation_tag"] = tag
        tag_constant_data(exported_program)
        return PartitionResult(exported_program, tags)
