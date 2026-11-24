import dataclasses
import logging
import operator
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from .compile_specs import RknnOptions
from .onnx_converter import unsupported_reason
from .onnx_converter.builder import QDQ, QUANTIZE, target_name
from .preprocess import RknnBackend

log = logging.getLogger(__name__)

# Nodes that do no work on the NPU worth a context of their own.
_TRIVIAL = QDQ | {"aten.view_copy.default", "aten.permute_copy.default", "dim_order_ops._clone_dim_order.default"}


_WEIGHTED = {"aten.convolution.default": 1, "aten.addmm.default": 2}  # weight argument index


def _float_weight(node):
    """True for a convolution or linear whose weight does not come through a
    dequantize: in a PT2E graph, an operator left in float."""
    if target_name(node) not in _WEIGHTED:
        return False
    w = node.args[_WEIGHTED[target_name(node)]]
    while target_name(w) in ("aten.permute_copy.default", "aten.view_copy.default"):
        w = w.args[0]
    return target_name(w) not in QDQ


class _Supported(OperatorSupportBase):
    def __init__(self, float_weights):
        super().__init__()
        self.float_weights = float_weights

    def is_node_supported(self, submodules, node):
        if node.op != "call_function":
            return False
        if node.target is operator.getitem:
            return self.is_node_supported(submodules, node.args[0])
        if not self.float_weights and _float_weight(node):
            return False
        return unsupported_reason(node) is None



def _float_boundaries(groups):
    """Adjust the groups so only float tensors cross their boundaries: the
    delegate's inputs and outputs are float. Every int8 tensor runs from a
    quantize to a dequantize. A dequantize outside the group of its quantize
    moves into it (its inputs are all there, so no cycle can form); a
    dequantize whose quantize is on the CPU leaves its group (its inputs are
    all outside, so the same holds)."""
    owner = {n: i for i, g in enumerate(groups) for n in g}
    for q in owner:
        if target_name(q) == QUANTIZE and len(q.users) > 1:
            raise ValueError("a quantize node feeds several dequantize nodes; lower with "
                             "transform_passes=executorch_rknn.transform_passes()")
    for q in [n for n in owner if target_name(n) == QUANTIZE]:
        for dq in q.users:
            if owner.get(dq) != owner[q]:
                if dq in owner:
                    groups[owner[dq]].discard(dq)
                groups[owner[q]].add(dq)
                owner[dq] = owner[q]
    for n in list(owner):
        if target_name(n) in QDQ and target_name(n) != QUANTIZE:
            src = n.args[0]
            if src.op != "placeholder" and owner.get(src) != owner[n]:
                groups[owner[n]].discard(n)
                del owner[n]
    return groups


def _quantize_at_entry(g, constants):
    """Move to the CPU any node that reads an activation from outside its
    group without quantising it. RKNN keeps such an input in float16, which
    RV1103/RV1106 refuse at run time. The nodes trimmed this way are views,
    permutes and copies left between a CPU operator and the next quantize,
    and quantize/dequantize pairs with nothing between them."""
    changed = True
    while changed:
        changed = False
        for n in list(g):
            if n not in g or n.target is operator.getitem:
                continue
            from_outside = any(a not in g and a.name not in constants for a in n.all_input_nodes)
            if target_name(n) == QUANTIZE:
                # A quantize/dequantize pair that only passes a value through
                # would run as a standalone conversion, which RV1106 cannot do.
                if from_outside and not any(u in g for dq in n.users for u in dq.users):
                    g.difference_update([n, *n.users])
                    changed = True
            elif from_outside:
                g.discard(n)
                changed = True
            elif target_name(n) not in _TRIVIAL and n.users and all(u not in g for u in n.users):
                # The same at the exit: an output not produced by a dequantize
                # would be float16. Only results nothing in the group reads
                # move, so no cycle can form.
                g.discard(n)
                changed = True
    return g


def _is_quantized(group):
    return any(target_name(n) in QDQ for n in group)


def _external_inputs(group, constants):
    """Activations the group reads from outside it, in graph order."""
    seen = {}
    for n in group:
        for a in n.all_input_nodes:
            if a not in group and a.name not in constants:
                seen[a] = None
    order = {n: i for i, n in enumerate(next(iter(group)).graph.nodes)}
    return sorted(seen, key=order.get)


def _capture(program, names, samples):
    """Lowers `program` (an ExportedProgram with real weights; partitioners
    only see fake ones) to Edge, runs each sample (a tuple of user inputs)
    through it and returns {node name: [value per sample]}. Edge lowering is
    deterministic, so node names match the program being partitioned."""
    from executorch.exir import to_edge

    edge = to_edge(program).exported_program()
    sig = edge.graph_signature
    state = {**edge.state_dict, **edge.constants}
    sources = {**sig.inputs_to_parameters, **sig.inputs_to_buffers, **sig.inputs_to_lifted_tensor_constants}
    placeholders = [n for n in edge.graph.nodes if n.op == "placeholder"]
    missing = set(names) - {n.name for n in edge.graph.nodes}
    if missing:
        raise ValueError(f"calibration_program does not match the program being lowered: no {sorted(missing)[:5]}")
    wanted = set(names)
    values = {name: [] for name in names}

    class Recorder(torch.fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if n.name in wanted:
                values[n.name].append(out.detach().numpy().copy())
            return out

    for sample in samples:
        user = iter(sample)
        args = [state[sources[p.name]].detach() if p.name in sources else next(user) for p in placeholders]
        with torch.no_grad():
            Recorder(edge.graph_module).run(*args)
    return values


def _compute_ops(members):
    return sum(1 for n in members if n.target is not operator.getitem and target_name(n) not in _TRIVIAL)


class RknnPartitioner(Partitioner):
    """Delegates the largest connected groups of operators RKNN can run; the
    rest stays on the CPU. Each group becomes its own .rknn and RKNN context.

    min_partition_ops: groups with fewer operators than this (not counting
    quantize, dequantize, view, permute and clone) stay on the CPU.

    Groups without PT2E quantisation need RKNN's own calibration. Given
    calibration_inputs (tuples of model inputs) and calibration_program (the
    ExportedProgram being lowered, which still has its weights), the
    partitioner runs the inputs through it and calibrates each such group on
    the activations that actually reach it. Otherwise these groups stay on
    the CPU.
    """

    def __init__(self, options=None, min_partition_ops=1, calibration_inputs=None, calibration_program=None):
        super().__init__()
        if (calibration_inputs is None) != (calibration_program is None):
            raise ValueError("calibration_inputs and calibration_program go together")
        self.options = options or RknnOptions()
        self.min_partition_ops = min_partition_ops
        self.calibration_inputs = calibration_inputs
        self.calibration_program = calibration_program

    def _buildable(self, group):
        """RV1103/RV1106 run int8 only: a group with no quantize or dequantize
        needs RKNN's own calibration."""
        return (self.options.calibration_dataset is not None or self.calibration_inputs is not None
                or _is_quantized(group))

    def _calibration_dirs(self, exported_program, groups, tags, constants):
        """For each unquantised group, writes the activations reaching it for
        every calibration input as <dir>/<tag>/<node name>_<sample>.npy, NCHW
        float32 as RKNN expects. The node names are the delegate's ONNX input
        names. Returns {tag: dir}."""
        root = Path(tempfile.mkdtemp(prefix="rknn-calib-"))
        inputs = {tag: _external_inputs(g, constants) for tag, g in zip(tags, groups) if not _is_quantized(g)}
        values = _capture(self.calibration_program, {n.name for ns in inputs.values() for n in ns},
                          self.calibration_inputs)
        dirs = {}
        for tag, nodes in inputs.items():
            d = root / tag
            d.mkdir()
            for n in nodes:
                for i, v in enumerate(values[n.name]):
                    np.save(d / f"{n.name}_{i:03d}.npy", v.astype(np.float32))
            dirs[tag] = str(d)
        return dirs

    def partition(self, exported_program):
        gm = exported_program.graph_module
        # In a PT2E graph, operators left in float run on the CPU, unless RKNN
        # calibrates: then it quantises them itself.
        quantized_graph = any(n.op == "call_function" and target_name(n) in QDQ for n in gm.graph.nodes)
        float_weights = not quantized_graph or self.calibration_inputs is not None
        proposed = CapabilityBasedPartitioner(gm, _Supported(float_weights),
                                              allows_single_node_partition=True).propose_partitions()
        sig = exported_program.graph_signature
        constants = {*sig.inputs_to_parameters, *sig.inputs_to_buffers, *sig.inputs_to_lifted_tensor_constants}
        groups = [_quantize_at_entry(g, constants) if _is_quantized(g) else g
                  for g in _float_boundaries([set(p.nodes) for p in proposed])]
        groups = [g for g in groups if _compute_ops(g) >= max(self.min_partition_ops, 1)]
        if any(not self._buildable(g) for g in groups):
            log.warning("RknnPartitioner: leaving %d unquantised group(s) on the CPU; quantise with PT2E "
                        "or set calibration_dataset", sum(not self._buildable(g) for g in groups))
            groups = [g for g in groups if self._buildable(g)]

        delegated = set().union(*groups) if groups else set()
        cpu = Counter(target_name(n) for n in gm.graph.nodes
                      if n.op == "call_function" and n not in delegated and n.target is not operator.getitem)
        if cpu:
            log.info("RknnPartitioner: %d partition(s); on the CPU: %s", len(groups),
                     ", ".join(f"{c} x {name}" for name, c in sorted(cpu.items())))

        names = [f"rknn_{i}" for i in range(len(groups))]
        calibration = {}
        if self.calibration_inputs is not None:
            calibration = self._calibration_dirs(exported_program, groups, names, constants)
        tags = {}
        for tag, members in zip(names, groups):
            options = self.options
            if options.dump_dir and len(groups) > 1:
                options = dataclasses.replace(options, dump_dir=f"{options.dump_dir}/{tag}")
            if tag in calibration:
                options = dataclasses.replace(options, calibration_dataset=calibration[tag])
            tags[tag] = DelegationSpec(RknnBackend.__name__, options.compile_specs())
            for n in members:
                n.meta["delegation_tag"] = tag
        tag_constant_data(exported_program)
        return PartitionResult(exported_program, tags)
