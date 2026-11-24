"""Builds an ONNX graph from an Edge-dialect ExportedProgram, one node at a time.

rknn-toolkit2 1.6.0 reads opset 19 at most and IR version 9 at most.
"""

import operator

import numpy as np
import onnx
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from onnx import helper, numpy_helper

OPSET = 19
IR_VERSION = 9

CONVERTERS = {}
CHECKS = {}


def converts(*targets, check=None):
    """Register a converter. check(node) -> str | None names a reason the node cannot be converted."""
    def register(fn):
        for t in targets:
            CONVERTERS[t] = fn
            if check is not None:
                CHECKS[t] = check
        return fn
    return register


def target_name(node):
    t = node.target
    return str(t._op if isinstance(t, EdgeOpOverload) else t)


def unsupported_reason(node):
    """None if the node converts, else why not."""
    if node.target is operator.getitem:
        return None
    name = target_name(node)
    if name not in CONVERTERS:
        return name
    check = CHECKS.get(name)
    return check(node) if check else None


class Builder:
    def __init__(self, ep):
        self.ep = ep
        self.nodes, self.inits = [], []
        self.names, self.values = {}, {}
        self._count = 0

    def name(self, arg):
        return self.names[arg.name]

    def value(self, arg):
        """The constant behind a placeholder, as a NumPy array."""
        return self.values[arg.name]

    def const(self, value, dtype=np.float32):
        self._count += 1
        name = f"const_{self._count}"
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def emit(self, op, inputs, node, **attrs):
        """Add the ONNX node that produces fx node `node`."""
        self.nodes.append(helper.make_node(op, inputs, [node.name], name=node.name, **attrs))
        self.names[node.name] = node.name

    def emit_tmp(self, op, inputs, name, **attrs):
        """Add an intermediate ONNX node and return its output name."""
        self.nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name


def _value_info(name, tensor):
    dtype = helper.np_dtype_to_tensor_dtype(np.dtype(str(tensor.dtype).removeprefix("torch.")))
    return helper.make_tensor_value_info(name, dtype, list(tensor.shape))


def _constants(ep):
    sig = ep.graph_signature
    sources = {**sig.inputs_to_parameters, **sig.inputs_to_buffers, **sig.inputs_to_lifted_tensor_constants}
    out = {}
    for name, key in sources.items():
        t = ep.state_dict[key] if key in ep.state_dict else ep.constants[key]
        out[name] = t.detach().numpy()
    return out


def _merge_quantizes(b):
    """Merge QuantizeLinear nodes with the same input and parameters.
    DuplicateQuantizePass makes such copies so that each delegate gets its
    own; when copies land in the same delegate, RKNN runs the second as a
    separate conversion, which RV1106 cannot do."""
    consts = {t.name: numpy_helper.to_array(t) for t in b.inits}
    kept, rename = {}, {}
    nodes = []
    for node in b.nodes:
        node.input[:] = [rename.get(i, i) for i in node.input]
        if node.op_type == "QuantizeLinear":
            key = (node.input[0], *(consts[i].tobytes() + str(consts[i].dtype).encode() for i in node.input[1:]))
            if key in kept:
                rename[node.output[0]] = kept[key]
                continue
            kept[key] = node.output[0]
        nodes.append(node)
    b.nodes[:] = nodes
    for k, v in b.names.items():
        b.names[k] = rename.get(v, v)


def _leaf_output(b, arg):
    """The ONNX name to export fx value `arg` as. RKNN keeps only outputs that
    nothing inside the model reads: its int8 graph has no tensor for a value
    that also feeds a quantize (the two are fused), and it drops such outputs
    without an error. Export a copy of that quantize/dequantize instead; the
    CPU side quantizes the value again with the same parameters, so nothing
    changes."""
    name = b.name(arg)
    readers = [node for node in b.nodes if name in node.input]
    if not readers:
        return name
    q = next((node for node in readers if node.op_type == "QuantizeLinear"), None)
    if q is None and not any(node.op_type == "QuantizeLinear" for node in b.nodes):
        return name  # a float graph: RKNN places the quantization itself
    if q is None:
        raise NotImplementedError(f"{arg.name} is a delegate output that the delegate also reads, "
                                  "and no quantize follows it")
    # An existing dequantize of that quantize with nothing reading it is the
    # same value; RKNN would merge a copy with it anyway.
    consts = {t.name: numpy_helper.to_array(t) for t in b.inits}

    def same_params(x, y):
        return all(np.array_equal(consts.get(i), consts.get(j)) for i, j in zip(x.input[1:], y.input[1:]))

    for dq in b.nodes:
        if (dq.op_type == "DequantizeLinear" and dq.input[0] == q.output[0] and same_params(dq, q)
                and not any(dq.output[0] in node.input for node in b.nodes)):
            return dq.output[0]
    qcopy = b.emit_tmp("QuantizeLinear", list(q.input), f"{arg.name}__out_q")
    return b.emit_tmp("DequantizeLinear", [qcopy, *q.input[1:]], f"{arg.name}__out")


def convert(ep):
    """Edge ExportedProgram -> onnx.ModelProto. Raises NotImplementedError on an unsupported node."""
    return convert_program(ep)[0]


def convert_program(ep):
    """Edge ExportedProgram -> (onnx.ModelProto, output_sources): the ONNX
    output each program output reads. Program outputs with the same value
    share one ONNX output."""
    b = Builder(ep)
    constants = _constants(ep)
    inputs, outputs, sources = [], [], []
    for n in ep.graph.nodes:
        if n.op == "placeholder":
            b.names[n.name] = n.name
            if n.name in constants:
                b.values[n.name] = constants[n.name]
                b.inits.append(numpy_helper.from_array(constants[n.name], n.name))
            else:
                inputs.append(_value_info(n.name, n.meta["val"]))
        elif n.op == "call_function" and n.target is operator.getitem:
            src, index = n.args
            if index != 0:
                raise NotImplementedError(f"{src.name}[{index}] is used")
            b.names[n.name] = b.name(src)
        elif n.op == "call_function":
            reason = unsupported_reason(n)
            if reason is not None:
                raise NotImplementedError(f"{n.name}: {reason}")
            CONVERTERS[target_name(n)](b, n)
        elif n.op == "output":
            _merge_quantizes(b)
            names = [_leaf_output(b, a) for a in n.args[0]]
            unique = list(dict.fromkeys(names))
            sources = [unique.index(name) for name in names]
            outputs = [_value_info(name, n.args[0][names.index(name)].meta["val"]) for name in unique]

    # Drop initializers nothing reads, such as zero points a converter re-emitted.
    used = {i for node in b.nodes for i in node.input}
    inits = [t for t in b.inits if t.name in used]
    graph = helper.make_graph(b.nodes, "edge_program", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, sources
