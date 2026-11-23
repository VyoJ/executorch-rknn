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


def convert(ep):
    """Edge ExportedProgram -> onnx.ModelProto. Raises NotImplementedError on an unsupported node."""
    b = Builder(ep)
    constants = _constants(ep)
    inputs, outputs = [], []
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
            outputs = [_value_info(b.name(a), a.meta["val"]) for a in n.args[0]]

    # Drop initializers nothing reads, such as zero points a converter re-emitted.
    used = {i for node in b.nodes for i in node.input}
    inits = [t for t in b.inits if t.name in used]
    graph = helper.make_graph(b.nodes, "edge_program", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model
