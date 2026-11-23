import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult

from . import blob, onnx_converter
from .compile_specs import RknnOptions

RKNN_PYTHON = os.environ.get("RKNN_PYTHON", "/opt/rknn/bin/python")
_CONVERT_SCRIPT = Path(__file__).with_name("rknn_convert.py")
_QDQ = {"QuantizeLinear", "DequantizeLinear"}


def _tensor_info(meta_val):
    return blob.TensorInfo(str(meta_val.dtype).removeprefix("torch."), tuple(meta_val.shape))


def _io(edge_program):
    user_inputs = set(edge_program.graph_signature.user_inputs)
    inputs = [_tensor_info(n.meta["val"]) for n in edge_program.graph.nodes
              if n.op == "placeholder" and n.name in user_inputs]
    output = next(n for n in edge_program.graph.nodes if n.op == "output")
    return inputs, [_tensor_info(a.meta["val"]) for a in output.args[0]]


def to_rknn(model, options):
    """ONNX model -> .rknn bytes, through the toolkit in its own venv."""
    quantized = any(n.op_type in _QDQ for n in model.graph.node)
    if not quantized and options.calibration_dataset is None and options.target_platform in ("rv1103", "rv1106"):
        raise ValueError(f"{options.target_platform} runs int8 only: quantise with PT2E first, "
                         "or set calibration_dataset to let RKNN calibrate")
    with tempfile.TemporaryDirectory(prefix="rknn-") as tmp:
        tmp = Path(tmp)
        (tmp / "model.onnx").write_bytes(model.SerializeToString())
        settings = {**asdict(options), "quantize": not quantized}
        (tmp / "options.json").write_text(json.dumps(settings))
        result = subprocess.run(
            [RKNN_PYTHON, str(_CONVERT_SCRIPT), str(tmp / "model.onnx"), str(tmp / "model.rknn"),
             str(tmp / "options.json")],
            capture_output=True, text=True, cwd=tmp)  # the toolkit drops files in its working directory
        if result.returncode != 0:
            raise RuntimeError(f"RKNN conversion failed:\n{result.stdout}\n{result.stderr}")
        if options.dump_dir:
            dump = Path(options.dump_dir)
            dump.mkdir(parents=True, exist_ok=True)
            for name in ("model.onnx", "model.rknn"):
                shutil.copy(tmp / name, dump / name)
            (dump / "rknn_convert.log").write_text(result.stdout + result.stderr)
        return (tmp / "model.rknn").read_bytes()


class RknnBackend(BackendDetails):
    @staticmethod
    def preprocess(edge_program, compile_specs):
        options = RknnOptions.from_compile_specs(compile_specs)
        model = onnx_converter.convert(edge_program)
        inputs, outputs = _io(edge_program)
        rknn = to_rknn(model, options)
        return PreprocessResult(processed_bytes=blob.pack(blob.Blob(inputs, outputs, rknn)))
