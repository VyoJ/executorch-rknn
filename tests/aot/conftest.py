import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from executorch.exir import to_edge

RKNN_PYTHON = os.environ.get("RKNN_PYTHON", "/opt/rknn/bin/python")
HELPER = Path(__file__).with_name("rknn_venv.py")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


def edge_program(module, *inputs):
    return to_edge(torch.export.export(module, inputs)).exported_program()


def _run_helper(*args):
    result = subprocess.run([RKNN_PYTHON, str(HELPER), *map(str, args)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{HELPER.name} {args[0]} failed:\n{result.stdout}\n{result.stderr}")


def run_onnx(model, *inputs):
    """Run an onnx.ModelProto with onnxruntime in the RKNN venv; returns a list of outputs."""
    feeds = {i.name: x for i, x in zip(model.graph.input, inputs)}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.onnx").write_bytes(model.SerializeToString())
        np.savez(tmp / "in.npz", **feeds)
        _run_helper("ort", tmp / "model.onnx", tmp / "in.npz", tmp / "out.npz")
        out = np.load(tmp / "out.npz")
        return [out[f"arr_{i}"] for i in range(len(out.files))]


def run_simulator(model, options_json, samples):
    """Build an ONNX model with RKNN and run each sample in its simulator."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.onnx").write_bytes(model.SerializeToString())
        (tmp / "options.json").write_text(options_json)
        np.savez(tmp / "in.npz", x=samples)
        _run_helper("sim", tmp / "model.onnx", tmp / "options.json", tmp / "in.npz", tmp / "out.npz")
        return np.load(tmp / "out.npz")["arr_0"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
