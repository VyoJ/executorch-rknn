"""ONNX -> .rknn with rknn-toolkit2. Runs in the toolkit's own Python
environment (/opt/rknn), which cannot import torch or ExecuTorch, so
RknnBackend calls it as a subprocess.

usage: rknn_convert.py model.onnx model.rknn options.json
"""

import json
import sys


def build(onnx_path, options, quantize):
    from rknn.api import RKNN

    rknn = RKNN()
    config = {"target_platform": options["target_platform"],
              "optimization_level": options["optimization_level"]}
    if options.get("mean_values") is not None:
        config["mean_values"] = [options["mean_values"]]
    if options.get("std_values") is not None:
        config["std_values"] = [options["std_values"]]
    if rknn.config(**config) != 0:
        raise RuntimeError(f"rknn.config({config}) failed")
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise RuntimeError("rknn.load_onnx failed")
    dataset = options.get("calibration_dataset") if quantize else None
    if rknn.build(do_quantization=quantize, dataset=dataset) != 0:
        raise RuntimeError("rknn.build failed")
    return rknn


def main(onnx_path, rknn_path, options_path):
    options = json.loads(open(options_path).read())
    rknn = build(onnx_path, options, quantize=options["quantize"])
    if rknn.export_rknn(rknn_path) != 0:
        raise RuntimeError("rknn.export_rknn failed")
    rknn.release()


if __name__ == "__main__":
    main(*sys.argv[1:])
