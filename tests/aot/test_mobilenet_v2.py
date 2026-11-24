"""MobileNetV2 end to end: PT2E, one delegate, and RKNN's simulator close to PT2E.

The simulator is only a quick check; the board is the reference
(tests/device/run_rknn_model.sh).
"""

import json

import numpy as np
import onnx
import torch
from conftest import run_simulator
from executorch.exir import to_edge_transform_and_lower
from executorch_rknn import RknnOptions, RknnPartitioner, transform_passes
from export_mobilenet_v2 import delegate_payloads, images, quantize
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def test_mobilenet_v2(tmp_path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    calib = [x for _, x in images("train", 2)]
    quantized = quantize(model, calib)
    edge = to_edge_transform_and_lower(torch.export.export(quantized, (calib[0],)),
                                       transform_passes=transform_passes(),
                                       partitioner=[RknnPartitioner(RknnOptions(dump_dir=str(tmp_path)))])
    delegate_payloads(edge.to_executorch().buffer)

    onnx_model = onnx.load(tmp_path / "model.onnx")
    samples = torch.cat([x for _, x in images("val", 1)])
    with torch.no_grad():
        expected = torch.cat([quantized(x[None]) for x in samples]).numpy()
    options = json.dumps({"target_platform": "rv1106", "optimization_level": 3, "quantize": False})
    got = run_simulator(onnx_model, options, samples.numpy()[:, None]).reshape(expected.shape)
    cos = [a @ b / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(got, expected)]
    assert np.mean(cos) > 0.95, cos
