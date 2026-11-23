"""Helpers the tests run in the RKNN venv (/opt/rknn), which has onnxruntime
and rknn-toolkit2 but not torch.

  rknn_venv.py ort model.onnx inputs.npz outputs.npz
  rknn_venv.py sim model.onnx options.json inputs.npz outputs.npz

inputs.npz holds arrays named by graph input; for sim, a leading batch of
samples, each run separately and given to RKNN in NCHW.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend/python/executorch_rknn"))


def ort(model, inputs, outputs):
    import onnxruntime

    sess = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
    feeds = dict(np.load(inputs))
    results = sess.run(None, feeds)
    np.savez(outputs, *results)


def sim(model, options, inputs, outputs):
    from rknn_convert import build

    options = json.loads(Path(options).read_text())
    rknn = build(model, options, quantize=options["quantize"])
    if rknn.init_runtime() != 0:
        raise RuntimeError("RKNN simulator init failed")
    samples = np.load(inputs)["x"]
    results = [rknn.inference(inputs=[s], data_format="nchw")[0] for s in samples]
    rknn.release()
    np.savez(outputs, np.stack(results))


if __name__ == "__main__":
    {"ort": ort, "sim": sim}[sys.argv[1]](*sys.argv[2:])
