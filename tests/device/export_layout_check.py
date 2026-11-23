"""A small quantised conv net with a 4-D output, odd sizes and a channel count
that is not a multiple of the NPU's channel block, to check RknnBackend's input
and output layout conversion on the board. Runs in the ExecuTorch venv.

Writes out/rknn_layout/: model.pte, inputs/*.bin (NCHW float32), reference.npz.
usage: python tests/device/export_layout_check.py, then
  tests/device/run_rknn_pte.sh out/rknn_layout/model.pte out/rknn_layout
  python tests/device/compare_rknn_outputs.py out/rknn_layout --min-cosine 0.99
"""

from pathlib import Path

import numpy as np
import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch_rknn import RknnOptions, RknnPartitioner
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

OUT = Path("out/rknn_layout")
SHAPE = (1, 3, 31, 29)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 10, 3, stride=2)).eval()
    calib = [torch.randn(SHAPE) for _ in range(16)]
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), quantizer)
    for x in calib:
        prepared(x)
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(torch.export.export(quantized, (calib[0],)),
                                       partitioner=[RknnPartitioner(RknnOptions(dump_dir=str(OUT)))])
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "model.pte").write_bytes(edge.to_executorch().buffer)

    pt2e, flt = [], []
    with torch.no_grad():
        for i in range(4):
            x = torch.randn(SHAPE)
            x.numpy().tofile(OUT / "inputs" / f"{i:03d}.bin")
            pt2e.append(quantized(x)[0].numpy().ravel())
            flt.append(model(x)[0].numpy().ravel())
    np.savez(OUT / "reference.npz", pt2e=np.stack(pt2e), float=np.stack(flt))
    print(f"wrote {OUT}/model.pte; output shape {tuple(quantized(calib[0]).shape)}")


if __name__ == "__main__":
    main()
