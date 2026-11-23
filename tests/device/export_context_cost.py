"""The same 8 conv blocks split into 1, 2, 4 and 8 RKNN delegates by a cheap
CPU operator (neg) between groups, to measure what each extra RKNN context
costs. Runs in the ExecuTorch venv.

Writes out/rknn_contexts/n<k>/: model.pte, rknn_<i>/model.rknn, inputs/000.bin,
reference.npz.
usage: python tests/device/export_context_cost.py, then tests/device/context_cost.sh
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

OUT = Path("out/rknn_contexts")
SHAPE = (1, 3, 64, 64)
BLOCKS = 8


class Chain(nn.Module):
    def __init__(self, splits):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(3 if i == 0 else 32, 32, 3, padding=1), nn.ReLU()) for i in range(BLOCKS))
        self.every = BLOCKS // splits

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % self.every == 0 and i + 1 < BLOCKS:
                x = -x
        return x


def main():
    for splits in (1, 2, 4, 8):
        torch.manual_seed(0)
        model = Chain(splits).eval()
        out = OUT / f"n{splits}"
        calib = [torch.randn(SHAPE) for _ in range(8)]
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), quantizer)
        for x in calib:
            prepared(x)
        quantized = convert_pt2e(prepared)
        # Always dump per partition, so n1 has the same layout as the others.
        options = RknnOptions(dump_dir=str(out / "rknn_0") if splits == 1 else str(out))
        edge = to_edge_transform_and_lower(torch.export.export(quantized, (calib[0],)),
                                           partitioner=[RknnPartitioner(options)])
        (out / "inputs").mkdir(parents=True, exist_ok=True)
        (out / "model.pte").write_bytes(edge.to_executorch().buffer)
        x = torch.randn(SHAPE)
        x.numpy().tofile(out / "inputs" / "000.bin")
        with torch.no_grad():
            np.savez(out / "reference.npz", pt2e=quantized(x).numpy()[None], float=model(x).numpy()[None])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
