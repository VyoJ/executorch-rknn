"""Small quantised models that RKNN can only run in parts, to check partial
delegation on the board. Runs in the ExecuTorch venv.

  middle_end: conv block, GroupNorm (CPU), conv block and classifier, softmax (CPU)
  gated:      conv block and two 1x1 convs, their elementwise maximum (CPU),
              conv; the first delegate has two outputs

Writes out/rknn_partial/<name>/: model.pte, inputs/*.bin (NCHW float32), reference.npz.
usage: python tests/device/export_partial_check.py, then for each <name>
  tests/device/run_rknn_pte.sh out/rknn_partial/<name>/model.pte out/rknn_partial/<name>
  python tests/device/compare_rknn_outputs.py out/rknn_partial/<name> --min-cosine 0.99
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch_rknn import RknnOptions, RknnPartitioner, transform_passes
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

OUT = Path("out/rknn_partial")
SHAPE = (1, 3, 32, 32)


class MiddleEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.norm = nn.GroupNorm(1, 16)
        self.tail = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(16, 10))

    def forward(self, x):
        return torch.softmax(self.tail(self.norm(self.head(x))), -1)


class Gated(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU())
        self.value = nn.Conv2d(16, 8, 1)
        self.gate = nn.Conv2d(16, 8, 1)
        self.tail = nn.Conv2d(8, 4, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        return self.tail(torch.maximum(self.value(x), self.gate(x)))


def summary(pte):
    plan = deserialize_pte_binary(pte).program.execution_plan[0]
    calls = Counter()
    for chain in plan.chains:
        for i in chain.instructions:
            args = i.instr_args
            if type(args).__name__ == "DelegateCall":
                calls["delegate"] += 1
            elif type(args).__name__ == "KernelCall":
                calls[plan.operators[args.op_index].name] += 1
    return dict(calls)


def export(name, model):
    out = OUT / name
    model = model.eval()
    calib = [torch.randn(SHAPE) for _ in range(16)]
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), quantizer)
    for x in calib:
        prepared(x)
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(torch.export.export(quantized, (calib[0],)),
                                       transform_passes=transform_passes(),
                                       partitioner=[RknnPartitioner(RknnOptions(dump_dir=str(out)))])
    pte = edge.to_executorch().buffer
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "model.pte").write_bytes(pte)

    pt2e, flt = [], []
    with torch.no_grad():
        for i in range(8):
            x = torch.randn(SHAPE)
            x.numpy().tofile(out / "inputs" / f"{i:03d}.bin")
            pt2e.append(quantized(x)[0].numpy().ravel())
            flt.append(model(x)[0].numpy().ravel())
    np.savez(out / "reference.npz", pt2e=np.stack(pt2e), float=np.stack(flt))
    print(f"{name}: {summary(pte)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    export("middle_end", MiddleEnd())
    export("gated", Gated())


if __name__ == "__main__":
    main()
