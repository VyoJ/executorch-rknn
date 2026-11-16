"""Export small models to .pte for the portable-kernel runtime.

For each model, writes to out/models/<name>/:
  model.pte        program with no delegates
  input-<i>.bin    raw input tensors, as executor_runner --inputs expects
  expected-<i>.bin outputs from the ExecuTorch Python runtime on x86
  meta.json        output shapes and dtypes, for tests/device/compare_outputs.py
"""

import argparse
import json
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU6(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(16, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def models():
    yield "add", Add(), (torch.randn(2, 3), torch.randn(2, 3))
    yield "convnet", ConvNet(), (torch.randn(1, 3, 32, 32),)


def export(name, model, inputs, out_dir):
    model.eval()
    program = to_edge_transform_and_lower(
        torch.export.export(model, inputs)
    ).to_executorch()

    out_dir.mkdir(parents=True, exist_ok=True)
    pte = out_dir / "model.pte"
    pte.write_bytes(program.buffer)
    for i, t in enumerate(inputs):
        t.contiguous().numpy().tofile(out_dir / f"input-{i}.bin")

    method = Runtime.get().load_program(pte).load_method("forward")
    outputs = method.execute(list(inputs))
    with torch.no_grad():
        eager = model(*inputs)
    eager = eager if isinstance(eager, tuple) else (eager,)

    meta = {"outputs": []}
    for i, (et, ref) in enumerate(zip(outputs, eager)):
        torch.testing.assert_close(et, ref, rtol=1e-4, atol=1e-5)
        et.contiguous().numpy().tofile(out_dir / f"expected-{i}.bin")
        meta["outputs"].append(
            {"shape": list(et.shape), "dtype": str(et.dtype).removeprefix("torch.")}
        )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {pte} ({pte.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/models"))
    args = parser.parse_args()

    torch.manual_seed(0)
    for name, model, inputs in models():
        export(name, model, inputs, args.out / name)


if __name__ == "__main__":
    main()
