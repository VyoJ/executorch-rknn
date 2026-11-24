"""Try MobileViT-XS with plain RKNN: float ONNX, RKNN's own int8 calibration,
target rv1106. Demo D1 starts by showing this does not work.

  scripts/dk aot python examples/mobilevit/rknn_alone.py export
  scripts/dk aot /opt/rknn/bin/python examples/mobilevit/rknn_alone.py build

Writes out/mobilevit_rknn_alone/: model.onnx, calib/*.npy, calib.txt, and
model.rknn plus build.log if the build gets that far.
"""

import argparse
import sys
from pathlib import Path

OUT = Path("out/mobilevit_rknn_alone")


def export(calib_per_class):
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import SIZE, MobileViT, images

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calib").mkdir(exist_ok=True)
    paths = []
    for i, (_, x) in enumerate(images("train", calib_per_class)):
        path = OUT / "calib" / f"{i:03d}.npy"
        np.save(path, x.numpy())
        paths.append(str(path.resolve()))
    (OUT / "calib.txt").write_text("\n".join(paths) + "\n")
    torch.onnx.export(MobileViT(), (torch.zeros(1, 3, SIZE, SIZE),), str(OUT / "model.onnx"),
                      opset_version=13, dynamo=False, input_names=["x"], output_names=["logits"])
    print(f"wrote {OUT}/model.onnx and {len(paths)} calibration inputs")


def build(target):
    from rknn.api import RKNN

    rknn = RKNN()
    rknn.config(target_platform=target)
    if rknn.load_onnx(model=str(OUT / "model.onnx")) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=True, dataset=str(OUT / "calib.txt")) != 0:
        raise SystemExit("build failed")
    if rknn.export_rknn(str(OUT / "model.rknn")) != 0:
        raise SystemExit("export_rknn failed")
    print(f"wrote {OUT}/model.rknn")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["export", "build"])
    parser.add_argument("--calib-per-class", type=int, default=5)
    parser.add_argument("--target", default="rv1106", choices=["rv1103", "rv1106"])
    args = parser.parse_args()
    export(args.calib_per_class) if args.step == "export" else build(args.target)


if __name__ == "__main__":
    main()
