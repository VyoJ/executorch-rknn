"""Top-1 of MobileViT-XS variants on the host, over all of Imagenette's
validation set (3,925 images), against the float model.

Runs in the ExecuTorch venv. Each image is loaded once and run through every
model. Prints top-1 with a 95% interval, and the paired difference from
float with its own interval: the models see the same images, so the
difference is far less noisy than the two top-1 figures.

  python examples/mobilevit/evaluate.py --program qat=out/mobilevit_qat/program.pt2
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MobileViT, images  # noqa: E402
from export import quantize  # noqa: E402

EARLY_BLOCK = "model.mobilevit.encoder.layer.0.layer.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--program", action="append", default=[], metavar="NAME=PATH",
                        help="an exported program saved by export.py (program.pt2)")
    parser.add_argument("--per-class", type=int, default=10**6, help="validation images per class (default: all)")
    parser.add_argument("--calib-per-class", type=int, default=5)
    args = parser.parse_args()

    model = MobileViT()
    calib = [x for _, x in images("train", args.calib_per_class)]
    models = {
        "float": model,
        "pt2e": quantize(model, calib),
        "pt2e, first block float": quantize(model, calib, [f"{EARLY_BLOCK}.expand_1x1", f"{EARLY_BLOCK}.conv_3x3"]),
    }
    for spec in args.program:
        name, path = spec.split("=", 1)
        models[name] = torch.export.load(path).module()

    correct = {name: [] for name in models}
    with torch.no_grad():
        for i, (label, x) in enumerate(images("val", args.per_class)):
            for name, m in models.items():
                correct[name].append(int(m(x).argmax()) == label)
            if i % 500 == 0:
                print(f"{i} images", flush=True)

    ref = np.array(correct["float"], dtype=float)
    n = len(ref)
    print(f"\n{n} images")
    for name, c in correct.items():
        c = np.array(c, dtype=float)
        p = c.mean()
        line = f"{name:28s} top-1 {p:.3f} ± {1.96 * math.sqrt(p * (1 - p) / n):.3f}"
        if name != "float":
            d = c - ref
            line += f"   vs float {d.mean():+.3f} ± {1.96 * d.std(ddof=1) / math.sqrt(n):.3f}"
        print(line)


if __name__ == "__main__":
    main()
