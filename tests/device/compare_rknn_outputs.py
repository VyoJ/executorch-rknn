"""Compare board outputs (<dir>/<outputs>/*.bin) with the PT2E and float
references written by an export script (<dir>/reference.npz: pt2e, float, and
labels for classifiers).

Run in the aot container after tests/device/run_rknn_pte.sh (outputs in
board-pte/) or tests/device/run_rknn_model.sh (board/). Exits non-zero if an
output is missing, the mean cosine with PT2E is below --min-cosine, or, for
classifiers, top-1 agreement with PT2E is below --min-agree.

PT2E is not a bit-exact reference for deep networks: its float arithmetic and
the NPU's integer arithmetic round differently near quantisation steps, and the
differences add up layer by layer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("--outputs", default="board-pte", help="subdirectory of dir with the board's outputs")
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--min-agree", type=float, default=0.8)
    args = parser.parse_args()

    ref = np.load(args.dir / "reference.npz")
    inputs = sorted((args.dir / "inputs").glob("*.bin"))
    out_dir = args.dir / args.outputs
    missing = [p.name for p in inputs if not (out_dir / p.name).exists()]
    if missing:
        print(f"missing board outputs: {missing}")
        return 1
    board = np.stack([np.fromfile(out_dir / p.name, np.float32) for p in inputs])
    pt2e, flt = ref["pt2e"].reshape(len(inputs), -1), ref["float"].reshape(len(inputs), -1)
    cos = [cosine(b, p) for b, p in zip(board, pt2e)]
    ok = np.mean(cos) >= args.min_cosine
    summary = (f"{len(inputs)} inputs; mean cosine vs PT2E {np.mean(cos):.4f} (min {np.min(cos):.4f}), "
               f"vs float {np.mean([cosine(b, f) for b, f in zip(board, flt)]):.4f}; "
               f"max abs vs PT2E {np.abs(board - pt2e).max():.4g}")

    if "labels" in ref.files:
        labels = ref["labels"]
        agree = float(np.mean(board.argmax(1) == pt2e.argmax(1)))
        ok &= agree >= args.min_agree
        summary += (f"\ntop-1 vs PT2E {agree:.3f}, vs float {np.mean(board.argmax(1) == flt.argmax(1)):.3f}; "
                    f"accuracy board {np.mean(board.argmax(1) == labels):.3f}, "
                    f"PT2E {np.mean(pt2e.argmax(1) == labels):.3f}, float {np.mean(flt.argmax(1) == labels):.3f}")
    print(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
