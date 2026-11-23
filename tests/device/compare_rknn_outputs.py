"""Compare board outputs (<dir>/board/*.bin) with the PT2E and float references
written by the export script (<dir>/reference.npz).

Run in the aot container after tests/device/run_rknn_model.sh. Exits non-zero
if an output is missing or top-1 agreement with PT2E is below --min-agree.
PT2E is not a bit-exact reference for deep networks: its float arithmetic and
the NPU's integer arithmetic round differently near quantisation steps, and the
differences add up layer by layer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("--min-agree", type=float, default=0.8)
    args = parser.parse_args()

    ref = np.load(args.dir / "reference.npz")
    inputs = sorted((args.dir / "inputs").glob("*.bin"))
    missing = [p.name for p in inputs if not (args.dir / "board" / p.name).exists()]
    if missing:
        print(f"missing board outputs: {missing}")
        return 1
    board = np.stack([np.fromfile(args.dir / "board" / p.name, np.float32) for p in inputs])
    pt2e, flt, labels = ref["pt2e_logits"], ref["float_logits"], ref["labels"]

    print(f"{'image':>5} {'label':>5} {'board':>5} {'PT2E':>5} {'float':>5} {'cos PT2E':>9} {'max abs PT2E':>13}")
    for i, p in enumerate(inputs):
        print(f"{p.stem:>5} {labels[i]:5d} {board[i].argmax():5d} {pt2e[i].argmax():5d} {flt[i].argmax():5d} "
              f"{cosine(board[i], pt2e[i]):9.6f} {np.abs(board[i] - pt2e[i]).max():13.4g}")
    agree = float(np.mean(board.argmax(1) == pt2e.argmax(1)))
    mean_cos = np.mean([cosine(b, p) for b, p in zip(board, pt2e)])
    print(f"{len(inputs)} images; mean cosine vs PT2E {mean_cos:.4f}; top-1 vs PT2E {agree:.3f}, vs float {np.mean(board.argmax(1) == flt.argmax(1)):.3f}; "
          f"accuracy board {np.mean(board.argmax(1) == labels):.3f}, PT2E {np.mean(pt2e.argmax(1) == labels):.3f}, "
          f"float {np.mean(flt.argmax(1) == labels):.3f}")
    return 0 if agree >= args.min_agree else 1


if __name__ == "__main__":
    sys.exit(main())
