"""Compare board outputs (board-<i>.bin) with x86 outputs (expected-<i>.bin).

Run in the aot container after tests/device/run_portable_models.sh.
Exits non-zero if any model is missing outputs or differs beyond tolerance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare(model_dir, rtol, atol):
    meta = json.loads((model_dir / "meta.json").read_text())
    ok = True
    for i, out in enumerate(meta["outputs"]):
        board_file = model_dir / f"board-{i}.bin"
        if not board_file.exists():
            print(f"{model_dir.name}[{i}]: missing {board_file.name}")
            ok = False
            continue
        dtype = np.dtype(out["dtype"])
        expected = np.fromfile(model_dir / f"expected-{i}.bin", dtype).reshape(out["shape"])
        board = np.fromfile(board_file, dtype).reshape(out["shape"])
        max_abs = float(np.max(np.abs(board.astype(np.float64) - expected)))
        exact = np.array_equal(board, expected)
        close = np.allclose(board, expected, rtol=rtol, atol=atol)
        print(
            f"{model_dir.name}[{i}]: shape={out['shape']} "
            f"{'bit-exact' if exact else f'max_abs_diff={max_abs:.3g}'} "
            f"{'OK' if close else 'MISMATCH'}"
        )
        ok &= close
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", type=Path, nargs="?", default=Path("out/models"))
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    results = [compare(d, args.rtol, args.atol) for d in sorted(args.models.iterdir()) if d.is_dir()]
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
