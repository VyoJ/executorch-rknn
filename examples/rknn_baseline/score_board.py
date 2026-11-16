"""Score the board run (out/baseline/board-eval.log) against eval.json."""

import argparse
import json
import re
from pathlib import Path


def parse_log(text):
    top5, current = {}, None
    for line in text.splitlines():
        if line.startswith("IMAGE "):
            current = line.split(maxsplit=1)[1].strip()
            top5[current] = []
        elif m := re.match(r"\s*[-\d.]+ - (\d+)\s*$", line):
            top5[current].append(int(m.group(1)))
    return top5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("out/baseline"))
    args = parser.parse_args()

    all_records = json.loads((args.dir / "eval.json").read_text())
    subset = set((args.dir / "board.txt").read_text().split())
    board = parse_log((args.dir / "board-eval.log").read_text())
    records = [r for r in all_records if r["image"] in subset]
    missing = [r["image"] for r in records if not board.get(r["image"])]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from board log, e.g. {missing[0]}")

    n = len(records)
    for r in records:
        r["board_top5"] = board[r["image"]]
    (args.dir / "eval.json").write_text(json.dumps(all_records, indent=1) + "\n")

    def top1(key):
        return sum(r[key][0] == r["label"] for r in records) / n

    def agree(a, b):
        return sum(r[a][0] == r[b][0] for r in records) / n

    print(f"top-1 on {n} images:")
    for key in ("torch_top5", "rknn_sim_top5", "board_top5"):
        print(f"  {key:14s} {top1(key):.3f}")
    print(f"top-1 agreement board vs simulator: {agree('board_top5', 'rknn_sim_top5'):.3f}")
    print(f"top-1 agreement board vs PyTorch:   {agree('board_top5', 'torch_top5'):.3f}")


if __name__ == "__main__":
    main()
