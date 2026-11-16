"""Convert MobileNetV2 ONNX to an int8 .rknn and score it in RKNN's x86 simulator.

Runs in the RKNN venv (/opt/rknn/bin/python), after prepare.py. Writes
out/baseline/mobilenet_v2.rknn and adds "rknn_sim_top5" to each record in
eval.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rknn.api import RKNN

MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("out/baseline"))
    parser.add_argument("--target", default="rv1106", choices=["rv1103", "rv1106"])
    args = parser.parse_args()

    rknn = RKNN()
    rknn.config(mean_values=[MEAN], std_values=[STD], target_platform=args.target)
    if rknn.load_onnx(model=str(args.dir / "mobilenet_v2.onnx")) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=True, dataset=str(args.dir / "calib.txt")) != 0:
        raise SystemExit("build failed")
    rknn_path = args.dir / "mobilenet_v2.rknn"
    if rknn.export_rknn(str(rknn_path)) != 0:
        raise SystemExit("export_rknn failed")
    print(f"wrote {rknn_path} ({rknn_path.stat().st_size} bytes)")

    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime (simulator) failed")
    eval_json = args.dir / "eval.json"
    records = json.loads(eval_json.read_text())
    for r in records:
        image = np.asarray(Image.open(args.dir / "eval" / r["image"]))[None]
        logits = rknn.inference(inputs=[image], data_format="nhwc")[0].reshape(-1)
        r["rknn_sim_top5"] = np.argsort(-logits)[:5].tolist()
    eval_json.write_text(json.dumps(records, indent=1) + "\n")
    rknn.release()

    def top1(key):
        return np.mean([r[key][0] == r["label"] for r in records])

    agree = np.mean([r["rknn_sim_top5"][0] == r["torch_top5"][0] for r in records])
    print(
        f"top-1 on {len(records)} images: PyTorch float {top1('torch_top5'):.3f}, "
        f"RKNN int8 simulator {top1('rknn_sim_top5'):.3f}; "
        f"top-1 agreement {agree:.3f}"
    )


if __name__ == "__main__":
    main()
