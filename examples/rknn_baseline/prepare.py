"""Export MobileNetV2 to ONNX and prepare images for the plain RKNN baseline.

Runs in the ExecuTorch venv. Writes to out/baseline/:
  mobilenet_v2.onnx    float model, normalisation not included
  calib/*.png          224x224 RGB crops from Imagenette train, for int8 calibration
  calib.txt            list of those files, the format rknn.build(dataset=...) takes
  eval/*.png           224x224 RGB crops from Imagenette val
  eval.json            ImageNet label and PyTorch float top-5 for each eval image
  board.txt            the subset of eval images run on the board
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import v2 as T

IMAGENETTE = Path("/opt/data/imagenette2-160")

# Imagenette WordNet ids -> ImageNet-1k class index.
WNID_TO_INDEX = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

crop = T.Compose([T.Resize(256), T.CenterCrop(224)])
normalize = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def use_global_average_pool(model):
    """Replace ReduceMean over H and W (keepdims=1) with GlobalAveragePool.

    The exporter lowers adaptive_avg_pool2d(x, 1) to ReduceMean, which the
    RV1106 runtime supports on neither the NPU nor its CPU fallback.
    """
    consts = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    rank = {
        v.name: len(v.type.tensor_type.shape.dim)
        for v in [*model.graph.value_info, *model.graph.input]
    }
    for i, node in enumerate(model.graph.node):
        if node.op_type != "ReduceMean" or len(node.input) != 2:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        axes = consts.get(node.input[1])
        if (
            attrs.get("keepdims", 1) == 1
            and rank.get(node.input[0]) == 4
            and axes is not None
            and sorted(int(a) % 4 for a in axes) == [2, 3]
        ):
            model.graph.node[i].CopyFrom(
                onnx.helper.make_node(
                    "GlobalAveragePool", [node.input[0]], list(node.output), name=node.name
                )
            )


def pick(split, per_class):
    for wnid in sorted(WNID_TO_INDEX):
        for path in sorted((IMAGENETTE / split / wnid).glob("*.JPEG"))[:per_class]:
            yield wnid, path


def save_crops(split, per_class, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for wnid, path in pick(split, per_class):
        dst = out_dir / f"{path.stem}.png"
        crop(Image.open(path).convert("RGB")).save(dst)
        saved.append((wnid, dst))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/baseline"))
    parser.add_argument("--calib-per-class", type=int, default=5)
    parser.add_argument("--eval-per-class", type=int, default=50)
    parser.add_argument("--board-per-class", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    assert categories[WNID_TO_INDEX["n03888257"]] == "parachute"
    model = mobilenet_v2(weights=weights).eval()

    onnx_path = args.out / "mobilenet_v2.onnx"
    torch.onnx.export(
        model,
        (torch.randn(1, 3, 224, 224),),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=19,  # newest rknn-toolkit2 1.6.0 accepts
        dynamo=True,
        external_data=False,
    )
    # The toolkit's onnx 1.14.1 rejects IR version 10; opset 19 is valid in IR 9.
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    use_global_average_pool(onnx_model)
    assert not any(n.op_type == "ReduceMean" for n in onnx_model.graph.node)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, onnx_path)
    print(f"wrote {onnx_path}")

    calib = save_crops("train", args.calib_per_class, args.out / "calib")
    (args.out / "calib.txt").write_text(
        "".join(f"{p.resolve()}\n" for _, p in calib)
    )
    print(f"wrote {len(calib)} calibration images")

    records = []
    with torch.no_grad():
        for wnid, png in save_crops("val", args.eval_per_class, args.out / "eval"):
            logits = model(normalize(Image.open(png)).unsqueeze(0))[0]
            records.append(
                {
                    "image": png.name,
                    "label": WNID_TO_INDEX[wnid],
                    "torch_top5": logits.topk(5).indices.tolist(),
                }
            )
    (args.out / "eval.json").write_text(json.dumps(records, indent=1) + "\n")
    per_label = {}
    for r in records:
        per_label.setdefault(r["label"], []).append(r["image"])
    (args.out / "board.txt").write_text(
        "".join(f"{img}\n" for imgs in per_label.values() for img in imgs[: args.board_per_class])
    )
    top1 = np.mean([r["torch_top5"][0] == r["label"] for r in records])
    print(f"wrote {len(records)} eval images; PyTorch float top-1 = {top1:.3f}")


if __name__ == "__main__":
    main()
