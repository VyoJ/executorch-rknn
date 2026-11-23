"""Export MobileNetV2, quantised with PT2E, to a .pte whose whole graph runs in
the RKNN delegate.

Runs in the ExecuTorch venv. Writes:
  out/mv2_rknn.pte           the program
  out/mv2_rknn/model.onnx    what the delegate handed to rknn-toolkit2
  out/mv2_rknn/model.rknn    the .rknn inside the .pte, for running on the board
  out/mv2_rknn/inputs/*.bin  check images, NHWC float32, normalised
  out/mv2_rknn/reference.npz labels and float and PT2E logits for those images
"""

import argparse
import sys
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
from executorch_rknn import RknnOptions, RknnPartitioner
from executorch_rknn import blob
from PIL import Image
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parent / "rknn_baseline"))
from prepare import WNID_TO_INDEX, crop, normalize, pick  # noqa: E402


def images(split, per_class):
    for wnid, path in pick(split, per_class):
        # ToImage returns channels-last strides, which export would carry into the
        # graph as an as_strided_copy.
        x = normalize(crop(Image.open(path).convert("RGB"))).unsqueeze(0).contiguous()
        yield WNID_TO_INDEX[wnid], x


def quantize(model, calib):
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), quantizer)
    with torch.no_grad():
        for x in calib:
            prepared(x)
    return convert_pt2e(prepared)


def delegate_payloads(pte):
    """Check the program is one RKNN delegate call and nothing else; return its blob."""
    program = deserialize_pte_binary(pte).program
    plan = program.execution_plan[0]
    calls = Counter(type(i.instr_args).__name__ for chain in plan.chains for i in chain.instructions)
    backends = [d.id for d in plan.delegates]
    if calls != Counter({"DelegateCall": 1}) or backends != ["RknnBackend"] or plan.operators:
        raise SystemExit(f"expected one RknnBackend call and no operators, got {dict(calls)}, "
                         f"delegates {backends}, operators {[op.name for op in plan.operators]}")
    return program.backend_delegate_data[plan.delegates[0].processed.index].data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pte", type=Path, default=Path("out/mv2_rknn.pte"))
    parser.add_argument("--work", type=Path, default=Path("out/mv2_rknn"))
    parser.add_argument("--target", default="rv1106", choices=["rv1103", "rv1106"])
    parser.add_argument("--calib-per-class", type=int, default=5)
    parser.add_argument("--check-per-class", type=int, default=1)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    calib = [x for _, x in images("train", args.calib_per_class)]
    quantized = quantize(model, calib)

    options = RknnOptions(target_platform=args.target, dump_dir=str(args.work))
    edge = to_edge_transform_and_lower(torch.export.export(quantized, (calib[0],)),
                                       partitioner=[RknnPartitioner(options)])
    pte = edge.to_executorch().buffer
    args.pte.write_bytes(pte)
    print(f"wrote {args.pte} ({len(pte)} bytes)")

    payload = blob.unpack(delegate_payloads(pte))
    if payload.rknn != (args.work / "model.rknn").read_bytes():
        raise SystemExit("the .rknn in the .pte differs from the one the delegate built")
    print(f"one RknnBackend call, no CPU operators; inputs {[t.shape for t in payload.inputs]}, "
          f"outputs {[t.shape for t in payload.outputs]}, .rknn {len(payload.rknn)} bytes")

    inputs_dir = args.work / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    labels, float_logits, pt2e_logits = [], [], []
    with torch.no_grad():
        for i, (label, x) in enumerate(images("val", args.check_per_class)):
            # RKNN's input on RV1106 is NHWC.
            x.permute(0, 2, 3, 1).contiguous().numpy().tofile(inputs_dir / f"{i:03d}.bin")
            labels.append(label)
            float_logits.append(model(x)[0].numpy())
            pt2e_logits.append(quantized(x)[0].numpy())
    np.savez(args.work / "reference.npz", labels=np.array(labels),
             float_logits=np.stack(float_logits), pt2e_logits=np.stack(pt2e_logits))
    print(f"wrote {len(labels)} check inputs to {inputs_dir}")


if __name__ == "__main__":
    main()
