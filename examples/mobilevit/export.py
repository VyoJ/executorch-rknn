"""Export MobileViT-XS, quantised with PT2E, to a .pte that runs its
convolutions and linears in RKNN and its LayerNorms and attention on the CPU.

Runs in the ExecuTorch venv. Writes:
  out/mobilevit.pte            the program
  out/mobilevit/rknn_<i>/      what each delegate handed to rknn-toolkit2
  out/mobilevit/inputs/*.bin   check images, NCHW float32 (the .pte's input)
  out/mobilevit/reference.npz  labels, float and PT2E logits for those images
  out/mobilevit/program.pt2    the exported model before lowering (PT2E int8, or float)
"""

import argparse
import copy
import logging
import random
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
from executorch_rknn import RknnOptions, RknnPartitioner, transform_passes
import torch.nn.functional as F
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MobileViT, images  # noqa: E402


def in_modules(prefixes):
    """Node filter: True if the node comes from a module under one of the prefixes."""
    def check(node):
        paths = [path for path, _ in node.meta.get("nn_module_stack", {}).values()]
        return any(p == prefix or p.startswith(prefix + ".") for p in paths for prefix in prefixes)
    return check


def quantize(model, calib, keep_float=()):
    """PT2E int8. Operators from modules under keep_float stay float, so the
    partitioner leaves them on the CPU."""
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    if keep_float:
        is_float = in_modules(keep_float)
        quantizer.filter_fn = lambda node: not is_float(node)
    prepared = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), quantizer)
    with torch.no_grad():
        for x in calib:
            prepared(x)
    return convert_pt2e(prepared)


def quantize_qat(model, calib, train, steps, lr, batch=8, seed=0):
    """PT2E quantisation-aware training, distilled from the float model's
    logits over all 1000 classes (training on the 10 Imagenette labels would
    favour the classes we evaluate on). BatchNorm stays frozen: the graph is
    traced at batch 1, so batch statistics would be one image's."""
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
    # torch.export shares parameters with the module: train a copy, or the
    # float model trains too.
    student = prepare_qat_pt2e(torch.export.export(copy.deepcopy(model), (calib[0],)).module(), quantizer)
    move_exported_model_to_eval(student)  # before calibrating, or BatchNorm's running statistics change
    with torch.no_grad():
        for x in calib:
            student(x)
        targets = [F.softmax(model(x), -1) for x in train]
    rng = random.Random(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for step in range(steps):
        loss = 0
        for i in rng.sample(range(len(train)), batch):
            loss = loss + F.kl_div(F.log_softmax(student(train[i]), -1), targets[i], reduction="batchmean")
        opt.zero_grad()
        (loss / batch).backward()
        opt.step()
        if step % 50 == 0:
            print(f"QAT step {step}: loss {float(loss.detach()) / batch:.4f}", flush=True)
    return convert_pt2e(student)


def program_summary(pte):
    """Delegate calls and CPU operator calls in the program."""
    plan = deserialize_pte_binary(pte).program.execution_plan[0]
    calls = Counter()
    for chain in plan.chains:
        for i in chain.instructions:
            args = i.instr_args
            if type(args).__name__ == "DelegateCall":
                calls[plan.delegates[args.delegate_index].id] += 1
            elif type(args).__name__ == "KernelCall":
                calls[plan.operators[args.op_index].name] += 1
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pte", type=Path, default=Path("out/mobilevit.pte"))
    parser.add_argument("--work", type=Path, default=Path("out/mobilevit"))
    parser.add_argument("--target", default="rv1106", choices=["rv1103", "rv1106"])
    parser.add_argument("--calib-per-class", type=int, default=5)
    parser.add_argument("--check-per-class", type=int, default=10)
    parser.add_argument("--min-partition-ops", type=int, default=1)
    parser.add_argument("--quant", default="pt2e", choices=["pt2e", "rknn"],
                        help="pt2e: PT2E int8; rknn: export float and let RKNN calibrate each delegate")
    parser.add_argument("--keep-float", nargs="*", default=[],
                        help="with pt2e: module paths whose operators stay float, on the CPU")
    parser.add_argument("--rknn-algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"])
    parser.add_argument("--qat-steps", type=int, default=0, help="with pt2e: quantisation-aware training steps")
    parser.add_argument("--qat-lr", type=float, default=5e-5)
    parser.add_argument("--qat-per-class", type=int, default=60, help="training images per class for QAT")
    parser.add_argument("--xnnpack", action="store_true",
                        help="run what RKNN leaves on the CPU through XNNPACK where it can")
    parser.add_argument("--program", type=Path,
                        help="lower this program.pt2 from an earlier run instead of quantising again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.work.mkdir(parents=True, exist_ok=True)

    model = MobileViT()
    calib = [x for _, x in images("train", args.calib_per_class)]
    options = RknnOptions(target_platform=args.target, dump_dir=str(args.work),
                          quantized_algorithm=args.rknn_algorithm)
    if args.program:
        program = torch.export.load(args.program)
        quantized = program.module()
        partitioner = RknnPartitioner(options, args.min_partition_ops)
    elif args.quant == "pt2e" and args.qat_steps:
        torch.manual_seed(0)
        train = [x for _, x in images("train", args.qat_per_class)]
        quantized = quantize_qat(model, calib, train, args.qat_steps, args.qat_lr)
        program = torch.export.export(quantized, (calib[0],))
        partitioner = RknnPartitioner(options, args.min_partition_ops)
    elif args.quant == "pt2e":
        quantized = quantize(model, calib, args.keep_float)
        program = torch.export.export(quantized, (calib[0],))
        partitioner = RknnPartitioner(options, args.min_partition_ops)
    else:
        quantized = None
        program = torch.export.export(model, (calib[0],))
        partitioner = RknnPartitioner(options, args.min_partition_ops, calibration_inputs=[(x,) for x in calib],
                                      calibration_program=program)
    partitioners = [partitioner]
    if args.xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioners.append(XnnpackPartitioner())
    edge = to_edge_transform_and_lower(program, transform_passes=transform_passes(), partitioner=partitioners)
    torch.export.save(program, args.work / "program.pt2")  # the exported (quantised) model, for evaluate.py
    pte = edge.to_executorch().buffer
    args.pte.write_bytes(pte)
    calls = program_summary(pte)
    delegates = {b: calls.pop(b) for b in ("RknnBackend", "XnnpackBackend") if b in calls}
    print(f"wrote {args.pte} ({len(pte)} bytes): "
          + ", ".join(f"{n} {b} calls" for b, n in delegates.items()) + "; CPU: "
          + ", ".join(f"{n} x {op}" for op, n in calls.most_common()))

    inputs_dir = args.work / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    labels, float_logits, pt2e_logits = [], [], []
    with torch.no_grad():
        for i, (label, x) in enumerate(images("val", args.check_per_class)):
            x.numpy().tofile(inputs_dir / f"{i:03d}.bin")
            labels.append(label)
            float_logits.append(model(x)[0].numpy())
            if quantized is not None:
                pt2e_logits.append(quantized(x)[0].numpy())
    reference = {"labels": np.array(labels), "float": np.stack(float_logits)}
    summary = f"top-1 float {np.mean(reference['float'].argmax(1) == reference['labels']):.3f}"
    if pt2e_logits:
        reference["pt2e"] = np.stack(pt2e_logits)
        summary += f", PT2E {np.mean(reference['pt2e'].argmax(1) == reference['labels']):.3f}"
    np.savez(args.work / "reference.npz", **reference)
    print(f"wrote {len(labels)} check inputs; {summary}")


if __name__ == "__main__":
    main()
