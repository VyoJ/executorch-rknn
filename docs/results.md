# Results

Pico Pro Max (RV1106) unless noted, `rkipc` stopped, ExecuTorch v1.6.0,
rknn-toolkit2 and `librknnmrt.so` 1.6.0. Top-1 is over all 1000 ImageNet
classes on Imagenette validation images.

## MobileNetV2

torchvision `IMAGENET1K_V1`, quantised with PT2E (`XNNPACKQuantizer`,
per-channel weights, 50 calibration images). The whole graph is one delegate.
100 images, 10 per class.

| | Per run | Top-1 |
|---|---|---|
| PyTorch float, host | | 0.640 |
| PT2E int8, host | | 0.620 |
| `rknn_run` alone, input already int8 | 9.04 ms | |
| Plain RKNN application, float input quantised each run | 10.62 ms | |
| Through ExecuTorch and the delegate | 10.64 ms | 0.600 |
| Same `.pte`, Pico Mini B (RV1103) | 14.6 ms | 0.600 |

The delegate's outputs are bit-identical to running its `.rknn` directly.
Peak RSS is 7.1 MB, plus 5.4 MB of NPU memory. Mean cosine of the logits
against PT2E is 0.972; the NPU rounds differently near quantisation steps and
that adds up over 52 convolutions.

Plain RKNN with its own calibration on the same model reaches 0.640 top-1 at
9.09 ms.

## MobileViT-XS

`apple/mobilevit-x-small`, 256x256. Plain RKNN builds the float model with its
own calibration but `rknn_init` fails on the board: it asks for 80 MB of NPU
memory, and the region is 66 MB (`examples/mobilevit/rknn_alone.py`).

Through the delegate, convolutions and linears run in 41 RKNN partitions;
LayerNorm, attention and the final mean run on the CPU.

Accuracy on the host over all 3,925 validation images
(`examples/mobilevit/evaluate.py`), with the paired difference from float:

| | Top-1 | vs float |
|---|---|---|
| Float | 0.709 +- 0.014 | |
| PT2E, per-channel weights | 0.528 +- 0.016 | -0.181 +- 0.016 |
| PT2E, first transformer block left in float | 0.661 +- 0.015 | -0.048 +- 0.011 |
| PT2E with 300 steps of quantisation-aware training | 0.756 +- 0.013 | +0.046 +- 0.012 |

Quantisation-aware training distils the float model's output over all 1000
classes, but only on Imagenette training images; it ends above float on these
10 classes because it has adapted to them.

On the board, the QAT model gets 0.744 top-1 on 500 images (float 0.712, PT2E
0.780), with a mean cosine of 0.959 against PT2E.

| Per image | Pico Pro Max | Pico Mini B |
|---|---|---|
| CPU part on portable kernels | 2,220 ms, 16.1 MB RSS | 2,155 ms, 14.9 MB RSS |
| CPU part on XNNPACK | 519 ms, 14.5 MB RSS | 555 ms, 13.6 MB RSS |

The two CPU builds agree on the outputs (cosine 0.9994, same top-1 on 49 of
50 images). Where the 519 ms goes, from an ETDump profile:

| | ms |
|---|---|
| RKNN delegates, including float/int8 conversion | 162 |
| XNNPACK batch matmul (attention) | 191 |
| XNNPACK softmax | 71 |
| XNNPACK reshapes, transposes, converts, divide | 55 |
| LayerNorm, portable kernels | 37 |

## XNNPACK on the CPU alone

Float MobileNetV2 entirely on the Cortex-A7 takes 0.615 s with XNNPACK and
29.1 s with ExecuTorch's portable kernels, with outputs matching PyTorch.
It needs 31 MB RSS, more than the Mini B has free.
