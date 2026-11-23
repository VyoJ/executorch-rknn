# executorch-rknn

ExecuTorch on Rockchip RV1103/RV1106 (Luckfox Pico), with the goal of an RKNN NPU
delegate. See `PLAN.md` for the phases and `docs/` for results.

## Setup

Needs Docker, git and adb on the host. Everything else runs in containers.

```
scripts/bootstrap.sh                                  # submodules
```

The `aot` and `cross` images build on first use of `scripts/dk`.

## Build and run

```
scripts/dk cross scripts/build_runtime.sh             # ExecuTorch executor_runner
scripts/dk aot python examples/export_portable_models.py
tests/device/run_portable_models.sh                   # board, over adb
scripts/dk aot python tests/device/compare_outputs.py
```

MobileNetV2 through the RKNN delegate:

```
scripts/dk aot python -m pytest -q tests/aot
scripts/dk aot python examples/export_mobilenet_v2.py --check-per-class 10
scripts/dk cross scripts/build_rknn_runner.sh        # executor_runner + RknnBackend
tests/device/run_rknn_pte.sh out/mv2_rknn.pte out/mv2_rknn   # board, over adb
scripts/dk aot python tests/device/compare_rknn_outputs.py out/mv2_rknn
```

To run a bare `.rknn` (the one the delegate built is kept in `out/mv2_rknn/`):
`scripts/dk cross scripts/build_rknn_bench.sh`, then
`tests/device/run_rknn_model.sh out/mv2_rknn` and add `--outputs board` to the
comparison.

Plain RKNN baseline: see `docs/baseline.md`.
