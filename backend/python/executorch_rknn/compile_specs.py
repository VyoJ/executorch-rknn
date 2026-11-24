"""Options for RknnBackend, carried as ExecuTorch CompileSpecs (JSON values)."""

import json
from dataclasses import asdict, dataclass

from executorch.exir.backend.compile_spec_schema import CompileSpec


@dataclass
class RknnOptions:
    target_platform: str = "rv1106"
    # Level 2 reproduces PT2E's int8 output exactly on shallow networks but made
    # MobileNetV2 68% slower for no accuracy gain.
    optimization_level: int = 3
    # Fold input normalisation into the model: input = (x - mean) / std per channel.
    mean_values: list | None = None
    std_values: list | None = None
    # Float graph only: let RKNN calibrate on this dataset list,
    # or on a directory of <input name>_<sample>.npy (RknnPartitioner writes these).
    calibration_dataset: str | None = None
    # RKNN's calibration method: normal, mmse or kl_divergence.
    quantized_algorithm: str = "normal"
    # Keep model.onnx and model.rknn here for inspection.
    dump_dir: str | None = None

    def compile_specs(self):
        return [CompileSpec(k, json.dumps(v).encode()) for k, v in asdict(self).items()]

    @classmethod
    def from_compile_specs(cls, specs):
        return cls(**{s.key: json.loads(s.value.decode()) for s in specs})
