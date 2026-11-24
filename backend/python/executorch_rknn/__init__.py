"""ExecuTorch backend for Rockchip RKNN NPUs (RV1103 / RV1106)."""

# Quantize/dequantize nodes left on the CPU need their out variants at export.
import executorch.kernels.quantized  # noqa: F401

from .compile_specs import RknnOptions
from .partitioner import RknnPartitioner
from .passes import DuplicateQuantizePass, transform_passes
from .preprocess import RknnBackend

__all__ = ["DuplicateQuantizePass", "RknnBackend", "RknnOptions", "RknnPartitioner", "transform_passes"]
