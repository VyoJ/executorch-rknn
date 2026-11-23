"""ExecuTorch backend for Rockchip RKNN NPUs (RV1103 / RV1106)."""

from .compile_specs import RknnOptions
from .partitioner import RknnPartitioner
from .preprocess import RknnBackend

__all__ = ["RknnBackend", "RknnOptions", "RknnPartitioner"]
