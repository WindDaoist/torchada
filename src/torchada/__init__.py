"""
torchada - Adapter package for torch_musa to act exactly like PyTorch CUDA.

torchada provides a unified interface that works transparently on both
NVIDIA GPUs (CUDA) and Moore Threads GPUs (MUSA).

Usage:
    Just import torchada at the top of your script, then use standard
    torch.cuda.* and torch.utils.cpp_extension APIs as you normally would.
    torchada patches PyTorch to transparently redirect to MUSA on
    Moore Threads hardware.

    # Add this at the top of your script:
    import torchada

    # Then use standard torch.cuda APIs - they work on MUSA too!
    x = torch.randn(3, 3).cuda()
    torch.cuda.synchronize()

    # Note: torch.cuda.is_available() is NOT patched (returns False on MUSA).
    # For GPU availability checks, see examples/migrate_existing_project.md.

    # Build extensions using standard imports:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

__version__ = "0.1.53"

import enum as _enum

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return format(self.value, format_spec)

    _enum.StrEnum = _StrEnum

from . import cuda, utils
from ._patch import (
    apply_patches,
    get_original_init_process_group,
    is_patched,
    patch_allocator_pool_lifecycle,
)
from ._platform import (
    Platform,
    detect_platform,
    get_device_name,
    get_torch_device_module,
    is_cpu_platform,
    is_cuda_like_device,
    is_cuda_platform,
    is_gpu_device,
    is_musa_platform,
)
from ._runtime import (
    cublas_to_mublas_name,
    cuda_to_musa_name,
    curand_to_murand_name,
    nccl_to_mccl_name,
)
from .triton.autotune.fused_moe import set_default_moe_config_dir
from .utils.cpp_extension import CUDA_HOME

# Automatically apply patches on import
apply_patches()

# Set default MoE config path for SGL and vLLM
set_default_moe_config_dir()

# Load C++ operator overrides if enabled via TORCHADA_ENABLE_CPP_OPS=1
from ._cpp_ops import load_cpp_ops

cpp_ops_module = load_cpp_ops()
patch_allocator_pool_lifecycle(cpp_ops_module)


def get_version() -> str:
    """Return the version of torchada."""
    return __version__


def get_platform() -> Platform:
    """Return the detected platform."""
    return detect_platform()


def get_backend():
    """
    Get the underlying torch device module (torch.cuda or torch.musa).

    Returns:
        The torch.cuda or torch.musa module.
    """
    return get_torch_device_module()


__all__ = [
    # Version
    "__version__",
    "get_version",
    # Modules
    "cuda",
    "utils",
    # Platform detection
    "Platform",
    "detect_platform",
    "is_musa_platform",
    "is_cuda_platform",
    "is_cpu_platform",
    "get_device_name",
    "get_platform",
    "get_backend",
    # Device helpers
    "is_gpu_device",
    "is_cuda_like_device",
    # Patching
    "apply_patches",
    "is_patched",
    "get_original_init_process_group",
    # C++ Extension building
    "CUDA_HOME",
    # Runtime name conversion utilities
    "cuda_to_musa_name",
    "nccl_to_mccl_name",
    "cublas_to_mublas_name",
    "curand_to_murand_name",
]
