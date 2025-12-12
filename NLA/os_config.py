"""
Runtime OS tuning for cross-platform runs.

On import, this module detects the host OS and applies light-weight runtime
adjustments. Windows 11 gets conservative multiprocessing defaults and ANSI
color support; all platforms attempt to enable cuDNN benchmarking if PyTorch
with CUDA is available.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Optional


SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"

# Public constant for downstream scripts to use when constructing DataLoaders.
# Physicist Notes: Windows uses spawn instead of fork, which often triggers
# BrokenPipeError when multiple workers try to serialize large tensors.
SAFE_NUM_WORKERS = 0 if IS_WINDOWS else os.cpu_count() or 1


def _init_windows_colors() -> None:
    """Enable ANSI colors in Windows terminals."""
    try:
        import colorama

        colorama.just_fix_windows_console()
        return
    except Exception:
        pass

    # Fallback to Windows API if colorama is unavailable.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        # If enabling colors fails, continue silently.
        return


def _enable_cudnn_benchmark() -> None:
    """Turn on cuDNN autotune when CUDA is present (safe for fixed grid sizes)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        return


def setup_gpu() -> None:
    """
    Inspect JAX devices and report GPU/CPU selection.

    Prints a short status message; avoids raising if jax is missing or GPU is absent.
    """
    try:
        import jax

        devices = jax.devices()
        if not devices:
            print("Running on CPU")
            return
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            d = gpu_devices[0]
            print(f"JAX running on GPU: {getattr(d, 'device_kind', d)}")
        else:
            d = devices[0]
            print(f"Running on CPU: {getattr(d, 'device_kind', d)}")
    except Exception:
        print("Running on CPU")
        return


def as_path(path_like: Any) -> Path:
    """Convert an input to pathlib.Path (avoids backslash escaping issues on Windows)."""
    return path_like if isinstance(path_like, Path) else Path(path_like)


def recommended_num_workers(requested: Optional[int] = None) -> int:
    """
    Return a safe worker count for DataLoaders.

    On Windows: force 0 to avoid BrokenPipeError due to spawn semantics.
    Else: honor requested or default to SAFE_NUM_WORKERS.
    """
    if IS_WINDOWS:
        return 0
    if requested is None:
        return SAFE_NUM_WORKERS
    return max(0, int(requested))


def configure_runtime() -> None:
    """Apply all runtime tweaks."""
    if IS_WINDOWS:
        _init_windows_colors()
    _enable_cudnn_benchmark()


# Apply configuration on import so scripts only need to import the module.
configure_runtime()

__all__ = [
    "SYSTEM",
    "IS_WINDOWS",
    "SAFE_NUM_WORKERS",
    "as_path",
    "recommended_num_workers",
    "configure_runtime",
    "setup_gpu",
]
