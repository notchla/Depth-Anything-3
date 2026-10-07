"""Time a TensorRT engine from export_trt.py over numbers of views.

    python trt/benchmark_trt.py exports/da3-giant-1.1_saddle_balanced_280x504_fp16.engine
"""

from functools import partial
from pathlib import Path

import tensorrt as trt
import torch
import tyro

from common import TRTEngine, latency_ms
from export_onnx import INPUT_NAMES, example_inputs


def main(
    engine_path: tyro.conf.Positional[Path],
    views: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    iters: int = 20,
):
    """
    Args:
        views: Numbers of views to time, within the engine's optimization profile.
    """
    engine = TRTEngine(engine_path)
    shape = engine.engine.get_tensor_shape("image")
    size = (shape[-2], shape[-1])
    with_poses = INPUT_NAMES[1] in engine.names
    _, _, max_views = engine.engine.get_tensor_profile_shape("image", 0)
    print(
        f"{engine_path.name}: {size[0]}x{size[1]}, TensorRT {trt.__version__}, "
        f"{torch.cuda.get_device_name()}, activation memory at {max_views[1]} views "
        f"{engine.engine.device_memory_size_v2 / 2**30:.2f} GiB"
    )
    print(f"{'views':>5} {'ms':>8} {'ms/view':>8} {'views/s':>8}")
    for S in views:
        inputs = dict(zip(INPUT_NAMES, example_inputs(S, size, with_poses)))
        ms = latency_ms(partial(engine, **inputs), iters)
        print(f"{S:>5} {ms:>8.1f} {ms / S:>8.2f} {S / ms * 1000:>8.1f}")


if __name__ == "__main__":
    tyro.cli(main)
