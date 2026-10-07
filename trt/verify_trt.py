"""Compare a TensorRT engine from export_trt.py with the PyTorch model on real frames.

    python trt/verify_trt.py --num-views 8
    python trt/verify_trt.py --num-views 8 --with-poses

With --with-poses, the poses fed to both models are the ones PyTorch predicts without poses.
"""

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro

from common import TRTEngine, latency_ms
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.utils.io.input_processor import InputProcessor
from export_onnx import (
    DTYPES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    DA3Export,
    Precision,
    RefViewStrategy,
    export_name,
    load_model,
    make_exportable,
)


def read_frames(video: Path, num_views: int) -> list[np.ndarray]:
    # Decoded sequentially: CAP_PROP_FRAME_COUNT can overcount, and seeking past the end fails.
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return [frames[i] for i in np.linspace(0, len(frames) - 1, num_views).round().astype(int)]


def rotation_error_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cos = ((a[..., :3, :3] @ b[..., :3, :3].mT).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def compare(outputs: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict:
    """Errors of `outputs` relative to `reference`, both with the DA3Export output names."""
    errors = {}
    for name in ("depth", "depth_conf"):
        rel = ((outputs[name] - reference[name]).abs() / reference[name].abs()).flatten()
        errors[f"{name} rel mean"] = rel.mean().item()
        errors[f"{name} rel p99"] = rel.quantile(0.99).item()
    ext, ext_ref = outputs["extrinsics"], reference["extrinsics"]
    errors["rotation max deg"] = rotation_error_deg(ext, ext_ref).max().item()
    # Relative to the scene depth rather than to the translations, which are ~0 for small baselines.
    t_err = (ext[..., :3, 3] - ext_ref[..., :3, 3]).norm(dim=-1).max()
    errors["translation max / depth"] = (t_err / reference["depth"].median()).item()
    focal = outputs["intrinsics"][..., [0, 1], [0, 1]]
    focal_ref = reference["intrinsics"][..., [0, 1], [0, 1]]
    errors["focal max rel"] = ((focal - focal_ref).abs() / focal_ref).max().item()
    return errors


def main(
    model_id: str = "depth-anything/DA3-GIANT-1.1",
    precision: Precision = "fp16",
    size: tuple[int, int] = (280, 504),
    with_poses: bool = False,
    ref_view_strategy: RefViewStrategy = "saddle_balanced",
    num_views: int = 8,
    video: Path = Path("assets/examples/robot_unitree.mp4"),
    engine_dir: Path = Path("exports"),
    depth_tol: float = 0.02,
    rotation_tol_deg: float = 1.0,
    focal_tol: float = 0.02,
):
    """
    Args:
        size: (H, W) the engine was exported with. Frames are resized to it as
            DepthAnything3.inference does, and must have a matching aspect ratio.
        depth_tol: Tolerance on the mean relative depth error; focal_tol on the max relative
            focal error. Both, and rotation_tol_deg, compare TensorRT with PyTorch.
    """
    images, _, _ = InputProcessor()(read_frames(video, num_views), process_res=max(size))
    assert tuple(images.shape[-2:]) == size, (
        f"frames preprocess to {tuple(images.shape[-2:])}, but the engine expects {size}"
    )
    image = images[None].cuda()
    net = load_model(model_id)
    autocast = torch.autocast("cuda", dtype=DTYPES[precision], enabled=precision != "fp32")

    @torch.no_grad()
    def run_pytorch(*inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        with autocast:
            output = net(*inputs, export_feat_layers=[], ref_view_strategy=ref_view_strategy)
        return {name: output[name].float() for name in OUTPUT_NAMES}

    inputs = (image,)
    if with_poses:
        predicted = run_pytorch(image)
        extrinsics = DepthAnything3._normalize_extrinsics(
            None, as_homogeneous(predicted["extrinsics"])
        )
        inputs = (image, extrinsics, predicted["intrinsics"])
    reference = run_pytorch(*inputs)
    pytorch_ms = latency_ms(partial(run_pytorch, *inputs))

    # Patching happens after the reference so that it runs the unmodified model.
    make_exportable(net, size)

    @torch.no_grad()
    def run_patched(strategy: RefViewStrategy | int) -> dict[str, torch.Tensor]:
        outputs = DA3Export(net, precision, strategy).eval()(*inputs)
        return dict(zip(OUTPUT_NAMES + ["reference_view"], outputs))

    patched = run_patched(ref_view_strategy)
    patched_diff = max(
        (patched[name] - reference[name]).abs().max().item() for name in OUTPUT_NAMES
    )

    name = export_name(model_id, size, precision, with_poses, ref_view_strategy)
    engine_path = engine_dir / f"{name}.engine"
    # Memory cached by the PyTorch runs can leave too little for the engine's context.
    torch.cuda.empty_cache()
    engine = TRTEngine(engine_path)
    feeds = dict(zip(INPUT_NAMES, inputs))
    outputs = engine(**feeds)
    trt_ms = latency_ms(partial(engine, **feeds))

    print(f"{engine_path.name}, {num_views} views of {video.name}")
    print(f"patched PyTorch vs PyTorch: max abs diff {patched_diff:.2e}")
    if not with_poses:
        pytorch_view, trt_view = patched["reference_view"].item(), outputs["reference_view"].item()
        print(f"reference view: PyTorch {pytorch_view}, TensorRT {trt_view}")
        if trt_view != pytorch_view:
            # The saddle strategies rank views by min-max normalized scores whose spread can be
            # within float32 rounding, so any change in reduction order can flip the choice.
            print("  differs: comparing with PyTorch forced to the TensorRT reference view")
            reference = run_patched(trt_view)
    errors = compare(outputs, reference)
    for name, value in errors.items():
        print(f"TensorRT vs PyTorch: {name:<24} {value:.2e}")
    print(
        f"latency: PyTorch {pytorch_ms:.1f} ms, TensorRT {trt_ms:.1f} ms "
        f"({pytorch_ms / trt_ms:.2f}x)"
    )
    failed = [
        name
        for name, value, tol in (
            ("depth rel mean", errors["depth rel mean"], depth_tol),
            ("rotation max deg", errors["rotation max deg"], rotation_tol_deg),
            ("focal max rel", errors["focal max rel"], focal_tol),
        )
        if not value <= tol
    ]
    print("FAIL: " + ", ".join(failed) if failed else "PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    tyro.cli(main)
