"""Export an any-view Depth Anything 3 model (depth + camera pose) to ONNX for TensorRT.

    uv pip install -e ".[export]"
    python trt/export_onnx.py --precision fp16 --size 280 504
    python trt/export_onnx.py --precision fp16 --size 280 504 --with-poses

The image size is static and the number of views is dynamic: build one engine per image size.
Without poses, the reference-view strategy is baked into the graph.
"""

import types
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch import nn

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from depth_anything_3.model.utils import transform
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
from depth_anything_3.utils.geometry import affine_inverse

Precision = Literal["bf16", "fp16", "fp32"]
RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# ONNX forbids a graph input and output with the same name.
INPUT_NAMES = ["image", "input_extrinsics", "input_intrinsics"]
OUTPUT_NAMES = ["depth", "depth_conf", "extrinsics", "intrinsics"]


def export_name(
    model_id: str,
    size: tuple[int, int],
    precision: Precision,
    with_poses: bool,
    ref_view_strategy: RefViewStrategy,
    half_head: bool = True,
) -> str:
    H, W = size
    variant = "poses" if with_poses else ref_view_strategy
    suffix = "" if half_head else "_fp32head"
    return f"{model_id.split('/')[-1].lower()}_{variant}_{H}x{W}_{precision}{suffix}"


def load_model(model_id: str) -> DepthAnything3Net:
    return DepthAnything3.from_pretrained(model_id).model.cuda().eval()


def _select_reference_view(x: torch.Tensor, strategy: RefViewStrategy | int) -> torch.Tensor:
    """select_reference_view without its `S <= 1` early return. An int strategy forces that view."""
    B, S = x.shape[:2]
    if isinstance(strategy, int):
        return torch.full((B,), strategy, dtype=torch.long, device=x.device)
    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    if strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)
    cls = x[:, :, 0]
    feat = cls / cls.norm(dim=-1, keepdim=True)
    sim = torch.matmul(feat, feat.transpose(1, 2)) - torch.eye(S, device=x.device).unsqueeze(0)
    if strategy == "saddle_sim_range":
        return (sim.max(dim=-1).values - sim.min(dim=-1).values).argmax(dim=1)

    def normalize(metric):
        min_val = metric.min(dim=1, keepdim=True).values
        max_val = metric.max(dim=1, keepdim=True).values
        return (metric - min_val) / (max_val - min_val + 1e-8)

    balance = (
        (normalize(sim.sum(dim=-1) / (S - 1)) - 0.5).abs()
        + (normalize(cls.norm(dim=-1)) - 0.5).abs()
        + (normalize(feat.var(dim=-1)) - 0.5).abs()
    )
    return balance.argmin(dim=1)


def _reorder_by_reference(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """reorder_by_reference without its `S <= 1` early return."""
    view = torch.arange(x.shape[1], device=x.device)[None]
    b = b_idx[:, None]
    idx = torch.where(view == 0, b, torch.where(view <= b, view - 1, view))
    return x[torch.arange(x.shape[0], device=x.device)[:, None], idx]


def _restore_original_order(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """restore_original_order without its `S <= 1` early return."""
    view = torch.arange(x.shape[1], device=x.device)[None]
    b = b_idx[:, None]
    idx = torch.where(view == b, 0, torch.where(view < b, view + 1, view))
    return x[torch.arange(x.shape[0], device=x.device)[:, None], idx]


def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
    """DinoVisionTransformer._get_intermediate_layers_not_chunked with the reference-view reordering
    selected by a tensor instead of a Python branch on the (dynamic) number of views."""
    B, S, _, H, W = x.shape
    x = self.prepare_tokens_with_masks(x)
    local_x = x
    output = []
    blocks_to_take = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
    pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
    cam_token = kwargs.get("cam_token")
    view = torch.arange(S, device=x.device)
    b_idx = None

    for i, blk in enumerate(self.blocks):
        if i < self.rope_start or self.rope is None:
            g_pos, l_pos = None, None
        else:
            g_pos, l_pos = pos_nodiff, pos

        if i == self.alt_start - 1 and cam_token is None:
            b_idx = _select_reference_view(x, kwargs.get("ref_view_strategy", "saddle_balanced"))
            # The last view index reaches THRESH - 1 exactly when PyTorch would reorder.
            b_idx = torch.where(view[-1] >= THRESH_FOR_REF_SELECTION - 1, b_idx, 0)
            # Read back by DA3Export, which returns it as an output.
            self.export_reference_view = b_idx
            x = _reorder_by_reference(x, b_idx)
            local_x = _reorder_by_reference(local_x, b_idx)

        if i == self.alt_start:
            if cam_token is None:
                ref_token, src_token = self.camera_token[:, :1], self.camera_token[:, 1:]
                cam_token = torch.where((view == 0)[None, :, None], ref_token, src_token)
                cam_token = cam_token.expand(B, -1, -1)
            x[:, :, 0] = cam_token

        if i >= self.alt_start and i % 2 == 1:
            x = self.process_attention(
                x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
            )
        else:
            x = self.process_attention(x, blk, "local", pos=l_pos)
            local_x = x

        if i in blocks_to_take:
            out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if b_idx is not None:
                out_x = _restore_original_order(out_x, b_idx)
            output.append((out_x[:, :, 0], out_x))
    return output, []


def _rope_forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """RotaryPositionEmbedding2D.forward with tables precomputed for the static image size."""
    assert tokens.dtype == self.export_cos.dtype, (tokens.dtype, self.export_cos.dtype)
    vertical, horizontal = tokens.chunk(2, dim=-1)
    vertical = self._apply_1d_rope(vertical, positions[..., 0], self.export_cos, self.export_sin)
    horizontal = self._apply_1d_rope(
        horizontal, positions[..., 1], self.export_cos, self.export_sin
    )
    return torch.cat((vertical, horizontal), dim=-1)


def _mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """transform.mat_to_quat, gathering the best candidate instead of boolean-mask indexing it."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = transform._sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    best = q_abs.argmax(dim=-1)[..., None, None].expand(batch_dim + (1, 4))
    out = quat_candidates.gather(-2, best).squeeze(-2)
    out = out[..., [1, 2, 3, 0]]
    return transform.standardize_quaternion(out)


def _cam_enc_forward(self, ext, ixt, image_size):
    """CameraEnc.forward, with extri_intri_to_pose_encoding inlined to use _mat_to_quat."""
    c2ws = affine_inverse(ext)
    H, W = image_size
    fov_h = 2 * torch.atan((H / 2) / ixt[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / ixt[..., 0, 0])
    pose_encoding = torch.cat(
        [c2ws[:, :, :3, 3], _mat_to_quat(c2ws[:, :, :3, :3]), fov_h[..., None], fov_w[..., None]],
        dim=-1,
    ).float()
    pose_tokens = self.pose_branch(pose_encoding)
    pose_tokens = self.token_norm(pose_tokens)
    pose_tokens = self.trunk(pose_tokens)
    return self.trunk_norm(pose_tokens)


def make_exportable(net: DepthAnything3Net, size: tuple[int, int]) -> None:
    """Patch `net` in place so that torch.export can trace it at a static image size and a dynamic
    number of views. Outputs are unchanged up to floating-point reordering."""
    H, W = size
    vit = net.backbone.pretrained
    assert vit.alt_start != -1, "only any-view models (with alternating attention) are supported"
    patch = vit.patch_size
    device = vit.pos_embed.device

    vit._get_intermediate_layers_not_chunked = types.MethodType(
        _get_intermediate_layers_not_chunked, vit
    )

    # Constant at a static size; precomputing it avoids relying on ONNX/TensorRT cubic Resize
    # matching PyTorch's offset scale factors.
    with torch.no_grad():
        pos_embed = vit.interpolate_pos_encoding(
            torch.empty(1, (H // patch) * (W // patch) + 1, vit.embed_dim, device=device), H, W
        )
    vit.register_buffer("export_pos_embed", pos_embed, persistent=False)

    def interpolate_pos_encoding(x, h, w):
        assert (h, w) == (H, W) and x.dtype == pos_embed.dtype, (h, w, x.dtype)
        return vit.export_pos_embed

    vit.interpolate_pos_encoding = interpolate_pos_encoding

    # PositionGetter caches positions per grid; fill it outside the trace so the cache holds a
    # real tensor rather than a traced one.
    vit.position_getter(1, H // patch, W // patch, device=device)
    # RoPE sizes its tables with `int(positions.max())`, which is data-dependent. Table rows do
    # not depend on the table length, and positions run from 0 (cls token) to max(h, w).
    rope = vit.rope
    cos, sin = rope._compute_frequency_components(
        vit.embed_dim // vit.num_heads // 2,
        max(H, W) // patch + 1,
        device,
        torch.float32,
    )
    rope.register_buffer("export_cos", cos, persistent=False)
    rope.register_buffer("export_sin", sin, persistent=False)
    rope.forward = types.MethodType(_rope_forward, rope)

    # With xformers installed, SwiGLUFFNFused dispatches to xformers ops; SwiGLUFFN computes the
    # same in plain PyTorch over the same w12/w3 weights.
    for m in net.modules():
        if isinstance(m, SwiGLUFFNFused):
            m.forward = types.MethodType(SwiGLUFFN.forward, m)

    if net.cam_enc is not None:
        net.cam_enc.forward = types.MethodType(_cam_enc_forward, net.cam_enc)


class DA3Export(nn.Module):
    """The depth and camera-decoder outputs of DepthAnything3Net.forward, under the autocast that
    DepthAnything3.forward applies, followed without poses by `reference_view`: the index of the
    view whose camera frame the extrinsics are expressed in. Call make_exportable on `net` first.
    With half_head, the DPT head runs under that autocast too, whereas PyTorch runs it in FP32."""

    def __init__(
        self,
        net: DepthAnything3Net,
        precision: Precision,
        ref_view_strategy: RefViewStrategy | int,
        half_head: bool = True,
    ) -> None:
        super().__init__()
        self.net = net
        self.precision = precision
        self.ref_view_strategy = ref_view_strategy
        self.half_head = half_head

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        net = self.net
        H, W = image.shape[-2:]
        with torch.autocast(
            "cuda", dtype=DTYPES[self.precision], enabled=self.precision != "fp32"
        ):
            cam_token = None
            if extrinsics is not None:
                with torch.autocast("cuda", enabled=False):
                    cam_token = net.cam_enc(extrinsics, intrinsics, (H, W))
            feats, _ = net.backbone(
                image, cam_token=cam_token, ref_view_strategy=self.ref_view_strategy
            )
            with nullcontext() if self.half_head else torch.autocast("cuda", enabled=False):
                # chunk_size=None: the default chunking is a Python loop over the views.
                output = net.head(feats, H, W, patch_start_idx=0, chunk_size=None)
            with torch.autocast("cuda", enabled=False):
                for name in output:
                    output[name] = output[name].float()
                output = net._process_camera_estimation(feats, H, W, output)
        outputs = tuple(output[name] for name in OUTPUT_NAMES)
        if extrinsics is None:
            outputs += (net.backbone.pretrained.__dict__.pop("export_reference_view"),)
        return outputs


def example_inputs(
    num_views: int, size: tuple[int, int], with_poses: bool
) -> tuple[torch.Tensor, ...]:
    H, W = size
    image = torch.randn(1, num_views, 3, H, W, device="cuda")
    if not with_poses:
        return (image,)
    extrinsics = torch.eye(4, device="cuda").repeat(1, num_views, 1, 1)
    extrinsics[..., 0, 3] = torch.linspace(0, 1, num_views, device="cuda")
    intrinsics = torch.tensor(
        [[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=torch.float32, device="cuda"
    ).repeat(1, num_views, 1, 1)
    return image, extrinsics, intrinsics


def main(
    model_id: str = "depth-anything/DA3-GIANT-1.1",
    precision: Precision = "fp16",
    size: tuple[int, int] = (280, 504),
    with_poses: bool = False,
    ref_view_strategy: RefViewStrategy = "saddle_balanced",
    max_views: int = 32,
    half_head: bool = True,
    out_dir: Path = Path("exports"),
):
    """
    Args:
        size: (H, W) of the input images, multiples of 14.
        with_poses: Add `input_extrinsics` (w2c, 4x4) and `input_intrinsics` (3x3) inputs. The
            extrinsics must be normalized as DepthAnything3._normalize_extrinsics does.
        ref_view_strategy: Baked into the graph without poses; unused with poses.
        half_head: Run the DPT head at `precision` rather than in FP32 like PyTorch, which is
            about 10% faster with FP16 at 16 views. --no-half-head names the export *_fp32head.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = export_name(model_id, size, precision, with_poses, ref_view_strategy, half_head)
    path = out_dir / f"{name}.onnx"
    net = load_model(model_id)
    make_exportable(net, size)
    model = DA3Export(net, precision, ref_view_strategy, half_head).eval()

    # torch.export specializes dimensions of size 0 and 1, and the reference-view selection only
    # runs from 3 views on.
    inputs = example_inputs(4, size, with_poses)
    views = torch.export.Dim("views", min=1, max=max_views)
    # optimize=True rewrites chunk/unbind from SplitToSequence + SequenceAt, which TensorRT cannot
    # parse, into Split.
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            path,
            dynamo=True,
            opset_version=18,
            optimize=True,
            input_names=INPUT_NAMES[: len(inputs)],
            output_names=OUTPUT_NAMES + ([] if with_poses else ["reference_view"]),
            dynamic_shapes=tuple({1: views} for _ in inputs),
        )
    print(f"Saved {path}")


if __name__ == "__main__":
    tyro.cli(main)
