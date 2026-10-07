"""Build a TensorRT engine from an ONNX file written by export_onnx.py.

    python trt/export_trt.py exports/da3-giant-1.1_saddle_balanced_280x504_fp16.onnx
"""

from pathlib import Path

import tensorrt as trt
import tyro


def main(
    onnx_path: tyro.conf.Positional[Path],
    engine_path: Path | None = None,
    views: tuple[int, int, int] = (1, 16, 32),
):
    """
    Args:
        engine_path: Defaults to onnx_path with an .engine suffix.
        views: (min, opt, max) number of views of the optimization profile. max must not exceed
            the --max-views of the export.
    """
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TensorRT 11 networks are strongly typed: layer precisions come from the ONNX graph.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file also resolves the external .onnx.data weights.
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"Failed to parse {onnx_path}")
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        x = network.get_input(i)
        min_, opt, max_ = (tuple(v if d == -1 else d for d in x.shape) for v in views)
        profile.set_shape(x.name, min_, opt, max_)
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit(f"Failed to build an engine from {onnx_path}")
    engine_path.write_bytes(engine)
    print(f"Saved {engine_path}")


if __name__ == "__main__":
    tyro.cli(main)
