import time
from pathlib import Path

import tensorrt as trt
import torch

# TensorRT expects a single logger per process.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TORCH_DTYPES = {trt.float32: torch.float32, trt.int64: torch.int64}


class TRTEngine:
    """Runs a TensorRT engine with torch CUDA tensors as input and output buffers."""

    def __init__(self, path: Path) -> None:
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.stream = torch.cuda.Stream()

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # Engines from export_trt.py have float32 inputs.
        inputs = {name: x.float().contiguous() for name, x in inputs.items()}
        self.stream.wait_stream(torch.cuda.current_stream())
        outputs = {}
        with torch.cuda.stream(self.stream):
            # Output shapes are only known once every input shape is set.
            for name, tensor in inputs.items():
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())
            for name in self.names:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    shape = tuple(self.context.get_tensor_shape(name))
                    dtype = TORCH_DTYPES[self.engine.get_tensor_dtype(name)]
                    tensor = outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
                    self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


def latency_ms(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000
