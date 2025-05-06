# kv_quantization.py
import torch

class KVQuantizer:
    """
    Controls whether and how to quantize old KV-cache entries.
    """
    def __init__(
        self,
        enabled: bool = False,
        quant_type: str = "int8",
        sink_size: int = 0,
        window_size: int = 0,
    ):
        self.enabled = enabled
        self.quant_type = quant_type
        self.sink_size = sink_size
        self.window_size = window_size

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        A simple dynamic per-channel int8 quantization.  You can swap in
        any routine from quantize.py here as needed.
        """
        if self.quant_type == "int8":
            # symmetric per-channel fake quant:
            #  x: [..., channels, ...] → clamp to int8 range
            max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / 127.0
            qi = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
            return x_int8, scale
        else:
            raise NotImplementedError(f"Unsupported quant_type={self.quant_type}")
    
    def dequantize_tensor(self, x_int8: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16):
        """
        Dequantizes back to original dtype.
        Args:
            x_int8: [B, H, T, D] in int8
            scale: same shape as x_int8[..., D:1]
        """
        return (x_int8.to(torch.float32) * scale).to(dtype)