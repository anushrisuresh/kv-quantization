# kv_compression.py

import torch

class KVCompressor:
    def __init__(self, enabled=False, window_size=None, sink_size=0):
        self.enabled = enabled
        self.window_size = window_size
        self.sink_size = sink_size

    def compress(self, k, v, input_pos):
        """
        Compress k, v based on attention sink and sliding window.

        Args:
            k: [batch, heads, seq_len, head_dim]
            v: [batch, heads, seq_len, head_dim]
            input_pos: [batch_size] tensor, current input positions

        Returns:
            compressed_k, compressed_v, mask_override
        """
        if not self.enabled:
            return k, v, None

        device = k.device
        max_pos = input_pos.max().item()

        keep_indices = []
        if max_pos < self.window_size:
            # Not enough tokens yet to compress
            return k, v, None
        if self.sink_size > 0:
            sink_indices = list(range(self.sink_size))  # always attend to the first sink_size tokens
            keep_indices.extend(sink_indices)

        if self.window_size is not None:
            start = max(max_pos - self.window_size + 1, 0)
            sliding_window_indices = list(range(start, max_pos + 1))
            keep_indices.extend(sliding_window_indices)

        keep_indices = sorted(set(keep_indices))

        # Compress k, v
        k = k[:, :, keep_indices, :]
        v = v[:, :, keep_indices, :]

        # Create mask override info
        mask_override = {
            "kv_positions": torch.tensor(keep_indices, device=device)
        }

        return k, v, mask_override