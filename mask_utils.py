# mask_utils.py

def causal_mask(b, h, q, kv, device=None):
    return q >= kv