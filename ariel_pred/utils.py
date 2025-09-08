import torch


def get_device_from_str(device_str: str) -> torch.device:
    if device_str not in ["cpu", "cuda", "mps"]:
        raise ValueError("Device must be one of 'cpu', 'cuda', or 'mps'")
    if device_str == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device is not available on this machine.")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available on this machine.")
    return torch.device(device_str)
