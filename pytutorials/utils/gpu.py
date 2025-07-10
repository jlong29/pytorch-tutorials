import torch

def get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_model(model, device=None, dtype=None, compile_model=False):
    """
    Moves model to specified device, optionally casts dtype and compiles.

    Args:
        model (nn.Module): PyTorch model.
        device (torch.device or None): Target device (default: autodetect).
        dtype (torch.dtype or None): torch.float16 or torch.bfloat16 for mixed precision.
        compile_model (bool): If True and torch >= 2.0, compile the model.
    
    Returns:
        model (nn.Module): Optimized model.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    if dtype is not None:
        model = model.to(dtype)

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model
