"""Device selection helpers for demo inference."""

import torch


def get_available_device():
    """Return the best available torch device for local inference."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_model_to_available_device(model):
    """Move a model to the best available device and set demo precision."""
    device = get_available_device()
    model.to(device)

    if device == "cuda":
        model = model.bfloat16()
    else:
        model = model.float()

    return model, device
