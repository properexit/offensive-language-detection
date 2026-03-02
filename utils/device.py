import torch


def get_device(prefer_gpu=True, force_device=None):
    """
    Pick a device automatically.
    """

    # if user explicitly forces one, use that
    if force_device is not None:
        return torch.device(force_device)

    # try CUDA first
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    # then Apple MPS
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # fallback
    return torch.device("cpu")