import imp


import torch


def zero_nan_hook(g: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(g), torch.zeros_like(g), g)
