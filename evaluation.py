import torch
import numpy as np

def directional_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_flat = pred.detach().cpu().numpy().flatten()
    true_flat = true.detach().cpu().numpy().flatten()
    mask = true_flat != 0.0
    if np.sum(mask) == 0:
        return 0.0
    correct = np.sum((pred_flat[mask] * true_flat[mask]) > 0)
    return correct / np.sum(mask)

def mean_squared_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((pred - true) ** 2).item()

def mean_absolute_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - true)).item()
