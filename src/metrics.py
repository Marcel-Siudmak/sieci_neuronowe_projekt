import torch

MAX_DEPTH_METERS = 10.0


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_depth: float = MAX_DEPTH_METERS,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Oblicza podstawowe metryki estymacji głębi (MAE, RMSE [m], AbsRel, delta1)."""
    with torch.no_grad():
        pred_m = pred.detach().float().clamp(min=eps) * max_depth
        target_m = target.detach().float() * max_depth

        mask = target_m > eps
        if not mask.any():
            return {"mae": 0.0, "rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}

        p = pred_m[mask]
        t = target_m[mask]
        diff = (p - t).abs()

        mae = diff.mean().item()
        rmse = torch.sqrt((diff**2).mean()).item()
        abs_rel = (diff / t).mean().item()
        delta1 = (torch.maximum(p / t, t / p) < 1.25).float().mean().item()

        return {
            "mae": mae,
            "rmse": rmse,
            "abs_rel": abs_rel,
            "delta1": delta1,
        }
