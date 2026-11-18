import math

import pytest
import torch

from src.metrics import compute_metrics


def test_compute_metrics_identical_inputs():
    target = torch.rand(2, 1, 32, 32) * 0.9 + 0.1
    metrics = compute_metrics(target, target)

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["delta1"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_known_values():
    target = torch.full((1, 1, 16, 16), 0.5)
    pred = torch.full((1, 1, 16, 16), 0.6)
    metrics = compute_metrics(pred, target)

    assert metrics["mae"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["rmse"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["abs_rel"] == pytest.approx(0.2, abs=1e-5)
    assert metrics["delta1"] == pytest.approx(1.0, abs=1e-5)


def test_compute_metrics_delta1_threshold():
    target = torch.full((1, 1, 2, 2), 0.4)
    pred = torch.tensor([[[[0.44, 0.44], [0.60, 0.60]]]])  # ratios: 1.1 (<1.25) and 1.5 (>=1.25)
    metrics = compute_metrics(pred, target)

    assert metrics["delta1"] == pytest.approx(0.5, abs=1e-5)


def test_compute_metrics_ignores_zero_targets():
    target = torch.tensor([[[[0.0, 0.5], [0.0, 0.5]]]])
    pred = torch.tensor([[[[0.9, 0.6], [0.2, 0.6]]]])
    metrics = compute_metrics(pred, target)

    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["mae"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["rmse"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["abs_rel"] == pytest.approx(0.2, abs=1e-5)
    assert metrics["delta1"] == pytest.approx(1.0, abs=1e-5)

    zeros = torch.zeros(1, 1, 4, 4)
    zero_metrics = compute_metrics(pred, zeros)
    assert zero_metrics == {"mae": 0.0, "rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
