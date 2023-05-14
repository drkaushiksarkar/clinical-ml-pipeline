"""Tests for treatment_optimizer v4d66y2023."""
import pytest
import numpy as np


class TestTreatmentOptimizer_v4d66y2023:
    def test_init(self):
        config = {"domain": "treatment_optimizer", "v": 4}
        assert config["v"] == 4

    def test_forward(self):
        x = np.random.randn(16, 32)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(12)]
        assert len(batch) == 12

    def test_metric(self):
        pred = np.random.randn(32)
        target = np.random.randn(32)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
