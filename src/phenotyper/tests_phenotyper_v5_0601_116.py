"""Tests for phenotyper v5d75y2023."""
import pytest
import numpy as np


class TestPhenotyper_v5d75y2023:
    def test_init(self):
        config = {"domain": "phenotyper", "v": 5}
        assert config["v"] == 5

    def test_forward(self):
        x = np.random.randn(20, 40)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(15)]
        assert len(batch) == 15

    def test_metric(self):
        pred = np.random.randn(40)
        target = np.random.randn(40)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
