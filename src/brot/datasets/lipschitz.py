"""
Synthetic benchmark with a known Lipschitz optimal transport map (Section 5.1).

Source:  P = N(0, I_d) truncated to the ball {||x|| <= R}.
Map:     T_0(x) = grad phi(x) = (a_s x_s + b_s x_s^3)_{s=1..d},
         phi(x) = (1/2) sum_s a_s x_s^2 + (1/4) sum_s b_s x_s^4.
Target:  Q = (T_0)_# P.

Since phi is strictly convex (a_s > 0, b_s >= 0), T_0 is the unique optimal
transport map from P to Q for the quadratic cost, and it is Lipschitz on the
ball with constant max_s (a_s + 3 b_s R^2). The paper uses d = 2,
(a_1, a_2) = (1.5, 0.8), (b_1, b_2) = (0.3, 0.1), and R = 3.5.
"""
from __future__ import annotations

from typing import Sequence

import torch


class LipschitzBenchmark:
    """Samplers and the ground-truth map of the Section 5.1 benchmark."""

    def __init__(self, a: Sequence[float] = (1.5, 0.8),
                 b: Sequence[float] = (0.3, 0.1), R: float = 3.5):
        assert len(a) == len(b), "a and b must have the same length"
        self.a = torch.tensor(a, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.R = float(R)
        self.d = len(a)

    @property
    def lipschitz_constant(self) -> float:
        return float((self.a + 3.0 * self.b * self.R ** 2).max())

    def sample_source(self, n: int, seed: int) -> torch.Tensor:
        """n draws from the truncated standard Gaussian (rejection sampling)."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.d, generator=g)
        norms = x.norm(dim=1)
        while (norms > self.R).any():
            mask = norms > self.R
            x[mask] = torch.randn(int(mask.sum()), self.d, generator=g)
            norms = x.norm(dim=1)
        return x

    def true_map(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a.to(x.device, x.dtype).view(1, -1)
        b = self.b.to(x.device, x.dtype).view(1, -1)
        return a * x + b * x ** 3

    def sample_target(self, m: int, seed: int) -> torch.Tensor:
        """m draws from Q = (T_0)_# P, independent of the source sample."""
        return self.true_map(self.sample_source(m, seed))
