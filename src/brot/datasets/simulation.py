"""
Synthetic distributions for simulation experiments.

Case 1 (Gaussian → Gaussian):
    P = N(mu_s, Sigma_s)  with mu_s=(15,15), Sigma_s=diag(3,3)
    Q = N(mu_t, Sigma_t)  with mu_t=(-6,4),  Sigma_t=[[1,0.5],[0.5,5]]

Case 2 (Gaussian → Uniform rectangle):
    P = N(mu_s, Sigma_s)  (same as above)
    Q = Unif([-15,-14] x [-2,2])

Case 3 (Gaussian → Uniform ellipse):
    P = N(mu_s, Sigma_s)  with mu_s=(15,15), Sigma_s=diag(3,3)
    Q = Unif(E)  where E = {y : (y-c)^T (R_θ diag(a²,b²) R_θ^T)^{-1} (y-c) ≤ 1}
    c=(-15,0), a=3, b=1, θ=π/6

The true OT map for Case 1 (Gaussian to Gaussian) is known analytically:
    T_0(x) = mu_t + A (x - mu_s)
where A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}.
"""

import torch
import numpy as np
from typing import Optional
import scipy.linalg


class GaussianSource:
    """Samples from N(mu, Sigma)."""

    def __init__(
        self,
        mu: list = [15.0, 15.0],
        sigma: list = [[3.0, 0.0], [0.0, 3.0]],
        seed: Optional[int] = None,
    ):
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> torch.Tensor:
        X = self.rng.multivariate_normal(self.mu, self.sigma, size=n).astype(np.float32)
        return torch.from_numpy(X)


class GaussianTarget:
    """Samples from N(mu, Sigma)."""

    def __init__(
        self,
        mu: list = [-6.0, 4.0],
        sigma: list = [[1.0, 0.5], [0.5, 5.0]],
        seed: Optional[int] = None,
    ):
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> torch.Tensor:
        X = self.rng.multivariate_normal(self.mu, self.sigma, size=n).astype(np.float32)
        return torch.from_numpy(X)


class UniformTarget:
    """Samples from Unif([a,b] x [c,d])."""

    def __init__(
        self,
        low: list = [-15.0, -2.0],
        high: list = [-14.0, 2.0],
        seed: Optional[int] = None,
    ):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> torch.Tensor:
        X = self.rng.uniform(self.low, self.high, size=(n, len(self.low))).astype(np.float32)
        return torch.from_numpy(X)


class UniformEllipseTarget:
    """Samples uniformly from the ellipse E = {y : (y-c)^T M^{-1} (y-c) ≤ 1}
    where M = R_θ diag(a², b²) R_θ^T.

    Sampling via affine transform of unit disk:
        z ~ Unif(unit disk)
        y = c + R_θ @ diag(a, b) @ z

    Case 3: c=(-15,0), a=3, b=1, θ=π/6
    """

    def __init__(
        self,
        center: list = [-15.0, 0.0],
        a: float = 3.0,
        b: float = 1.0,
        theta: float = np.pi / 6,
        seed: Optional[int] = None,
    ):
        self.center = np.array(center, dtype=np.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        # Linear map: z (unit disk) -> R @ diag(a,b) @ z
        self.L = (R @ np.diag([a, b])).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> torch.Tensor:
        # Sample uniformly from unit disk: r = sqrt(U), phi ~ Unif(0, 2pi)
        r = np.sqrt(self.rng.uniform(0.0, 1.0, size=n)).astype(np.float32)
        phi = self.rng.uniform(0.0, 2 * np.pi, size=n).astype(np.float32)
        z = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)  # (n, 2)
        X = z @ self.L.T + self.center                             # (n, 2)
        return torch.from_numpy(X)


def gaussian_ot_map(
    mu_s: np.ndarray,
    sigma_s: np.ndarray,
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
) -> callable:
    """
    Closed-form OT map from N(mu_s, Sigma_s) to N(mu_t, Sigma_t).

        T_0(x) = mu_t + A @ (x - mu_s)
        A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}

    Returns a callable that maps (n, d) tensors.
    """
    sqrt_s = scipy.linalg.sqrtm(sigma_s).real
    inv_sqrt_s = np.linalg.inv(sqrt_s)
    M = sqrt_s @ sigma_t @ sqrt_s
    A = inv_sqrt_s @ scipy.linalg.sqrtm(M).real @ inv_sqrt_s

    mu_s_t = torch.from_numpy(mu_s.astype(np.float32))
    mu_t_t = torch.from_numpy(mu_t.astype(np.float32))
    A_t = torch.from_numpy(A.astype(np.float32))

    def _map(X: torch.Tensor) -> torch.Tensor:
        return mu_t_t + (X - mu_s_t) @ A_t.T

    return _map
