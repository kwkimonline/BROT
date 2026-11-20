"""
Evaluation metrics for OT map estimation.

  transport_cost : (1/n) sum_i c(X_i, T(X_i))
  wasserstein    : W_2 distance between two empirical distributions
  mmd            : Maximum Mean Discrepancy with Gaussian kernel
"""

import numpy as np
import torch
import ot
from typing import Literal


def transport_cost(
    X: torch.Tensor,
    T_X: torch.Tensor,
    cost: Literal["sqeuclidean", "euclidean"] = "sqeuclidean",
) -> float:
    """
    Empirical transport cost of the map T evaluated at source samples X.

        TC = (1/n) sum_i ||X_i - T(X_i)||^2    (sqeuclidean)
           = (1/n) sum_i ||X_i - T(X_i)||       (euclidean)
    """
    X_np = X.detach().cpu().numpy()
    T_X_np = T_X.detach().cpu().numpy()
    diff = X_np - T_X_np
    if cost == "sqeuclidean":
        return float(np.mean(np.sum(diff ** 2, axis=1)))
    else:
        return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))


def wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    p: int = 2,
    reg: float = 0.0,
) -> float:
    """
    Empirical W_p distance between P_X and Q_Y.

    If reg <= 0, computes exact EMD (LP).
    If reg > 0, computes Sinkhorn approximation.

    Args:
        X: (n, d) samples from distribution P.
        Y: (m, d) samples from distribution Q.
        p: Wasserstein order (2 by default).
        reg: Entropic regularization for Sinkhorn. Use 0 for exact EMD.

    Returns:
        W_p(P_X, Q_Y).
    """
    X_np = X.detach().cpu().numpy().astype(np.float64)
    Y_np = Y.detach().cpu().numpy().astype(np.float64)
    n, m = len(X_np), len(Y_np)
    a = np.ones(n) / n
    b = np.ones(m) / m

    def _sinkhorn2_normalized(C_raw):
        # Normalize cost to avoid exp(-large/small_reg) underflow.
        # Sinkhorn is scale-invariant: W_p(P,Q;C) = scale * W_p(P,Q;C/scale)
        scale = C_raw.max()
        if scale < 1e-10:
            return 0.0
        C_norm = C_raw / scale
        reg_norm = max(reg / scale, 1e-3)   # keep reg meaningful after scaling
        result = ot.sinkhorn2(a, b, C_norm, reg=reg_norm, numItermax=10000)
        val = float(result[0]) if hasattr(result, '__len__') else float(result)
        return val * scale  # rescale back

    if p == 2:
        C = ot.dist(X_np, Y_np, metric="sqeuclidean")
    else:
        C = ot.dist(X_np, Y_np, metric="euclidean") ** p

    if reg <= 0.0:
        loss = float(ot.emd2(a, b, C))
    else:
        loss = _sinkhorn2_normalized(C)

    if p == 2:
        return float(np.sqrt(max(loss, 0.0)))
    return float(max(loss, 0.0) ** (1.0 / p))


def mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    bandwidth: float = 1.0,
) -> float:
    """
    Maximum Mean Discrepancy with Gaussian kernel of given bandwidth.

        MMD^2(P, Q) = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
        k(x, y) = exp(-||x-y||^2 / (2 * bandwidth^2))

    Args:
        X: (n, d) samples from P.
        Y: (m, d) samples from Q.
        bandwidth: Gaussian kernel bandwidth (1.0 as in the paper).

    Returns:
        MMD value (non-negative).
    """
    X = X.detach().cpu().float()
    Y = Y.detach().cpu().float()
    n, m = len(X), len(Y)

    def _rbf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Pairwise squared distances
        diff = A.unsqueeze(1) - B.unsqueeze(0)      # (n, m, d)
        sq_dist = diff.pow(2).sum(-1)               # (n, m)
        return torch.exp(-sq_dist / (2.0 * bandwidth ** 2))

    K_XX = _rbf(X, X).mean()
    K_YY = _rbf(Y, Y).mean()
    K_XY = _rbf(X, Y).mean()
    return float((K_XX - 2.0 * K_XY + K_YY).clamp(min=0.0).sqrt())
