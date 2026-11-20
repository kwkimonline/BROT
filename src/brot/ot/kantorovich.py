"""
Solve the Kantorovich (discrete OT) problem.

Given source samples X (n, d) and target samples Y (m, d) with uniform marginals,
returns the coupling matrix gamma of shape (n, m) satisfying:
    gamma.sum(dim=1) = 1/n
    gamma.sum(dim=0) = 1/m

Two solvers:
  - Exact EMD (reg=0): LP via POT.  O(n^3) → use batched_emd for large n.
  - Sinkhorn  (reg>0): fast approximation, good for n > 2000.

Batched EMD:
  Splits X into mini-batches, solves EMD per batch, rescales and stacks.
  Marginal identities hold exactly after rescaling:
      gamma_i_scaled = gamma_i * (n_i / n)
      gamma_i_scaled.sum(1) = 1/n  (source marginal per row)
      sum_i gamma_i_scaled.sum(0) = 1/m  (target marginal)
"""

import numpy as np
import torch
import ot
from typing import Literal, Optional

CostFn = Literal["sqeuclidean", "euclidean", "cosine"]


def pairwise_cost(
    X: torch.Tensor,
    Y: torch.Tensor,
    cost: CostFn = "sqeuclidean",
) -> np.ndarray:
    """Compute pairwise cost matrix C[i,j] = c(X_i, Y_j)."""
    X_np = X.detach().cpu().numpy().astype(np.float64)
    Y_np = Y.detach().cpu().numpy().astype(np.float64)
    if cost == "cosine":
        X_norm = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y_np / (np.linalg.norm(Y_np, axis=1, keepdims=True) + 1e-8)
        return 1.0 - X_norm @ Y_norm.T
    return ot.dist(X_np, Y_np, metric=cost)


def solve_kantorovich(
    X: torch.Tensor,
    Y: torch.Tensor,
    cost: CostFn = "sqeuclidean",
    reg: float = 0.0,
    reg_type: Literal["entropy", "kl"] = "entropy",
    numItermax: int = 1_000_000,
    stopThr: float = 1e-9,
    emd_batch_size: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Solve the Kantorovich OT problem and return the coupling matrix.

    Args:
        X:              Source samples (n, d).
        Y:              Target samples (m, d).
        cost:           Cost function. 'sqeuclidean' (default), 'euclidean', 'cosine'.
        reg:            Regularization strength. 0 → exact EMD; >0 → Sinkhorn.
        reg_type:       Sinkhorn regularization type ('entropy' or 'kl').
        numItermax:     Max iterations (EMD LP or Sinkhorn).
        stopThr:        Sinkhorn convergence threshold.
        emd_batch_size: If set and reg==0, use batched EMD (source split into
                        chunks of this size).  None = full EMD (may be slow).
        device:         Device for the returned tensor.

    Returns:
        gamma: (n, m) coupling matrix. gamma.sum(1) ≈ 1/n, gamma.sum(0) ≈ 1/m.
    """
    if reg == 0.0 and emd_batch_size is not None:
        return _batched_emd(X, Y, cost=cost, batch_size=emd_batch_size,
                            numItermax=numItermax, device=device)

    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    C = pairwise_cost(X, Y, cost)

    if reg == 0.0:
        gamma_np = ot.emd(a, b, C, numItermax=numItermax)
    else:
        # Normalize cost so that reg is scale-invariant.
        # Without this, reg=0.05 on costs ~500 gives exp(-500/0.05)≈0 → underflow.
        scale = C.max()
        if scale > 0:
            C = C / scale
            reg_scaled = reg / scale
        else:
            reg_scaled = reg
        reg_scaled = max(reg_scaled, 1e-8)   # safety floor (lowered for small-eps testing)

        if reg_type == "entropy":
            gamma_np = ot.sinkhorn(
                a, b, C, reg=reg_scaled, method="sinkhorn_log",
                numItermax=numItermax, stopThr=stopThr,
            )
        else:
            gamma_np = ot.sinkhorn(
                a, b, C, reg=reg_scaled, method="sinkhorn_log",
                numItermax=numItermax, stopThr=stopThr,
            )

    return torch.from_numpy(gamma_np.astype(np.float32)).to(device)


def _batched_emd(
    X: torch.Tensor,
    Y: torch.Tensor,
    cost: CostFn = "sqeuclidean",
    batch_size: int = 500,
    numItermax: int = 1_000_000,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Batched exact EMD: split X into chunks, solve EMD per chunk, rescale & stack.

    For chunk X_i of size n_i:
        gamma_i = emd(1/n_i, 1/m, C_i)        # (n_i, m), uniform marginals
        gamma_i_scaled = gamma_i * (n_i / n)   # rescale so rows sum to 1/n

    After stacking:
        gamma.sum(1) = 1/n  (exact)
        gamma.sum(0) = 1/m  (exact)

    This is an approximation of the full EMD (each batch sees all of Y but
    only a subset of X), and is much faster for large n.
    """
    n = X.shape[0]
    Y_np = Y.detach().cpu().numpy().astype(np.float64)
    b = np.ones(Y_np.shape[0], dtype=np.float64) / Y_np.shape[0]

    rows = []
    for start in range(0, n, batch_size):
        X_batch = X[start : start + batch_size]
        n_i = X_batch.shape[0]

        a_i = np.ones(n_i, dtype=np.float64) / n_i
        C_i = pairwise_cost(X_batch, Y, cost)
        gamma_i = ot.emd(a_i, b, C_i, numItermax=numItermax)  # (n_i, m)

        # Rescale: rows should sum to 1/n instead of 1/n_i
        gamma_i_scaled = gamma_i * (n_i / n)
        rows.append(gamma_i_scaled.astype(np.float32))

    gamma_np = np.concatenate(rows, axis=0)  # (n, m)
    return torch.from_numpy(gamma_np).to(device)
