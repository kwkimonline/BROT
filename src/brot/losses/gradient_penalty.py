"""
Gradient penalty for Lipschitz regularization of the transport map.

The paper constrains the transport map T to be lambda-Lipschitz via:

    alpha * (1/n) * sum_i || J_T(X_i) ||_F^2

where J_T(X_i) is the Jacobian of T at X_i (a matrix of shape [out_dim, in_dim]).

For efficiency, we use the Hutchinson trace estimator:
    E_v[ ||J_T(x)^T v||^2 ] = ||J_T(x)||_F^2     (v ~ N(0, I))
which requires only one backward pass regardless of out_dim.

IMPORTANT: To avoid double-backward issues, always compute MSE and GP from the
same forward pass.  Use `brot_loss` which handles this internally, or pass
T_X (pre-computed) to `gradient_penalty` along with the matching X_gp leaf.
"""

import torch
import torch.nn as nn
from typing import Literal, Dict, Optional


def gradient_penalty(
    model: nn.Module,
    X: torch.Tensor,
    T_X: Optional[torch.Tensor] = None,
    hutchinson: bool = True,
) -> torch.Tensor:
    """
    Compute (1/B) * sum_i || J_T(X_i) ||_F^2.

    Two modes:
      1. T_X is None  → fresh forward pass.  X must NOT have requires_grad (it
         will be detached internally and a new leaf created).
      2. T_X provided → reuse the pre-computed output.  X must ALREADY be the
         requires_grad=True leaf used for that forward pass (i.e., X_gp from
         brot_loss).  This avoids a second forward pass and the associated
         double-backward error.

    Args:
        model:      The transport map T (only needed when T_X is None).
        X:          Input batch (B, d).
        T_X:        Pre-computed T(X_gp) (B, p), or None.
        hutchinson: Use Hutchinson estimator (1 backward pass).

    Returns:
        Scalar gradient penalty term (differentiable w.r.t. model parameters).
    """
    if T_X is None:
        # Fresh forward pass – detach X so X_gp is a proper leaf
        X_gp = X.detach().requires_grad_(True)
        T_X_gp = model(X_gp)
    else:
        # Reuse pre-computed output; X must already be the requires_grad leaf
        X_gp = X
        T_X_gp = T_X

    if hutchinson:
        v = torch.randn_like(T_X_gp)
        grad = torch.autograd.grad(
            (T_X_gp * v).sum(),
            X_gp,
            create_graph=True,
            retain_graph=True,   # keep graph so the caller can still backward
        )[0]  # (B, d)
        gp = grad.pow(2).sum(dim=1).mean()
    else:
        p = T_X_gp.shape[1]
        gp = torch.zeros(1, device=X.device)
        for k in range(p):
            grad_k = torch.autograd.grad(
                T_X_gp[:, k].sum(),
                X_gp,
                create_graph=True,
                retain_graph=True,
            )[0]
            gp = gp + grad_k.pow(2).sum(dim=1).mean()

    return gp


def brot_loss(
    model: nn.Module,
    X: torch.Tensor,
    Y_tilde: torch.Tensor,
    alpha: float = 0.1,
    hutchinson: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Full BROT training loss = MSE + alpha * gradient_penalty.

    Uses a SINGLE forward pass for both terms to avoid double-backward errors.

    Args:
        model:      Transport map T.
        X:          Source batch (B, d).
        Y_tilde:    Barycentric targets (B, d).
        alpha:      Gradient penalty weight.
        hutchinson: Use Hutchinson estimator.

    Returns:
        Dict with keys 'mse', 'gp', 'total'.
    """
    # Single forward pass: X_gp is the requires_grad leaf for GP computation
    X_gp = X.detach().requires_grad_(True)
    T_X = model(X_gp)

    mse = (Y_tilde - T_X).pow(2).sum(dim=1).mean()

    if alpha > 0.0:
        gp = gradient_penalty(model, X_gp, T_X=T_X, hutchinson=hutchinson)
        total = mse + alpha * gp
    else:
        gp = torch.zeros(1, device=X.device)
        total = mse

    return {"mse": mse, "gp": gp, "total": total}
