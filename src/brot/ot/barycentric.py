"""
Barycentric projection from coupling matrix.

Given coupling gamma (n, m) and target samples Y (m, d),
computes barycentric targets:

    Y_tilde_i = sum_j n * gamma_ij * Y_j   ==>   Y_tilde = n * gamma @ Y
"""

import torch


def barycentric_projection(
    gamma: torch.Tensor,  # (n, m), rows sum to 1/n
    Y: torch.Tensor,      # (m, d)
) -> torch.Tensor:        # (n, d)
    """
    Compute barycentric targets Y_tilde = n * gamma @ Y.

    Each Y_tilde_i is the weighted centroid of target points
    under the conditional coupling gamma[i, :] / (1/n).
    """
    n = gamma.shape[0]
    return n * (gamma @ Y)
