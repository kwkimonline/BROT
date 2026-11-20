"""
Lipschitz-constrained MLP used as the transport map T.

Supports ReLU and ReQU (Rectified Quadratic Unit) activations,
as used in the paper for different experiment settings:
  - Simulation / domain adaptation: ReLU, hidden=(64,128,256,256,128,64)
  - Theory section: ReQU (smooth approximation, supports theoretical guarantees)
  - Domain adaptation (CLIP embedding): ReLU, hidden=(1024,2048,1024)

Lipschitz constraint is enforced via gradient penalty (see losses/gradient_penalty.py).
Optionally, spectral normalization can be applied for a hard constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

ActivationType = Literal["relu", "leaky_relu", "elu", "gelu", "tanh", "requ", "sigmoid"]


class ReQU(nn.Module):
    """Rectified Quadratic Unit: f(x) = 0.5 * max(x, 0)^2.

    Smooth activation used for theoretical guarantees on the approximation
    rate of Lipschitz functions by DNNs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.relu(x).pow(2)


def _make_activation(name: ActivationType) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "requ": ReQU(),
        "sigmoid": nn.Sigmoid(),
    }[name]


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: ActivationType, spectral_norm: bool):
        super().__init__()
        def make_linear(d_in, d_out):
            l = nn.Linear(d_in, d_out)
            return nn.utils.spectral_norm(l) if spectral_norm else l
        self.block = nn.Sequential(
            make_linear(dim, dim),
            _make_activation(activation),
            make_linear(dim, dim),
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        return self.act((x + self.block(x)) / math.sqrt(2))


class LipschitzMLP(nn.Module):
    """
    Multi-layer perceptron transport map T: R^{in_dim} → R^{out_dim}.

    Lipschitz constraint is enforced by:
      - gradient_penalty=True (soft, via training loss)  — default, as in the paper
      - spectral_norm=True    (hard, on each Linear layer)
      - Both can be combined for stronger control.

    Args:
        in_dim:        Source dimension.
        out_dim:       Target dimension.
        hidden_dims:   List of hidden layer widths.
        activation:    Nonlinearity between layers.
        spectral_norm: Whether to apply spectral normalization to weight matrices.
        use_residual:  Use residual blocks (all hidden dims must be equal).
        use_bn:        Whether to use BatchNorm (not recommended for OT maps).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int] = [64, 128, 256, 256, 128, 64],
        activation: ActivationType = "relu",
        spectral_norm: bool = False,
        use_residual: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()

        def make_linear(d_in, d_out):
            l = nn.Linear(d_in, d_out)
            return nn.utils.spectral_norm(l) if spectral_norm else l

        if use_residual:
            # All hidden dims must be the same for residual connections.
            # Use the first hidden dim as the uniform width.
            h = hidden_dims[0]
            n_blocks = len(hidden_dims)
            layers: List[nn.Module] = [
                make_linear(in_dim, h),
                _make_activation(activation),
            ]
            for _ in range(n_blocks):
                layers.append(_ResidualBlock(h, activation, spectral_norm))
            layers.append(make_linear(h, out_dim))
        else:
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(make_linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    if use_bn:
                        layers.append(nn.BatchNorm1d(dims[i + 1]))
                    layers.append(_make_activation(activation))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
