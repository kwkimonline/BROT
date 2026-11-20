"""
BROTTrainer: orchestrates the two-step BROT algorithm.

Step 1 – Solve the Kantorovich problem:
    gamma = solve_kantorovich(X, Y)       # (n, m) coupling matrix
    Y_tilde = n * gamma @ Y              # (n, d) barycentric targets

Step 2 – Fit a Lipschitz DNN by regression:
    min_T  (1/n) sum_i ||Y_tilde_i - T(X_i)||^2  +  alpha * GradPenalty(T, X)

Usage:
    trainer = BROTTrainer(in_dim=2, out_dim=2)
    trainer.fit(X_train, Y_train)
    T_X_test = trainer.transform(X_test)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from brot.losses.gradient_penalty import brot_loss
from brot.metrics import mmd, wasserstein
from brot.models.mlp import ActivationType, LipschitzMLP
from brot.ot.barycentric import barycentric_projection
from brot.ot.kantorovich import CostFn, solve_kantorovich


def _as_tensor(x, device) -> torch.Tensor:
    """Accept torch tensors or array-likes (e.g. NumPy) and return float32 on device."""
    return torch.as_tensor(x, dtype=torch.float32).to(device)


@dataclass
class BROTConfig:
    # ── OT solver ─────────────────────────────────────────────────────────
    ot_cost: CostFn = "sqeuclidean"
    ot_reg: float = 0.0            # 0 = exact EMD; >0 = Sinkhorn
    ot_reg_type: str = "entropy"
    emd_batch_size: Optional[int] = None  # if set, use batched EMD (reg must be 0)

    # ── Network ───────────────────────────────────────────────────────────
    in_dim: int = 2
    out_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 256, 128, 64])
    activation: ActivationType = "relu"
    spectral_norm: bool = False
    use_residual: bool = False     # residual MLP (all hidden dims treated as uniform width)

    # ── Training ──────────────────────────────────────────────────────────
    optimizer: str = "adam"         # 'sgd' or 'adam'
    lr: float = 1e-3
    momentum: float = 0.9          # for SGD
    weight_decay: float = 0.0
    n_epochs: int = 1000
    batch_size: int = 1024
    alpha: float = 0.01             # gradient penalty weight
    lr_scheduler: Optional[str] = None  # None | "cosine" | "step"
    lr_step_size: int = 300         # for "step" scheduler
    lr_gamma: float = 0.5           # for "step" scheduler

    # ── Logging ───────────────────────────────────────────────────────────
    log_every: int = 100
    verbose: bool = True


class BROTTrainer:
    """
    Two-step BROT estimator.

    Can accept a custom model (e.g., UNet for image translation).
    When model=None, builds a LipschitzMLP from config.
    """

    def __init__(
        self,
        config: Optional[BROTConfig] = None,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        **kwargs,
    ):
        if config is None:
            config = BROTConfig(**kwargs)
        self.config = config
        self.device = torch.device(device)

        # Build model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = LipschitzMLP(
                in_dim=config.in_dim,
                out_dim=config.out_dim,
                hidden_dims=config.hidden_dims,
                activation=config.activation,
                spectral_norm=config.spectral_norm,
                use_residual=config.use_residual,
            ).to(self.device)

        # Build optimizer
        if config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Build LR scheduler
        self.scheduler = None
        if config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )
        elif config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
            )

        # Stored after Step 1
        self.gamma_: Optional[torch.Tensor] = None
        self.Y_tilde_: Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: OT coupling
    # ──────────────────────────────────────────────────────────────────────

    def compute_coupling(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve Kantorovich problem and compute barycentric targets.

        Returns:
            gamma:   (n, m) coupling matrix
            Y_tilde: (n, d) barycentric targets
        """
        X = _as_tensor(X, self.device)
        Y = _as_tensor(Y, self.device)
        cfg = self.config
        if cfg.verbose:
            print("  [Step 1] Solving Kantorovich problem ...", end=" ", flush=True)
        t0 = time.time()

        gamma = solve_kantorovich(
            X, Y,
            cost=cfg.ot_cost,
            reg=cfg.ot_reg,
            reg_type=cfg.ot_reg_type,
            emd_batch_size=cfg.emd_batch_size,
            device=str(self.device),
        )
        Y_tilde = barycentric_projection(gamma, Y)

        if cfg.verbose:
            print(f"done ({time.time() - t0:.1f}s)")

        self.gamma_ = gamma
        self.Y_tilde_ = Y_tilde
        return gamma, Y_tilde

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: DNN regression
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Y_tilde: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        Y_val: Optional[torch.Tensor] = None,
        select_best: bool = False,
        val_every: int = 10,
        val_metric: str = "wass",
    ) -> Dict[str, List[float]]:
        """
        Full BROT training: Step 1 + Step 2.

        Args:
            X:       Source samples (n, d).
            Y:       Target samples (m, d).
            Y_tilde: Precomputed barycentric targets (n, d).
                     If None, computed from gamma.

        Returns:
            history: Dict with 'mse', 'gp', 'total' loss lists.
        """
        X = _as_tensor(X, self.device)
        Y = _as_tensor(Y, self.device)
        if X_val is not None:
            X_val = _as_tensor(X_val, self.device)
        if Y_val is not None:
            Y_val = _as_tensor(Y_val, self.device)

        # Step 1
        if Y_tilde is None:
            _, Y_tilde = self.compute_coupling(X, Y)
        Y_tilde = _as_tensor(Y_tilde, self.device)

        # Step 2
        cfg = self.config
        if cfg.verbose:
            print("  [Step 2] Training DNN ...")

        dataset = TensorDataset(X, Y_tilde)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        history: Dict[str, List[float]] = {"mse": [], "gp": [], "total": []}
        if select_best and X_val is not None and Y_val is not None:
            history[f"val_{val_metric}"] = []
        best_score = float("inf")
        best_state = None
        select_by_train = select_best and val_metric == "train_loss"
        self.model.train()

        for epoch in range(cfg.n_epochs):
            epoch_losses: Dict[str, float] = {"mse": 0.0, "gp": 0.0, "total": 0.0}

            for X_batch, Yt_batch in loader:
                self.optimizer.zero_grad()
                losses = brot_loss(
                    self.model, X_batch, Yt_batch, alpha=cfg.alpha
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()

            n_batches = len(loader)
            for k in ("mse", "gp", "total"):
                history[k].append(epoch_losses[k] / n_batches)

            if self.scheduler is not None:
                self.scheduler.step()

            if cfg.verbose and (epoch + 1) % cfg.log_every == 0:
                mse = history["mse"][-1]
                gp = history["gp"][-1]
                total = history["total"][-1]
                print(
                    f"    Epoch [{epoch+1:4d}/{cfg.n_epochs}] "
                    f"mse={mse:.4f}  gp={gp:.4f}  total={total:.4f}"
                )

            if select_by_train:
                # Pick the epoch with the lowest training objective (mse + α·gp).
                # No held-out data needed; uses history["total"][-1] just appended.
                score = history["total"][-1]
                if score < best_score:
                    best_score = score
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    if cfg.verbose:
                        print(f"    best checkpoint updated (train_loss={best_score:.4f})")
            elif (
                select_best
                and X_val is not None
                and Y_val is not None
                and ((epoch + 1) % max(val_every, 1) == 0 or (epoch + 1) == cfg.n_epochs)
            ):
                self.model.eval()
                with torch.no_grad():
                    T_X_val = self.transform(X_val, batch_size=cfg.batch_size)
                if val_metric == "mmd":
                    score = float(mmd(T_X_val, Y_val, bandwidth=1.0))
                else:
                    score = float(wasserstein(T_X_val, Y_val, p=2, reg=0.0))
                history[f"val_{val_metric}"].append(score)
                if cfg.verbose:
                    print(f"    val_{val_metric}={score:.4f}")
                if score < best_score:
                    best_score = score
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    if cfg.verbose:
                        print(f"    best checkpoint updated (val_{val_metric}={best_score:.4f})")
                self.model.train()

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if cfg.verbose:
                tag = "train_loss" if select_by_train else f"val_{val_metric}"
                print(f"  [BROT] loaded best checkpoint ({tag}={best_score:.4f})")
        self.model.eval()
        return history

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transform(
        self,
        X: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """
        Apply learned transport map T to source samples X.

        Args:
            X: (n, d) source samples.

        Returns:
            T(X): (n, d) transported samples.
        """
        self.model.eval()
        X = _as_tensor(X, self.device)
        results = []
        for i in range(0, len(X), batch_size):
            results.append(self.model(X[i : i + batch_size]))
        return torch.cat(results, dim=0)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )
        print(f"  [BROT] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"  [BROT] Loaded checkpoint ← {path}")
