"""
Statistical convergence of BROT to a known Lipschitz OT map (Section 5.1).

For each training size n = m, BROT is fitted on n source and m target samples
of the benchmark in `brot.datasets.LipschitzBenchmark`, and the estimation
error E||T_hat(X) - T_0(X)||^2 is measured on 2,000 held-out source points.
The settings below are the ones used for BROT in Figure 2 of the paper:
a 5-layer ReLU MLP (64-128-256-128-64), Adam (lr 1e-3, weight decay 1e-6),
full-batch training for 600 epochs, Jacobian penalty weight 0.01, the full
(unregularized, unchunked) empirical OT plan, and z-scored targets.

Usage:
    python examples/convergence.py                  # quick run
    python examples/convergence.py --paper          # full grid, 10 seeds
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from brot import BROTConfig, BROTTrainer
from brot.datasets import LipschitzBenchmark

PAPER_NS = [500, 1000, 2000, 3000, 5000, 8000, 10000]
N_TEST, TEST_SEED = 2000, 12345


def fit_brot(x: torch.Tensor, y: torch.Tensor, device: str, n_epochs: int = 600):
    """Fit BROT with z-scored targets; return the trainer and the target scaling."""
    mu, sd = y.mean(0, keepdim=True), y.std(0, keepdim=True).clamp(min=1e-8)
    config = BROTConfig(
        in_dim=x.shape[1], out_dim=y.shape[1],
        hidden_dims=[64, 128, 256, 128, 64],
        activation="relu", optimizer="adam", lr=1e-3, weight_decay=1e-6,
        n_epochs=n_epochs, batch_size=len(x),
        alpha=0.01, ot_reg=0.0, emd_batch_size=None,
        log_every=max(1, n_epochs // 2), verbose=False,
    )
    trainer = BROTTrainer(config=config, device=device)
    trainer.fit(x, (y - mu) / sd)
    return trainer, mu, sd


def estimation_error(trainer, mu, sd, x_test, t_test) -> float:
    with torch.no_grad():
        pred = trainer.transform(x_test).cpu() * sd + mu
    return float(((pred - t_test) ** 2).sum(-1).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--paper", action="store_true",
                   help="use the full grid and 10 seeds of the paper")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="convergence.csv")
    p.add_argument("--threads", type=int, default=3,
                   help="CPU threads; the reported numbers use 3")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.paper:
        args.ns, args.n_repeats = PAPER_NS, 10

    bench = LipschitzBenchmark()
    x_test = bench.sample_source(N_TEST, seed=TEST_SEED)
    t_test = bench.true_map(x_test)
    print(f"Lipschitz constant of T_0 on the support: {bench.lipschitz_constant:.2f}")

    results = {}
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n", "repeat", "seed", "error", "fit_s"])
        for n in args.ns:
            errs = []
            for r in range(args.n_repeats):
                seed = 1000 + 7 * n + r
                x = bench.sample_source(n, seed=seed)
                y = bench.sample_target(n, seed=seed + 100000)
                t0 = time.time()
                torch.manual_seed(seed)  # network initialization and penalty probes
                trainer, mu, sd = fit_brot(x, y, args.device)
                err = estimation_error(trainer, mu, sd, x_test, t_test)
                errs.append(err)
                writer.writerow([n, r, seed, f"{err:.6f}", f"{time.time() - t0:.1f}"])
                fh.flush()
                print(f"  n={n:6d}  seed={seed}  error={err:.4f}")
            results[n] = float(np.mean(errs))
            print(f"n={n:6d}  mean error over {len(errs)} seeds: {results[n]:.3f}")

    print("\nn      mean error")
    for n, e in results.items():
        print(f"{n:<6d} {e:.3f}")


if __name__ == "__main__":
    main()
