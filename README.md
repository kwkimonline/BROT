# BROT: Barycentric Regression for Optimal Transport

Official repository for **[Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality](https://arxiv.org/abs/2609.06598)**, published in NeurIPS2026, selected as a spotlight presentation.

BROT estimates the optimal transport (OT) map in two steps.
It first computes the unregularized empirical OT plan between the source and target data,
and then fits a Lipschitz-constrained DNN to the induced barycentric targets by least-squares regression.
The DNN estimator of BROT attains the minimax convergence rate when the ground-truth OT map is Lipschitz,
while remaining trainable by a standard regression without min-max optimization.

## Installation

```bash
git clone https://github.com/kwkimonline/BROT.git
cd BROT
pip install -e .
```

The only dependencies are PyTorch, NumPy, SciPy, and [POT](https://pythonot.github.io/).

## Environment

The code was developed and tested with the following environment:

| | version |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.7.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| POT | 0.9.6 |

```bash
conda create -n brot python=3.10 -y && conda activate brot
pip install torch==2.7.1 numpy==2.2.6 scipy==1.15.3 pot==0.9.6.post1
pip install -e .
```

The example in Section 5.1 was run on CPU (Intel Xeon Gold 6226R) with 3 threads (`--threads 3`).
With the seeds fixed, the results are deterministic for a given machine,
but they can differ slightly across hardware, thread counts, and library versions.

## Quick start

```python
import torch
from brot import BROTConfig, BROTTrainer

x = torch.randn(1000, 2)                    # source data
y = 2.0 * torch.randn(1000, 2) + 3.0        # target data

config = BROTConfig(in_dim=2, out_dim=2, hidden_dims=[64, 128, 64],
                    n_epochs=200, alpha=0.01)   # alpha: Jacobian penalty weight
trainer = BROTTrainer(config=config)
trainer.fit(x, y)                            # Step 1 (OT plan) + Step 2 (regression)
y_hat = trainer.transform(x)                 # transported source data
```

Main options of `BROTConfig`:
`hidden_dims` and `activation` (`"relu"`, `"requ"`, `"gelu"`, ...) define the MLP,
`alpha` is the weight of the Jacobian penalty that enforces the Lipschitz constraint,
`ot_reg=0.0` uses the unregularized empirical OT plan,
and `emd_batch_size` enables the chunked OT plan for large-scale data.

## Using your own data

`fit` and `transform` accept PyTorch tensors, NumPy arrays, or nested lists of shape `(n, d)` (source) and `(m, d)` (target), with `n` and `m` possibly different.
Data are cast to `float32` and moved to `device` (`"cpu"` or `"cuda"`).

```python
import numpy as np
from brot import BROTConfig, BROTTrainer

x = np.load("source.npy")                   # (n, d)
y = np.load("target.npy")                   # (m, d)
mu, sd = y.mean(0), y.std(0) + 1e-8         # standardizing the target is recommended

trainer = BROTTrainer(config=BROTConfig(in_dim=x.shape[1], out_dim=y.shape[1]), device="cuda")
trainer.fit(x, (y - mu) / sd)
y_hat = trainer.transform(x).cpu().numpy() * sd + mu

trainer.save("brot.pt")                     # model and optimizer state; restore with trainer.load("brot.pt")
```

The full OT plan needs an `n x m` matrix in memory; for large `n` and `m`, set `emd_batch_size` (e.g. `5000`) to compute it in chunks.
Validation-based model selection is available through `trainer.fit(x, y, X_val=..., Y_val=..., select_best=True)`.

## Example: convergence to a known Lipschitz OT map (Section 5.1)

`examples/convergence.py` runs BROT on the synthetic benchmark of Section 5.1 (Figure 2),
where the ground-truth OT map is known:
the source is the standard Gaussian truncated to the disk of radius 3.5,
and the OT map is $\mathbf T_0(x) = (1.5x_1 + 0.3x_1^3,\ 0.8x_2 + 0.1x_2^3)$
(`brot.datasets.LipschitzBenchmark`).

```bash
python examples/convergence.py            # quick run: n in {500, 1000, 2000}, 3 seeds
python examples/convergence.py --paper    # the full setting of the paper: 7 sample sizes, 10 seeds
```

The script prints the mean estimation error $\mathbb E\|\widehat{\mathbf T}(X)-\mathbf T_0(X)\|^2$ on 2,000 held-out source points for each $n$
and saves the per-seed errors to `convergence.csv`.

## Scope

This repository contains the BROT estimator and the synthetic benchmarks.
The experiments on real data (AFHQv2, single-cell perturbation, and domain adaptation) and the baseline methods are not included.

## Citation

```bibtex
@article{kim2026brot,
  title={Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality},
  author={Kim, Kunwoong and Kong, Insung and Kim, Yongdai},
  year={2026},
  journal={arXiv preprint arXiv:2609.06598},
  url={https://arxiv.org/abs/2609.06598}
}
```

## License

[MIT License](LICENSE).
