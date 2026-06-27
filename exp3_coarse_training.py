"""
exp3_coarse_training.py
=======================
Experiment 3: Data-scarce / compute-constrained regime.

Vary number of collocation points and training epochs.
Hypothesis: when training budget is tight (realistic for
large-scale problems), H-PINN's explicit conservation
penalty provides a stronger inductive bias, compensating
for insufficient PDE residual convergence.

This is the most practically relevant experiment — it
mirrors real deployment where you can't just train longer.

Run from D:\\PINN_Conservative:
    python experiments/exp3_coarse_training.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.makedirs("results/experiments", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pinn_core        import PINN, PINNTrainer, WaveLoss
from src.conservation_audit import audit_wave
from src.hamiltonian_pinn import HamiltonianWaveLoss, HPINNTrainer

C = 1.0
T = 1.0

# Grid of (epochs, N_f) — from very coarse to moderate
configs = [
    {"epochs": 1_000,  "N_f": 500,  "label": "1k ep / 500 pts"},
    {"epochs": 3_000,  "N_f": 1000, "label": "3k ep / 1k pts"},
    {"epochs": 5_000,  "N_f": 2000, "label": "5k ep / 2k pts"},
    {"epochs": 10_000, "N_f": 5000, "label": "10k ep / 5k pts"},
]

std_drift   = []
hpinn_drift = []
std_err     = []
hpinn_err   = []
labels      = []

for cfg in configs:
    ep   = cfg["epochs"]
    nf   = cfg["N_f"]
    lbl  = cfg["label"]
    print(f"\n{'='*55}\n  Config: {lbl}\n{'='*55}")

    m_std = PINN(hidden=64, depth=5)
    PINNTrainer(m_std, WaveLoss(c=C),
                N_f=nf, N_b=100, N_i=100, T=T, lr=1e-3
                ).train(ep, print_every=ep)
    a_std = audit_wave(m_std, c=C, T=T, label=f"Std {lbl}")

    m_hp = PINN(hidden=64, depth=5)
    HPINNTrainer(m_hp, HamiltonianWaveLoss(c=C, w_H=0.01),
                 N_f=nf, N_b=100, N_i=100, T=T, lr=1e-3
                 ).train(ep, print_every=ep)
    a_hp = audit_wave(m_hp, c=C, T=T, label=f"H-PINN {lbl}")

    std_drift.append(np.max(np.abs(a_std["drift_ex"])) * 100)
    hpinn_drift.append(np.max(np.abs(a_hp["drift_ex"])) * 100)
    std_err.append(a_std["err_L2"])
    hpinn_err.append(a_hp["err_L2"])
    labels.append(lbl)

# ── Plot ──────────────────────────────────────────────────────────────────────
x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Experiment 3: Training Budget vs Conservation Quality",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.bar(x - width/2, std_drift,   width, label="Std PINN",  color="tab:blue",   alpha=0.8)
ax.bar(x + width/2, hpinn_drift, width, label="H-PINN",    color="tab:orange", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Max energy drift (%)"); ax.set_title("Energy conservation")
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x - width/2, std_err,   width, label="Std PINN",  color="tab:blue",   alpha=0.8)
ax.bar(x + width/2, hpinn_err, width, label="H-PINN",    color="tab:orange", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Relative L2 error"); ax.set_title("Pointwise accuracy")
ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/experiments/exp3_coarse_training.png", dpi=150, bbox_inches='tight')
print("\nSaved: results/experiments/exp3_coarse_training.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  {'Config':<22} | {'Std drift':>10} | {'H-PINN drift':>13} | "
      f"{'Std L2':>8} | {'H-PINN L2':>10}")
print(f"{'─'*70}")
for i, lbl in enumerate(labels):
    print(f"  {lbl:<22} | {std_drift[i]:>9.2f}% | {hpinn_drift[i]:>12.2f}% | "
          f"{std_err[i]:>8.3e} | {hpinn_err[i]:>10.3e}")
print(f"{'─'*70}")

with open("results/experiments/exp3_summary.txt", "w") as f:
    f.write("Experiment 3: Coarse Training Regime\n")
    f.write(f"{'─'*70}\n")
    for i, lbl in enumerate(labels):
        f.write(f"{lbl}: std_drift={std_drift[i]:.2f}%  hpinn_drift={hpinn_drift[i]:.2f}%  "
                f"std_L2={std_err[i]:.3e}  hpinn_L2={hpinn_err[i]:.3e}\n")