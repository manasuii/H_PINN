"""
exp1_long_horizon.py
====================
Experiment 1: Long time horizons T = 1, 5, 10.

Hypothesis:
  Standard PINN energy drift accumulates over time.
  H-PINN suppresses this drift because its dH/dt penalty
  acts at every epoch, not just at t close to zero.

Run from D:\\PINN_Conservative:
    python experiments/exp1_long_horizon.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.makedirs("results/experiments", exist_ok=True)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pinn_core        import PINN, PINNTrainer, WaveLoss, DEVICE
from src.conservation_audit import audit_wave
from src.hamiltonian_pinn import HamiltonianWaveLoss, HPINNTrainer

# ── Config ────────────────────────────────────────────────────────────────────
C        = 1.0
EPOCHS   = 15_000
LR       = 1e-3
N_F      = 8000      # more points needed for longer domain
N_B      = 300
N_I      = 300
HORIZONS = [1.0, 5.0, 10.0]

results = {}   # keyed by T

for T in HORIZONS:
    print(f"\n{'='*60}")
    print(f"  T = {T}")
    print(f"{'='*60}")

    # Standard PINN
    m_std = PINN(hidden=64, depth=5)
    PINNTrainer(m_std, WaveLoss(c=C), N_f=N_F, N_b=N_B, N_i=N_I,
                T=T, lr=LR).train(EPOCHS, print_every=3000)
    a_std = audit_wave(m_std, c=C, T=T, label=f"Std T={T}")

    # H-PINN  (gentle w_H, warmup handled inside HPINNTrainer)
    m_hpinn = PINN(hidden=64, depth=5)
    HPINNTrainer(m_hpinn, HamiltonianWaveLoss(c=C, w_H=0.01),
                 N_f=N_F, N_b=N_B, N_i=N_I,
                 T=T, lr=LR).train(EPOCHS, print_every=3000)
    a_hpinn = audit_wave(m_hpinn, c=C, T=T, label=f"H-PINN T={T}")

    results[T] = {"std": a_std, "hpinn": a_hpinn}

# ── Plot: energy drift vs time for each T ─────────────────────────────────────
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(6 * len(HORIZONS), 5), sharey=False)
fig.suptitle("Wave Equation – Energy Drift vs Time Horizon", fontsize=14, fontweight='bold')

for ax, T in zip(axes, HORIZONS):
    r = results[T]
    ax.plot(r["std"]["t"],   r["std"]["drift_ex"]  * 100,
            label="Std PINN", color="tab:blue", lw=1.8)
    ax.plot(r["hpinn"]["t"], r["hpinn"]["drift_ex"] * 100,
            label="H-PINN",   color="tab:orange", lw=1.8)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title(f"T = {T}")
    ax.set_xlabel("t");  ax.set_ylabel("Energy drift (%)")
    ax.legend();  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/experiments/exp1_long_horizon.png", dpi=150, bbox_inches='tight')
print("\nSaved: results/experiments/exp1_long_horizon.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  {'T':>5} | {'Std max drift':>15} | {'H-PINN max drift':>17} | {'Improvement':>12}")
print(f"{'─'*65}")
for T in HORIZONS:
    r   = results[T]
    sd  = np.max(np.abs(r["std"]["drift_ex"]))   * 100
    hd  = np.max(np.abs(r["hpinn"]["drift_ex"])) * 100
    imp = sd / max(hd, 1e-6)
    print(f"  {T:>5} | {sd:>14.2f}% | {hd:>16.2f}% | {imp:>10.2f}×")
print(f"{'─'*65}")

# save summary
with open("results/experiments/exp1_summary.txt", "w") as f:
    f.write("Experiment 1: Long Horizon Energy Drift\n")
    f.write(f"{'─'*65}\n")
    f.write(f"  {'T':>5} | {'Std max drift':>15} | {'H-PINN max drift':>17} | Improvement\n")
    f.write(f"{'─'*65}\n")
    for T in HORIZONS:
        r  = results[T]
        sd = np.max(np.abs(r["std"]["drift_ex"])) * 100
        hd = np.max(np.abs(r["hpinn"]["drift_ex"])) * 100
        f.write(f"  {T:>5} | {sd:>14.2f}% | {hd:>16.2f}% | {sd/max(hd,1e-6):.2f}×\n")
print("Saved: results/experiments/exp1_summary.txt")