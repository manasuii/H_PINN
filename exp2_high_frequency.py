"""
exp2_high_frequency.py
======================
Experiment 2: High-frequency initial conditions.

  u(x,0) = sin(n*pi*x),   n = 1, 3, 5, 10

Hypothesis:
  Higher n means faster oscillation in space and time.
  The network must capture finer structure — PDE residual
  minimisation alone struggles, and energy drift grows with n.
  H-PINN's explicit conservation penalty helps most at high n.

Key physics:
  Exact solution: u(x,t) = sin(n*pi*x) * cos(n*pi*c*t)
  Exact energy:   E = (n*pi*c)^2 / 2   (scales as n^2)

Run from D:\\PINN_Conservative:
    python experiments/exp2_high_frequency.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.makedirs("results/experiments", exist_ok=True)

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pinn_core        import PINN, PINNTrainer, DEVICE
from src.conservation_audit import audit_wave, eval_on_grid
from src.hamiltonian_pinn import HamiltonianWaveLoss, HPINNTrainer

# ── Generalised losses for frequency n ───────────────────────────────────────

class WaveLossFreq:
    """Wave loss for IC: u(x,0) = sin(n*pi*x), u_t(x,0) = 0."""
    def __init__(self, c=1.0, n=1, w_f=1.0, w_b=10.0, w_i=10.0):
        self.c, self.n = c, n
        self.w_f, self.w_b, self.w_i = w_f, w_b, w_i

    def __call__(self, model, x_f, t_f, x0, t0, x1, t1, x_i, t_i):
        from src.pinn_core import compute_derivatives, grad
        import math
        _, _, u_xx, u_t, u_tt = compute_derivatives(model, x_f, t_f)
        r_f  = u_tt - self.c**2 * u_xx
        L_f  = (r_f**2).mean()
        L_b  = (model(x0, t0)**2).mean() + (model(x1, t1)**2).mean()
        u_i  = model(x_i, t_i)
        u_ex = torch.sin(self.n * math.pi * x_i)
        _, _, _, u_t_i, _ = compute_derivatives(model, x_i, t_i)
        L_i  = ((u_i - u_ex)**2).mean() + (u_t_i**2).mean()
        total = self.w_f * L_f + self.w_b * L_b + self.w_i * L_i
        return {"total": total, "pde": L_f, "bc": L_b, "ic": L_i}


class HWaveLossFreq:
    """H-PINN wave loss for frequency n."""
    def __init__(self, c=1.0, n=1, w_f=1.0, w_b=10.0, w_i=10.0, w_H=0.01):
        self.c, self.n = c, n
        self.w_f, self.w_b, self.w_i, self.w_H = w_f, w_b, w_i, w_H

    def __call__(self, model, x_f, t_f, x0, t0, x1, t1, x_i, t_i,
                 x_h=None, t_h=None):
        from src.pinn_core import compute_derivatives, grad
        from src.hamiltonian_pinn import compute_dHdt
        import math
        _, _, u_xx, u_t, u_tt = compute_derivatives(model, x_f, t_f)
        r_f  = u_tt - self.c**2 * u_xx
        L_f  = (r_f**2).mean()
        L_b  = (model(x0, t0)**2).mean() + (model(x1, t1)**2).mean()
        u_i  = model(x_i, t_i)
        u_ex = torch.sin(self.n * math.pi * x_i)
        _, _, _, u_t_i, _ = compute_derivatives(model, x_i, t_i)
        L_i  = ((u_i - u_ex)**2).mean() + (u_t_i**2).mean()
        x_hh = x_h if x_h is not None else x_f
        t_hh = t_h if t_h is not None else t_f
        L_H  = (compute_dHdt(model, x_hh, t_hh, self.c)**2).mean()
        total = self.w_f * L_f + self.w_b * L_b + self.w_i * L_i + self.w_H * L_H
        return {"total": total, "pde": L_f, "bc": L_b, "ic": L_i, "hamiltonian": L_H}


def exact_wave_freq(X, T, c=1.0, n=1):
    import math
    return np.sin(n * math.pi * X) * np.cos(n * math.pi * c * T)

def exact_energy_freq(c=1.0, n=1):
    import math
    return (n * math.pi * c)**2 / 2


# ── Audit for arbitrary frequency ─────────────────────────────────────────────

def audit_wave_freq(model, c=1.0, n=1, T=1.0, label=""):
    """Same as audit_wave but uses frequency-n exact energy."""
    from src.conservation_audit import eval_on_grid, wave_energy
    import math
    grid  = eval_on_grid(model, nx=256, nt=200, T=T)
    E     = wave_energy(grid, c)
    E0_ex = exact_energy_freq(c, n)
    drift_ex = (E - E0_ex) / E0_ex

    x_np = np.linspace(0, 1, 256)
    t_np = np.linspace(0, T, 200)
    X, T_ = np.meshgrid(x_np, t_np, indexing='ij')
    u_ex  = exact_wave_freq(X, T_, c, n)
    err   = np.sqrt(np.sum((grid["u"] - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))

    print(f"  [{label}] L2 err={err:.3e} | E0={E[0]:.4f} vs exact {E0_ex:.4f} "
          f"| max drift={np.max(np.abs(drift_ex))*100:.2f}%")
    return {"t": grid["t"], "E": E, "E0_exact": E0_ex,
            "drift_ex": drift_ex, "err_L2": err}


# ── Run experiments ───────────────────────────────────────────────────────────
C      = 1.0
T      = 1.0
EPOCHS = 15_000
FREQS  = [1, 3, 5, 10]

std_results   = {}
hpinn_results = {}

for n in FREQS:
    print(f"\n{'='*60}\n  Frequency n={n}   (IC = sin({n}πx))\n{'='*60}")

    # wider network for harder problems
    hidden = 64 if n <= 3 else 96
    depth  = 5  if n <= 3 else 6

    m_std = PINN(hidden=hidden, depth=depth)
    PINNTrainer(m_std, WaveLossFreq(c=C, n=n),
                N_f=6000, N_b=200, N_i=200, T=T, lr=1e-3
                ).train(EPOCHS, print_every=5000)
    std_results[n] = audit_wave_freq(m_std, c=C, n=n, T=T, label=f"Std n={n}")

    m_hp = PINN(hidden=hidden, depth=depth)
    HPINNTrainer(m_hp, HWaveLossFreq(c=C, n=n, w_H=0.01),
                 N_f=6000, N_b=200, N_i=200, T=T, lr=1e-3
                 ).train(EPOCHS, print_every=5000)
    hpinn_results[n] = audit_wave_freq(m_hp, c=C, n=n, T=T, label=f"H-PINN n={n}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(FREQS), figsize=(6 * len(FREQS), 9))
fig.suptitle("Wave Equation – High-Frequency Initial Conditions", fontsize=14, fontweight='bold')

for col, n in enumerate(FREQS):
    rs = std_results[n];   rh = hpinn_results[n]
    t  = rs["t"]

    # Row 0: absolute energy
    ax = axes[0, col]
    ax.axhline(rs["E0_exact"], color="k", ls="--", lw=1.2, label=f"Exact E₀")
    ax.plot(t, rs["E"], color="tab:blue",   lw=1.8, label="Std PINN")
    ax.plot(t, rh["E"], color="tab:orange", lw=1.8, label="H-PINN")
    ax.set_title(f"n={n}  |  IC = sin({n}πx)")
    ax.set_xlabel("t");  ax.set_ylabel("E(t)")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # Row 1: drift %
    ax = axes[1, col]
    ax.plot(t, rs["drift_ex"] * 100, color="tab:blue",   lw=1.8, label="Std PINN")
    ax.plot(t, rh["drift_ex"] * 100, color="tab:orange", lw=1.8, label="H-PINN")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("t");  ax.set_ylabel("Drift (%)")
    ax.set_title(f"Energy drift (%) — n={n}")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/experiments/exp2_high_frequency.png", dpi=150, bbox_inches='tight')
print("\nSaved: results/experiments/exp2_high_frequency.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  {'n':>4} | {'Std L2 err':>12} | {'H-PINN L2 err':>14} | "
      f"{'Std drift':>10} | {'H-PINN drift':>13}")
print(f"{'─'*65}")
for n in FREQS:
    rs = std_results[n];  rh = hpinn_results[n]
    print(f"  {n:>4} | {rs['err_L2']:>12.3e} | {rh['err_L2']:>14.3e} | "
          f"{np.max(np.abs(rs['drift_ex']))*100:>9.2f}% | "
          f"{np.max(np.abs(rh['drift_ex']))*100:>12.2f}%")
print(f"{'─'*65}")