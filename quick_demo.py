"""
quick_demo.py
=============
Fast demo run (2000 epochs) to verify everything works end-to-end.
For full results use train_heat.py and train_wave.py (15k epochs each).

Usage:
    python quick_demo.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("results", exist_ok=True)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless for demo
import matplotlib.pyplot as plt

from src.pinn_core          import PINN, PINNTrainer, HeatLoss, WaveLoss, exact_heat, exact_wave
from src.conservation_audit import audit_heat, audit_wave, eval_on_grid
from src.hamiltonian_pinn   import HamiltonianWaveLoss, HPINNTrainer

EPOCHS = 3000   # short for demo; use 15_000 for publication quality

# ─── 1. Heat PINN ────────────────────────────────────────────────────────────
print("\n━━━  HEAT PINN  ━━━")
m_heat = PINN(hidden=48, depth=4)
trainer_heat = PINNTrainer(m_heat, HeatLoss(alpha=0.01),
                            N_f=3000, N_b=100, N_i=100, T=1.0, lr=1e-3)
trainer_heat.train(epochs=EPOCHS, print_every=500)
a_heat = audit_heat(m_heat, alpha=0.01, T=1.0, label="PINN")

# ─── 2. Wave PINN (standard) ─────────────────────────────────────────────────
print("\n━━━  WAVE PINN (standard)  ━━━")
m_wave_std = PINN(hidden=48, depth=4)
trainer_wave_std = PINNTrainer(m_wave_std, WaveLoss(c=1.0),
                                N_f=3000, N_b=100, N_i=100, T=1.0, lr=1e-3)
trainer_wave_std.train(epochs=EPOCHS, print_every=500)
a_wave_std = audit_wave(m_wave_std, c=1.0, T=1.0, label="Standard PINN")

# ─── 3. Wave H-PINN ──────────────────────────────────────────────────────────
print("\n━━━  WAVE H-PINN  ━━━")
m_wave_hpinn = PINN(hidden=48, depth=4)
trainer_wave_hpinn = HPINNTrainer(m_wave_hpinn, HamiltonianWaveLoss(c=1.0, w_H=0.1),
                                   N_f=3000, N_b=100, N_i=100, T=1.0, lr=1e-3)
trainer_wave_hpinn.train(epochs=EPOCHS, print_every=500)
a_wave_hpinn = audit_wave(m_wave_hpinn, c=1.0, T=1.0, label="H-PINN")

# ─── 4. Combined figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("PINN Conservation Audit — Quick Demo", fontsize=14, fontweight='bold')

# Row 0: Heat
g_h = eval_on_grid(m_heat, nx=128, nt=128, T=1.0)
x_h, t_h = g_h["x"], g_h["t"]
X_h, T_h = np.meshgrid(x_h, t_h, indexing='ij')
u_ex_h   = exact_heat(X_h, T_h)

axes[0,0].pcolormesh(t_h, x_h, g_h["u"], cmap="RdBu_r", shading='auto')
axes[0,0].set_title("Heat PINN – u(x,t)"); axes[0,0].set_xlabel("t"); axes[0,0].set_ylabel("x")

axes[0,1].plot(a_heat["t"], a_heat["M"],    label="PINN", color="tab:blue")
axes[0,1].plot(a_heat["t"], a_heat["M_ex"],label="Exact", color="k", ls="--")
axes[0,1].set_title("Mass integral ∫u dx"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[0,2].plot(a_heat["t"], a_heat["E"], color="tab:blue", label="L2 energy")
axes[0,2].set_title("L2 energy E(t) – heat (monotone ↓ expected)")
axes[0,2].legend(); axes[0,2].grid(alpha=0.3)

# Row 1: Wave energy comparison
axes[1,0].plot(a_wave_std["t"],   a_wave_std["E"],   label="Std PINN",  color="tab:blue")
axes[1,0].plot(a_wave_hpinn["t"], a_wave_hpinn["E"], label="H-PINN",    color="tab:orange")
axes[1,0].axhline(a_wave_std["E0_exact"], color="k", ls="--", label="Exact E₀")
axes[1,0].set_title("Wave: Mechanical Energy E(t)"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(a_wave_std["t"],   a_wave_std["drift_ex"]*100,   label="Std PINN",  color="tab:blue")
axes[1,1].plot(a_wave_hpinn["t"], a_wave_hpinn["drift_ex"]*100, label="H-PINN",    color="tab:orange")
axes[1,1].axhline(0, color="k", ls="--")
axes[1,1].set_title("Wave: Energy drift (%)"); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

axes[1,2].semilogy(trainer_wave_std.history["total"],   label="Std PINN total",  color="tab:blue")
axes[1,2].semilogy(trainer_wave_hpinn.history["total"], label="H-PINN total",    color="tab:orange")
axes[1,2].semilogy(trainer_wave_hpinn.history["hamiltonian"], label="H-PINN Hamiltonian loss",
                    color="tab:green", ls="--")
axes[1,2].set_title("Wave: Training loss"); axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/demo_overview.png", dpi=150, bbox_inches='tight')
print("\n✓ Demo figure saved to results/demo_overview.png")

# ─── 5. Print summary ─────────────────────────────────────────────────────────
E0_ex = a_wave_std["E0_exact"]
print(f"\n{'─'*55}")
print(f"  QUICK DEMO SUMMARY  ({EPOCHS} epochs each)")
print(f"{'─'*55}")
print(f"  Heat PINN   | L2 err: {a_heat['err_L2']:.3e}")
print(f"  Wave PINN   | L2 err: {a_wave_std['err_L2']:.3e} | "
      f"max E-drift: {np.max(np.abs(a_wave_std['drift_ex']))*100:.2f}%")
print(f"  Wave H-PINN | L2 err: {a_wave_hpinn['err_L2']:.3e} | "
      f"max E-drift: {np.max(np.abs(a_wave_hpinn['drift_ex']))*100:.2f}%")
print(f"{'─'*55}")
print("Run train_heat.py and train_wave.py for publication-quality results.")