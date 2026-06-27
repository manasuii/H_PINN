"""
train_wave.py
=============
Train both standard PINN and Hamiltonian PINN on the 1D wave equation.
Run conservation audits and compare energy drift.

Usage:
    python train_wave.py

Outputs (in results/):
    wave_std_solution.png     -- standard PINN spacetime comparison
    wave_hpinn_solution.png   -- H-PINN spacetime comparison
    wave_energy_audit.png     -- energy conservation comparison
    wave_loss_curves.png      -- loss curve comparison
    wave_summary.txt          -- numerical summary table
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.pinn_core          import PINN, PINNTrainer, WaveLoss, exact_wave, DEVICE
from src.conservation_audit import (audit_wave, plot_wave_audit,
                                     plot_solution_comparison)
from src.hamiltonian_pinn   import (HamiltonianWaveLoss, HPINNTrainer,
                                     plot_loss_curves)

os.makedirs("results", exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
C       = 1.0       # wave speed
T_END   = 1.0
EPOCHS  = 15_000
LR      = 1e-3
N_F     = 5000
N_B     = 200
N_I     = 200

# ═══════════════════════════════════════════════════════════════════════════════
# A.  STANDARD PINN
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  WAVE EQUATION — STANDARD PINN")
print("=" * 60)

model_std   = PINN(hidden=64, depth=5)
loss_std    = WaveLoss(c=C, w_f=1.0, w_b=10.0, w_i=10.0)
trainer_std = PINNTrainer(model_std, loss_std,
                           N_f=N_F, N_b=N_B, N_i=N_I,
                           T=T_END, lr=LR)

trainer_std.train(epochs=EPOCHS, resample_every=2000, print_every=1000)
trainer_std.lbfgs_finetune(steps=300)
torch.save(model_std.state_dict(), "results/wave_std_model.pt")

# ═══════════════════════════════════════════════════════════════════════════════
# B.  HAMILTONIAN PINN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  WAVE EQUATION — HAMILTONIAN PINN (H-PINN)")
print("=" * 60)

model_hpinn   = PINN(hidden=64, depth=5)
loss_hpinn    = HamiltonianWaveLoss(c=C, w_f=1.0, w_b=10.0, w_i=10.0, w_H=0.1)
trainer_hpinn = HPINNTrainer(model_hpinn, loss_hpinn,
                              N_f=N_F, N_b=N_B, N_i=N_I,
                              T=T_END, lr=LR)

trainer_hpinn.train(epochs=EPOCHS, resample_every=2000, print_every=1000)
torch.save(model_hpinn.state_dict(), "results/wave_hpinn_model.pt")

# ═══════════════════════════════════════════════════════════════════════════════
# C.  CONSERVATION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\nRunning conservation audits …")
audit_std   = audit_wave(model_std,   c=C, T=T_END, label="Standard PINN")
audit_hpinn = audit_wave(model_hpinn, c=C, T=T_END, label="H-PINN")

# ═══════════════════════════════════════════════════════════════════════════════
# D.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
exact_fn = lambda X, T: exact_wave(X, T, C)

plot_solution_comparison(model_std, exact_fn, pde="wave", T=T_END,
                          save_path="results/wave_std_solution.png",
                          label="Standard PINN")

plot_solution_comparison(model_hpinn, exact_fn, pde="wave", T=T_END,
                          save_path="results/wave_hpinn_solution.png",
                          label="H-PINN")

plot_wave_audit(audit_std, audit_hpinn,
                save_path="results/wave_energy_audit.png")

plot_loss_curves(trainer_std.history, trainer_hpinn.history,
                 save_path="results/wave_loss_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# E.  NUMERICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
E0_ex = audit_std["E0_exact"]

summary = f"""
{'='*60}
  WAVE EQUATION — CONSERVATION AUDIT SUMMARY
{'='*60}
  Exact initial energy E₀ = π²c²/2 = {E0_ex:.6f}

  Standard PINN
  ─────────────
  Relative L2 error vs exact:        {audit_std['err_L2']:.4e}
  E(0) predicted:                    {audit_std['E'][0]:.6f}
  Max |E(t)-E_exact|/E_exact (%):    {np.max(np.abs(audit_std['drift_ex']))*100:.3f}%
  Energy drift at t=T:               {audit_std['drift_ex'][-1]*100:.3f}%

  Hamiltonian PINN (H-PINN)
  ─────────────────────────
  Relative L2 error vs exact:        {audit_hpinn['err_L2']:.4e}
  E(0) predicted:                    {audit_hpinn['E'][0]:.6f}
  Max |E(t)-E_exact|/E_exact (%):    {np.max(np.abs(audit_hpinn['drift_ex']))*100:.3f}%
  Energy drift at t=T:               {audit_hpinn['drift_ex'][-1]*100:.3f}%

  Improvement in max energy drift:
  {np.max(np.abs(audit_std['drift_ex']))/max(np.max(np.abs(audit_hpinn['drift_ex'])), 1e-12):.2f}× reduction
{'='*60}
"""
print(summary)
with open("results/wave_summary.txt", "w") as f:
    f.write(summary)
print("Summary saved to results/wave_summary.txt")