"""
train_heat.py
=============
Train PINN on the 1D heat equation, run conservation audit, save results.

Usage:
    python train_heat.py

Outputs (in results/):
    heat_solution.png       -- spacetime heatmap comparison
    heat_conservation.png   -- mass integral and L2 energy audit
    heat_loss.png           -- training loss curves
    heat_model.pt           -- saved model weights
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.pinn_core        import PINN, PINNTrainer, HeatLoss, exact_heat, DEVICE
from src.conservation_audit import (audit_heat, plot_heat_audit,
                                     plot_solution_comparison)

os.makedirs("results", exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALPHA   = 0.01      # thermal diffusivity
T_END   = 1.0       # time horizon
EPOCHS  = 15_000
LR      = 1e-3
N_F     = 5000      # collocation points
N_B     = 200       # boundary points
N_I     = 200       # initial condition points

# ── Build and train ───────────────────────────────────────────────────────────
print("=" * 60)
print("  HEAT EQUATION PINN")
print("=" * 60)

model   = PINN(hidden=64, depth=5)
loss_fn = HeatLoss(alpha=ALPHA, w_f=1.0, w_b=10.0, w_i=10.0)
trainer = PINNTrainer(model, loss_fn,
                      N_f=N_F, N_b=N_B, N_i=N_I,
                      T=T_END, lr=LR)

trainer.train(epochs=EPOCHS, resample_every=2000, print_every=1000)

# Optional LBFGS fine-tune
trainer.lbfgs_finetune(steps=300)

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "results/heat_model.pt")
print("Model saved to results/heat_model.pt")

# ── Training loss plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for key, color in [("total","k"), ("pde","tab:blue"), ("bc","tab:orange"), ("ic","tab:green")]:
    ax.semilogy(trainer.history[key], label=key, color=color, lw=1.4)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Heat PINN – Training Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/heat_loss.png", dpi=150)
plt.close()
print("Loss curve saved.")

# ── Solution comparison ───────────────────────────────────────────────────────
plot_solution_comparison(model, exact_heat, pde="heat", T=T_END,
                          save_path="results/heat_solution.png", label="PINN")

# ── Conservation audit ────────────────────────────────────────────────────────
audit = audit_heat(model, alpha=ALPHA, T=T_END)
plot_heat_audit(audit, save_path="results/heat_conservation.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  HEAT EQUATION SUMMARY")
print("=" * 50)
print(f"  Relative L2 error (vs exact): {audit['err_L2']:.4e}")
print(f"  Mass at t=0:                  {audit['M'][0]:.6f}")
print(f"  Mass at t=T:                  {audit['M'][-1]:.6f}")
print(f"  Expected M(T)/M(0):           {np.exp(-ALPHA*np.pi**2*T_END):.6f}")
print("=" * 50)