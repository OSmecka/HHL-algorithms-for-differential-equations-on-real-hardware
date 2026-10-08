import numpy as np
 
from PDE_preparation import PDE_preper   # file containing the PDE_preper class
from HHL_solver import HHL_solver        # file containing the HHL_solver class
 
# --- inputs ---
c = 1.0
Lx, Nx = 1.0, 2
T, Nt = 1.0, 2
u0_func = lambda x: np.sin(np.pi * x / Lx)
v0_func = lambda x: np.zeros_like(x)
f_func = lambda x, t: np.zeros_like(x)      # no forcing: standing wave
 
prep = PDE_preper(c, u0_func, v0_func, f_func, Lx, Nx, T, Nt)
hhl = HHL_solver.from_pde_preper(prep)
 
# --- 1. Exact statevector run (keeps signs) ---

out = hhl.solve()
fid, rel = hhl.compare(out["solution"])
u, v = hhl.reshape_solution(out["solution"])
print(f"[Statevector] success probability = {out['success_prob']:.4f}")
print(f"[Statevector] fidelity = {fid:.6f}, relative error = {rel:.4f}")
print("u (rows = time steps):\n", u)
 
# --- 2. Shot-based local Aer run (magnitudes only) ---

SHOTS_AER = 200000
aer = hhl.run_aer(shots=SHOTS_AER)
fid_a, rel_a = hhl.compare(aer["solution_abs"], magnitude=True)
u_abs, v_abs = hhl.reshape_solution(aer["solution_abs"])
print(f"[Aer] fidelity vs |classical| = {fid_a:.6f}, relative error = {rel_a:.4f}")
print("|u| (rows = time steps):\n", u_abs)
