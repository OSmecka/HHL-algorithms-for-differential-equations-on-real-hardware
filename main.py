import numpy as np
 
from PDE_preparation import PDE_preper   
from HHL_solver import HHL_solver  
from viz.pyimport PDE_postprocessor

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

# --- 3. Post-processing: u(x,t), v(x,t) and classical-vs-quantum plots ---

post = PDE_postprocessor(prep)

w_classical = hhl.classical_solution()
u_cl, v_cl = post.reshape_to_uv(w_classical)

w_aer = post.apply_signs(aer["solution_abs"], out["solution"])
u_q, v_q = post.reshape_to_uv(w_aer)

print("Aer (signed) vs classical:", post.error_summary(u_cl, u_q))

u_of_xt, v_of_xt = post.make_uv_of_xt(u_q, v_q)
post.plot_comparison(u_cl, u_q, dense=True)


# --- 4. IBM quantum hardwere run ---

service, backend, isa_pm = IBM_instance_key(token, instance, opl=3)

Sampler_time_estimate(hhl.circuit, backend, "HHL_wave", SHOTS=10000)

job, job_id = Sampler_RUN(hhl.circuit, backend, isa_pm, service, "HHL_wave", SHOTS=10000, SUBMIT_JOB=False)














