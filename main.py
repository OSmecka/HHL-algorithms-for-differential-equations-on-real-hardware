import numpy as np
 
from PDE_preparation import PDE_preper   
from HHL_solver import HHL_solver  
from viz import PDE_postprocessor, IBM_QPU_helper

import os
import numpy as np

from PDE_preparation import PDE_preper
from HHL_solver import HHL_solver
from viz import PDE_postprocessor, IBM_QPU_helper
from IBM_runner import IBM_instance_key, Sampler_time_estimate, Sampler_RUN


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

# --- 3. Post-processing: u(x,t), v(x,t) and classical-vs-Aer plots ---
post = PDE_postprocessor(prep)
qpu = IBM_QPU_helper(hhl, post)      # counts / IBM-hardware helpers

w_classical = hhl.classical_solution()
u_cl, v_cl = post.reshape_to_uv(w_classical)

# Counts give magnitudes only; borrow signs from the exact statevector
# solution (diagnostic aid, see PDE_postprocessor.apply_signs).
w_aer = post.apply_signs(aer["solution_abs"], out["solution"])
u_q, v_q = post.reshape_to_uv(w_aer)

print("Aer (signed) vs classical:", post.error_summary(u_cl, u_q))

u_of_xt, v_of_xt = post.make_uv_of_xt(u_q, v_q)
post.plot_comparison(u_cl, u_q, dense=True)

# --- 4. IBM quantum hardware run ---
# Read credentials from environment variables instead of hard-coding them.
token = os.environ["IBM_TOKEN"]
instance = os.environ["IBM_INSTANCE"]

SHOTS_IBM = 10000
SUBMIT_JOB = False      # True = actually submit (spends QPU time)
JOB_ID = None           # set to a saved job id to reload a finished job

service, backend, isa_pm = IBM_instance_key(token, instance, opl=3)

Sampler_time_estimate(hhl.circuit, backend, "HHL_wave", SHOTS=SHOTS_IBM)

job, job_id = Sampler_RUN(hhl.circuit, backend, isa_pm, service, "HHL_wave",
                          SHOTS=SHOTS_IBM, SUBMIT_JOB=SUBMIT_JOB, JOB_ID=JOB_ID)

# --- 5. Visualize the IBM hardware result ---
if job is not None:
    counts = qpu.get_counts_from_job(job)   # blocks until the job is done
    ibm = qpu.counts_to_solution_abs(counts)
 
    survival = ibm["n_post"] / ibm["shots"]
    noise_floor = 1 / 2 ** (hhl.nl + 1)
    print(f"[IBM] {ibm['n_post']}/{ibm['shots']} shots survived post-selection")
    print(f"[IBM] survival rate = {survival:.4f} "
          f"(noise floor ~ {noise_floor:.4f}, ideal = {out['success_prob']:.4f})")
    if survival < 3 * noise_floor:
        print("[IBM] WARNING: survival rate is near the noise floor; the "
              "result below is probably noise, not a solution.")

    if ibm["n_post"] > 0:
        fid_i, rel_i = hhl.compare(ibm["solution_abs"], magnitude=True)
        print(f"[IBM] fidelity vs |classical| = {fid_i:.6f}, relative error = {rel_i:.4f}")

        w_ibm = post.apply_signs(ibm["solution_abs"], out["solution"])
        u_ibm, v_ibm = post.reshape_to_uv(w_ibm)
        print("IBM (signed) vs classical:", post.error_summary(u_cl, u_ibm))

        # Classical vs Aer vs IBM, with difference maps
        qpu.plot_runs(u_cl, {"Aer": u_q, "IBM hardware": u_ibm}, dense=True)
 








