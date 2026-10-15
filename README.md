# HHL Algorithms for Differential Equations on Real Hardware

A Qiskit implementation of the Harrow–Hassidim–Lloyd (HHL) quantum linear-system algorithm applied to a partial differential equation, the 1D wave equation, with a full pipeline from PDE discretization to execution on IBM quantum hardware.

The code discretizes the PDE into one large linear system, solves it with HHL on three backends (exact statevector, shot-based Aer simulator, IBM QPU), and compares each result against a classical solve.

## How it works

**Problem.** The 1D wave equation with homogeneous Dirichlet boundary conditions:

```
u_tt = c² u_xx + f(x, t),   x ∈ [0, Lx], t ∈ [0, T]
u(0, t) = u(Lx, t) = 0
u(x, 0) = u0(x),   u_t(x, 0) = v0(x)
```

**1. Discretization (`PDE_preparation.py`)**
- Space: central finite differences on `Nx` interior points.
- Time: the equation is rewritten as a first-order system in `w = [u; v]` with `v = du/dt`, then stepped with Crank–Nicolson.
- All `Nt` time steps are stacked into a single block-bidiagonal system `M W = rhs` (a "time-all-at-once" formulation), which is what a quantum linear solver needs. `Nt` must be a power of 2.

**2. Hermitian dilation.** `M` is not symmetric, but HHL needs a Hermitian matrix. The system is embedded as

```
A = [[0, M], [Mᵀ, 0]],   b = [rhs; 0]
```

so that `A z = b` gives `z = [0; W]` and the PDE solution is the second half of `z`.

**3. HHL circuit (`HHL_solver.py`)**
- `A` is padded with an identity block to a power-of-2 dimension.
- The circuit is: state preparation of `|b⟩` → quantum phase estimation → eigenvalue-inversion rotation → inverse QPE.
- The evolution time is `t = π / λ_max`, so negative eigenvalues are encoded in two's complement in the clock register.
- Controlled `e^{iAt}` is implemented either exactly (`HamiltonianGate`, `use_exact=True`) or with a Suzuki–Trotter approximation (`use_exact=False`).
- The solution is obtained by post-selecting the ancilla on `|1⟩` and the clock register on `|0⟩`, then rescaling by `‖b‖ / C`.
- The number of clock qubits defaults to the minimum needed to resolve the smallest eigenvalue; the solver prints κ(A), λ_min, λ_max, `t`, `C` and the qubit counts.
- Qubit layout (little-endian): `[ sys | clock | ancilla ]`.

**4. Execution backends**

| Backend | Method | Returns |
|---|---|---|
| Statevector | `hhl.solve()` | exact amplitudes, **with signs** |
| Aer (local, optional noise model) | `hhl.run_aer(shots=...)` | magnitudes `\|w\|` only |
| IBM QPU | `IBM_runner.py` (Qiskit Runtime `SamplerV2`) | magnitudes `\|w\|` only |

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | End-to-end example: builds the system, runs statevector, Aer and IBM, and plots the comparison |
| `PDE_preparation.py` | `PDE_preper`: discretizes the wave equation and builds the dilated Hermitian system |
| `HHL_solver.py` | `HHL_solver`: parameter selection, circuit construction, statevector/Aer execution, comparison metrics |
| `HHL_approx_solver.py` | `HHL_reciprocal`: replaces the exact eigenvalue-inversion loop (O(2^nl) multi-controlled rotations) with a piecewise-linear rotation gate, plus a validation routine. Not used by `main.py` |
| `IBM_runner.py` | IBM Quantum connection, QPU time estimate, job submission and reloading |
| `viz.py` | `PDE_postprocessor` and `IBM_QPU_helper`: reshape the solution into u(x,t) and v(x,t), and plot classical vs. quantum results |

## Installation

```bash
git clone https://github.com/OSmecka/HHL-algorithms-for-differential-equations-on-real-hardware.git
cd HHL-algorithms-for-differential-equations-on-real-hardware
pip install numpy scipy matplotlib qiskit qiskit-aer qiskit-ibm-runtime
```

The code uses `QFTGate` and falls back to `QFT` on older Qiskit versions.

## Usage

Edit the inputs at the top of `main.py`:

```python
c = 1.0
Lx, Nx = 1.0, 2            # domain length, interior grid points
T, Nt = 1.0, 2             # final time, time steps (power of 2)
u0_func = lambda x: np.sin(np.pi * x / Lx)
v0_func = lambda x: np.zeros_like(x)
f_func  = lambda x, t: np.zeros_like(x)   # no forcing: standing wave
```

Then run:

```bash
python main.py
```

The script runs the exact statevector solve, the Aer shot-based run, and the classical comparison with plots. It then connects to IBM Quantum and prints a QPU time estimate.

### Running on IBM hardware

1. Create an IBM Cloud account and get an API key and instance.
2. Set `token` and `instance` in `main.py`. To avoid committing credentials, read them from environment variables instead, for example `token = os.environ["IBM_TOKEN"]`.
3. Set `SUBMIT_JOB = True` to submit. It is `False` by default so a stray run never spends QPU time.
4. To reload a finished job instead of resubmitting, set `JOB_ID` to its id.

`IBM_instance_key` saves your credentials with `QiskitRuntimeService.save_account(..., overwrite=True)`, replacing any previously saved account, and by default picks the least busy operational backend. Pass `least_busy=False` and `backend_name=...` to target a specific device. Circuits are transpiled to the backend's native gates with the optimization level set by `opl` (3 in `main.py`).

## Interpreting the results

- **Fidelity and relative error** against the classical solution are printed for each backend. Fidelity (normalized overlap) is the natural HHL metric, since HHL returns a quantum state.
- **Signs.** Shot counts only give probabilities, so Aer and IBM results contain magnitudes only. For plotting, `PDE_postprocessor.apply_signs` borrows the signs from the exact statevector solution. This is a diagnostic aid, not an independent quantum result; the sign-free fidelity is reported separately.
- **Noise floor.** On hardware, `main.py` reports the post-selection survival rate and compares it with a noise floor of about `1 / 2^(nl+1)`. If the survival rate is close to that floor, the output is probably noise rather than a solution, and a warning is printed.
- **Smoothed plots.** With `dense=True`, plots are linearly interpolated onto a finer grid. This is the same data resampled and adds no information.

## Limitations

- HHL circuits grow quickly: the exact controlled-evolution gate has synthesis cost roughly 4^nb in the number of system qubits, and the eigenvalue-inversion step uses O(2^nl) multi-controlled rotations. This is why the default example is a very small grid (`Nx = 2`, `Nt = 2`); larger grids need `use_exact=False` and/or the approximate reciprocal in `HHL_approx_solver.py`.
- The condition number of the dilated matrix grows with grid refinement. `HHL_solver.kappa_scaling` prints this scaling for a list of grid sizes.
- Reading out the full solution vector requires many shots, which offsets much of HHL's theoretical advantage for this use case.

## References

- A. W. Harrow, A. Hassidim, S. Lloyd, *Quantum algorithm for linear systems of equations*, Phys. Rev. Lett. 103, 150502 (2009).
