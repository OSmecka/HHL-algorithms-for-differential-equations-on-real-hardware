# HHL Algorithms for Differential Equations on Real Hardware

A Qiskit implementation of the Harrow–Hassidim–Lloyd (HHL) quantum linear-system algorithm applied to a partial differential equation, the 1D wave equation, with a full pipeline from PDE discretization to execution on IBM quantum hardware, plus a study of how each error source converges against its theoretical rate.

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
| `main.py` | Entry point with command-line options: builds the system, runs statevector and Aer, plots the comparison, and optionally runs the convergence study and the IBM hardware run |
| `PDE_preparation.py` | `PDE_preper`: discretizes the wave equation and builds the dilated Hermitian system |
| `HHL_solver.py` | `HHL_solver`: parameter selection, circuit construction, statevector/Aer execution, comparison metrics |
| `HHL_approx_solver.py` | `HHL_reciprocal`: replaces the exact eigenvalue-inversion loop (O(2^nl) multi-controlled rotations) with a piecewise-linear rotation gate, plus a validation routine. Not used by `main.py` |
| `IBM_runner.py` | IBM Quantum connection, QPU time estimate, job submission and reloading |
| `viz.py` | `PDE_postprocessor` and `IBM_QPU_helper`: reshape the solution into u(x,t) and v(x,t), and plot classical vs. quantum results |
| `Convergence.py` | Observed vs. theoretical convergence rates for the discretization, HHL and sampling errors, with a summary plot. Also holds the shared command-line parser used by `main.py` |

## Installation

```bash
git clone https://github.com/OSmecka/HHL-algorithms-for-differential-equations-on-real-hardware.git
cd HHL-algorithms-for-differential-equations-on-real-hardware
pip install numpy scipy matplotlib qiskit qiskit-aer qiskit-ibm-runtime
```

The code uses `QFTGate` and falls back to `QFT` on older Qiskit versions.

## Usage

```bash
python main.py [options]
```

With no options it solves the default problem: a standing wave (`c = 1`, `Lx = 1`, `T = 1`, `Nx = 2`, `Nt = 2`, `u0 = sin(πx/Lx)`, `v0 = 0`, no forcing). It runs the exact statevector solve and the Aer shot-based run, compares both with the classical solution, and plots the result.

| Option | Default | Meaning |
|---|---|---|
| `--c` | 1.0 | wave speed |
| `--Lx` | 1.0 | domain length |
| `--T` | 1.0 | final time |
| `--mode` | 1 | standing-wave mode, `u0 = sin(mode·πx/Lx)` |
| `--nx` | 2 | interior spatial points |
| `--nt` | 2 | time steps (must be a power of 2) |
| `--shots-aer` | 200000 | shots for the local Aer run |
| `--no-plot` | off | skip all figures |
| `--convergence` | off | also run the convergence study (see below) |
| `--skip-quantum` | off | convergence study: classical discretization only |
| `--save-dir` | none | convergence study: directory for PNG figures |
| `--ibm` | off | also connect to IBM Quantum (see below) |
| `--submit` | off | with `--ibm`: actually submit the job (spends QPU time) |
| `--job-id` | none | with `--ibm`: reload a finished job instead of submitting |
| `--shots-ibm` | 10000 | shots for the IBM run |

To solve a different equation (another `v0` or a forcing term `f`), edit the three input functions at the top of `main()`.

### Running on IBM hardware

1. Create an IBM Cloud account and get an API key and instance.
2. Provide the credentials as environment variables so they are never committed:

   ```bash
   export IBM_TOKEN="<your API key>"
   export IBM_INSTANCE="<your instance CRN or name>"
   ```

3. Run:

   ```bash
   python main.py --ibm                  # connect and print the QPU time estimate only
   python main.py --ibm --submit         # submit the job (spends QPU time)
   python main.py --ibm --job-id <id>    # reload a finished job
   ```

Nothing is submitted unless you pass `--submit`. Because a queued job can take a long time, the IBM step runs last, after any convergence study.

`IBM_instance_key` saves your credentials with `QiskitRuntimeService.save_account(..., overwrite=True)`, replacing any previously saved account, and by default picks the least busy operational backend. Pass `least_busy=False` and `backend_name=...` to target a specific device. Circuits are transpiled to the backend's native gates with optimization level 3.

## Convergence analysis

`Convergence.py` measures how each error source in the pipeline shrinks as its resolution parameter improves, and compares the observed order with the theoretical one.

| Error source | Functions | Parameter | Expected behavior |
|---|---|---|---|
| Spatial discretization | `spatial_convergence` | `dx` | second order |
| Temporal discretization | `temporal_convergence` | `dt` | second order (Crank–Nicolson) |
| Total discretization | `joint_convergence` | `dx` with `dt/dx` fixed | second order |
| HHL clock resolution | `hhl_clock_convergence` | clock qubits `nl` | error ∼ κ / 2^nl |
| Trotter approximation | `hhl_trotter_convergence` | repetitions | error ∼ reps^(-order) |
| Finite shots | `sampling_convergence` | shots | error ∼ shots^(-1/2), with a closed-form prediction printed alongside |

The reference problem is the standing wave `u(x,t) = sin(mπx/Lx) cos(mπct/Lx)`, which is the default input of `main.py`. The discretization studies need only numpy, scipy and matplotlib. On the default setup their observed order approaches 2 as the grid is refined.

Run it from `main.py` or on its own:

```bash
python main.py --convergence                   # after the normal run
python Convergence.py                          # everything, with the summary plot
python Convergence.py --skip-quantum           # classical discretization only (no Qiskit needed)
python Convergence.py --save-dir figures       # also save PNGs
python Convergence.py --no-plot                # tables only
```

Each study prints a table with the fitted order. At the end a summary figure shows one log-log panel per study with the theoretical slope as a dashed line. It is drawn by default and can be turned off with `--no-plot`, or with `plot=False` when calling `run_all(...)` from Python. With `--save-dir`, the summary is saved as `convergence_summary.png` together with one PNG per study. The summary can also be drawn from saved results with `plot_summary(results)`.

Notes:
- **Cost.** The HHL studies run many statevector simulations. For the default grid the system needs at least 7 clock qubits, so the clock sweep (from one below the minimum to two above) is the slowest part. The Trotter study uses a fixed 4 clock qubits to stay affordable; `HHL_solver` prints a resolution warning for that, which is harmless there because both runs share the same clock.
- **The clock trend is rough.** Eigenphases fall between clock bins irregularly, so individual points are not monotone and the fitted order is only indicative.
- **Sampling.** By default the shots are drawn from the exact statevector probabilities, which equals a perfect noise-free simulator. The error against the classical solution flattens at the HHL algorithmic floor, which more shots cannot remove.
- **Reference solution.** The exact solution is known only for the standing wave with no forcing. For other inputs only the HHL-versus-classical comparisons apply.

## Interpreting the results

- **Fidelity and relative error** against the classical solution are printed for each backend. Fidelity (normalized overlap) is the natural HHL metric, since HHL returns a quantum state.
- **Signs.** Shot counts only give probabilities, so Aer and IBM results contain magnitudes only. For plotting, `PDE_postprocessor.apply_signs` borrows the signs from the exact statevector solution. This is a diagnostic aid, not an independent quantum result; the sign-free fidelity is reported separately.
- **Noise floor.** On hardware, the post-selection survival rate is compared with a noise floor of about `1 / 2^(nl+1)`. If the survival rate is close to that floor, the output is probably noise rather than a solution, and a warning is printed.
- **Smoothed plots.** With `dense=True`, plots are linearly interpolated onto a finer grid. This is the same data resampled and adds no information.

## Limitations

- HHL circuits grow quickly: the exact controlled-evolution gate has synthesis cost roughly 4^nb in the number of system qubits, and the eigenvalue-inversion step uses O(2^nl) multi-controlled rotations. This is why the default example is a very small grid (`Nx = 2`, `Nt = 2`); larger grids need `use_exact=False` and/or the approximate reciprocal in `HHL_approx_solver.py`.
- The condition number of the dilated matrix grows with grid refinement. `HHL_solver.kappa_scaling` prints this scaling for a list of grid sizes.
- Reading out the full solution vector requires many shots, which offsets much of HHL's theoretical advantage for this use case.

## References

- A. W. Harrow, A. Hassidim, S. Lloyd, *Quantum algorithm for linear systems of equations*, Phys. Rev. Lett. 103, 150502 (2009).
