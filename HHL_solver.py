import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate, RYGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

# QFT moved from a circuit class (QFT) to a gate (QFTGate) in recent Qiskit
# versions. Support both so the class works on either.
try:
    from qiskit.circuit.library import QFTGate

    def _qft(n, inverse=False):
        gate = QFTGate(n)
        return gate.inverse() if inverse else gate
except ImportError:  # older Qiskit
    from qiskit.circuit.library import QFT

    def _qft(n, inverse=False):
        return QFT(n, inverse=inverse)


class HHL_solver:
    """
    HHL linear-system solver for the system produced by `PDE_preper`.

    Intended pipeline
    -----------------
        prep = PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt)
        hhl  = HHL_solver.from_pde_preper(prep)     # runs prep.build()
        out  = hhl.solve()
        u, v = hhl.reshape_solution(out["solution"])

    What it solves
    --------------
    `PDE_preper.build()` returns the Hermitian dilation
        A = [[0, M], [M^T, 0]],   b = [rhs; 0]
    of the (non-symmetric) Crank-Nicolson system M w = rhs. Solving
    A z = b gives z = [0; w], so the PDE solution w is the SECOND HALF of z.
    This class handles that slicing, so `solve()` returns w directly.

    Steps performed
    ---------------
    1. Analyse the spectrum of A and pick the evolution time `t`, the
       rotation constant `C` and the number of clock qubits.
    2. Pad A (identity) and b (zeros) up to a power-of-2 dimension.
    3. Build the HHL circuit: state prep -> QPE -> eigenvalue-inversion
       rotation -> inverse QPE.
    4. Simulate with Statevector, post-select the ancilla on |1> and the
       clock register on |0>, then rescale to the true solution scale.

    Qubit layout in the circuit (little-endian, as in Qiskit)
    ---------------------------------------------------------
        [ sys (nb) | clock (nl) | ancilla (1) ]   -> low bits ... high bit
    """

    def __init__(self, A, b_vec, block_dim=None, n_clock_qubits=None,
                 use_exact=True, trotter_reps=4, trotter_order=2,
                 C_factor=0.9, verbose=True):
        """
        Parameters
        ----------
        A, b_vec       : Hermitian matrix and right-hand side, e.g. from
                         `PDE_preper.hermitian_dilation` / `PDE_preper.build`.
        block_dim      : per-time-step block size (2*Nx) from PDE_preper.
                         Only needed by `reshape_solution`.
        n_clock_qubits : number of QPE clock qubits. If None, the smallest
                         number that resolves the smallest eigenvalue is used
                         (more clock qubits = more accurate, but a bigger
                         simulation).
        use_exact      : True  -> exact controlled exp(iAt) (HamiltonianGate),
                                  no Trotter error, fine for small systems.
                         False -> Suzuki-Trotter approximation.
        trotter_reps, trotter_order : Trotter settings (use_exact=False only).
        C_factor       : rotation constant C = C_factor * lambda_min. Must be
                         <= lambda_min so that C / lambda <= 1 for every
                         eigenvalue.
        verbose        : print parameter diagnostics.
        """
        A = np.asarray(A)
        b_vec = np.asarray(b_vec)
        assert np.allclose(A, A.conj().T), "A must be Hermitian (use the dilation)"

        self.A_raw, self.b_raw = A, b_vec
        self.N_raw = A.shape[0]            # size of the dilated system
        self.block_dim = block_dim
        self.use_exact = use_exact
        self.trotter_reps = trotter_reps
        self.trotter_order = trotter_order
        self.verbose = verbose
        self.prep = None                   # set by from_pde_preper
        self._A_op = None                  # Pauli form, built lazily

        # Spectrum-based parameters, computed on the UNPADDED matrix so the
        # artificial padding eigenvalues (=1) do not change t or C.
        self.params = self.hhl_parameters(A, b_vec, C_factor)

        # Number of clock qubits
        self.nl_min = self.min_clock_qubits(self.params["phi_min"])
        self.nl = n_clock_qubits if n_clock_qubits is not None else self.nl_min
        if self.nl < self.nl_min:
            print(f"WARNING: nl={self.nl} < minimum {self.nl_min} needed to "
                  f"resolve lambda_min; the solution will be inaccurate.")

        # Pad to a power of 2 so the system maps onto qubits
        self.A, self.b, _ = self.pad_to_pow2(A, b_vec)
        self.nb = int(np.round(np.log2(self.A.shape[0])))

        self.circuit = None
        if verbose:
            self.print_parameters()

    @classmethod
    def from_pde_preper(cls, prep, **kwargs):
        """
        Build directly from a `PDE_preper` instance: runs `prep.build()`
        (discretize + Hermitian dilation) and forwards `block_dim`.
        """
        A, b_vec, x_grid, t_grid, block_dim = prep.build()
        obj = cls(A, b_vec, block_dim=block_dim, **kwargs)
        obj.prep = prep
        obj.x_grid, obj.t_grid = x_grid, t_grid
        return obj

    # ------------------------------------------------------------------
    # 1. Parameter analysis
    # ------------------------------------------------------------------
    @staticmethod
    def hhl_parameters(A, b_vec, C_factor=0.9):
        """
        Compute the quantities HHL needs from the spectrum of A.

        Returns a dict with:
          norm_b, eigvals, eigvecs : ||b|| and eigendecomposition of A
          lam_min, lam_max         : smallest / largest |nonzero eigenvalue|
          kappa                    : condition number lam_max / lam_min
          t                        : evolution time, pi / lam_max. This puts
                                     phi = lam*t/(2 pi) inside (-1/2, 1/2],
                                     so negative eigenvalues are encoded as
                                     two's complement in the clock register.
          phi_min                  : smallest phase, lam_min * t / (2 pi)
          C                        : rotation constant, C_factor * lam_min
        """
        norm_b = np.linalg.norm(b_vec)
        eigvals, eigvecs = np.linalg.eigh(A)
        nonzero = np.abs(eigvals) > 1e-10
        lam_min = np.min(np.abs(eigvals[nonzero]))
        lam_max = np.max(np.abs(eigvals[nonzero]))
        t = np.pi / lam_max
        return dict(norm_b=norm_b, eigvals=eigvals, eigvecs=eigvecs,
                    lam_min=lam_min, lam_max=lam_max, kappa=lam_max / lam_min,
                    t=t, phi_min=lam_min * t / (2 * np.pi),
                    C=lam_min * C_factor)

    @staticmethod
    def min_clock_qubits(phi_min):
        """
        Smallest nl with phi_min >= 1 / 2^nl, i.e. the clock register can
        distinguish the smallest eigenvalue from zero.
        (Closed form of the 'increase nl until the condition holds' loop.)
        """
        return max(1, int(np.ceil(np.log2(1.0 / phi_min))))

    def print_parameters(self):
        p = self.params
        resolvable = 1 / 2 ** self.nl
        status = "OK" if p["phi_min"] >= resolvable else "TOO SMALL"
        print(f"--- HHL parameters (system size {self.N_raw}, padded to {self.A.shape[0]}) ---")
        print(f"kappa(A)      = {p['kappa']:.2f}")
        print(f"lambda_min    = {p['lam_min']:.6f}")
        print(f"lambda_max    = {p['lam_max']:.4f}")
        print(f"t (evolution) = {p['t']:.5f}")
        print(f"C             = {p['C']:.6f}")
        print(f"phi_min       = {p['phi_min']:.5f}  vs 1/2^nl = {resolvable:.5f}  [{status}]")
        print(f"clock qubits  = {self.nl} (minimum {self.nl_min}), system qubits = {self.nb}")

    @staticmethod
    def kappa_scaling(make_prep, sizes):
        """
        Print how the condition number of the dilated matrix grows with grid
        refinement. `make_prep(Nx, Nt)` must return a PDE_preper.
        Example:
            HHL_solver.kappa_scaling(lambda Nx, Nt: PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt),
                                     [(4, 4), (6, 8), (8, 8)])
        """
        print("kappa(A) scaling:")
        for Nx, Nt in sizes:
            At, _, _, _, _ = make_prep(Nx, Nt).build()
            print(f"  Nx={Nx:3d} Nt={Nt:3d}  size={At.shape[0]:4d}  kappa={np.linalg.cond(At):.2f}")

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def pad_to_pow2(A, b_vec):
        """
        Pad A with an identity block and b with zeros so the dimension is a
        power of 2. Because A_pad is block diagonal and b has no component in
        the padded part, the padding does not affect the solution.

        Returns A_pad, b_pad, original_dimension.
        """
        N = A.shape[0]
        n = int(np.ceil(np.log2(N)))
        dim = 2 ** n
        if dim == N:
            return A, b_vec, N
        A_pad = np.eye(dim, dtype=complex)
        A_pad[:N, :N] = A
        b_pad = np.zeros(dim, dtype=complex)
        b_pad[:N] = b_vec
        return A_pad, b_pad, N

    def build_pauli_hamiltonian(self):
        """Pauli decomposition of the padded A (needed for Trotterization)."""
        if self._A_op is None:
            self._A_op = SparsePauliOp.from_operator(self.A).simplify()
            if self.verbose:
                print(f"Pauli decomposition: {len(self._A_op)} terms "
                      f"(of {4 ** self.nb} possible)")
        return self._A_op

    # ------------------------------------------------------------------
    # 3. Circuit building blocks
    # ------------------------------------------------------------------
    def controlled_evolution_power(self, power):
        """
        Controlled gate implementing (e^{iAt})^power = e^{iA t power}.

        use_exact=True : exact matrix exponential (HamiltonianGate). No
                         Trotter error, but synthesis cost grows ~4^nb, so
                         only practical for small system registers.
        use_exact=False: Suzuki-Trotter approximation of the same unitary.
        """
        t = self.params["t"]
        if self.use_exact:
            # HamiltonianGate implements exp(-i * data * time), so
            # time = -t*power gives exp(+i * A * t * power).
            gate = HamiltonianGate(self.A, time=-t * power)
        else:
            gate = PauliEvolutionGate(
                self.build_pauli_hamiltonian(),
                time=t * power,
                synthesis=SuzukiTrotter(order=self.trotter_order,
                                        reps=self.trotter_reps * power),
            )
        return gate.control(1)

    def qpe(self, qc, qb, ql):
        """Quantum phase estimation of e^{iAt} on the clock register."""
        qc.h(ql[:])                               # superposition on the clock
        for j in range(len(ql)):                  # controlled-U^{2^j}, LSB first
            qc.append(self.controlled_evolution_power(2 ** j), [ql[j]] + qb[:])
        qc.append(_qft(len(ql), inverse=True), ql[:])
        return qc

    def inverse_qpe(self, qc, qb, ql):
        """Exact inverse of `qpe` (uncomputes the clock register)."""
        qc.append(_qft(len(ql), inverse=False), ql[:])
        for j in reversed(range(len(ql))):
            qc.append(self.controlled_evolution_power(2 ** j).inverse(), [ql[j]] + qb[:])
        qc.h(ql[:])
        return qc

    def eigenvalue_inversion(self, qc, ql, qa):
        """
        For every clock value k != 0, rotate the ancilla by
        2*arcsin(C / lambda_k), where lambda_k is the eigenvalue encoded by k
        (two's complement, so the upper half of the range is negative).
        Each rotation is controlled on the clock register being exactly |k>.
        """
        nl = self.nl
        t, C = self.params["t"], self.params["C"]
        for raw in range(1, 2 ** nl):
            k_signed = raw - 2 ** nl if raw >= 2 ** (nl - 1) else raw
            phi = k_signed / 2 ** nl
            lam_est = phi * 2 * np.pi / t
            ratio = C / lam_est
            # |ratio| > 1 can only happen for under-resolved eigenvalues;
            # saturate the rotation instead of calling arcsin out of domain.
            angle = np.sign(ratio) * np.pi if abs(ratio) > 1 else 2 * np.arcsin(ratio)

            # Flip the clock qubits that are 0 in |k> so an all-ones control
            # fires exactly when the clock equals k.
            bits = format(raw, f"0{nl}b")         # MSB first
            flip = [ql[nl - 1 - i] for i, bit in enumerate(bits) if bit == "0"]
            if flip:
                qc.x(flip)
            qc.append(RYGate(angle).control(nl), ql[:] + [qa[0]])
            if flip:
                qc.x(flip)
        return qc

    def build_circuit(self):
        """Assemble and store the full HHL circuit."""
        qb = QuantumRegister(self.nb, "sys")
        ql = QuantumRegister(self.nl, "clock")
        qa = QuantumRegister(1, "anc")
        qc = QuantumCircuit(qb, ql, qa)

        qc.prepare_state(self.b / np.linalg.norm(self.b), qb[:])   # |b>
        self.qpe(qc, qb, ql)
        self.eigenvalue_inversion(qc, ql, qa)
        self.inverse_qpe(qc, qb, ql)

        self.circuit = qc
        return qc

    # ------------------------------------------------------------------
    # 4. Solving and post-processing
    # ------------------------------------------------------------------
    def extract_solution_statevector(self, qc=None):
        """
        Simulate the circuit and post-select.

        Keeps amplitudes where ancilla = 1 (rotation succeeded) and
        clock = 0 (QPE uncomputed). Returns
            amp          : un-normalised amplitudes on the system register
            success_prob : total probability of ancilla = 1
        """
        qc = qc if qc is not None else self.circuit
        sv = Statevector(qc)
        dim = 2 ** self.nb
        amp = np.zeros(dim, dtype=complex)
        success_prob = 0.0
        for idx, val in enumerate(sv.data):
            if abs(val) < 1e-12:
                continue
            anc_bit = (idx >> (self.nb + self.nl)) & 1
            if anc_bit == 1:
                success_prob += abs(val) ** 2
            clock_val = (idx >> self.nb) & (2 ** self.nl - 1)
            if anc_bit == 1 and clock_val == 0:
                amp[idx & (dim - 1)] = val
        return amp, success_prob

    def solve(self):
        """
        Run HHL end to end.

        The post-selected amplitudes equal C * A^{-1} b / ||b||, so
        multiplying by ||b|| / C restores the true scale. The result is then
        sliced to remove padding and the dilation's zero half.

        Returns a dict with:
          solution     : PDE solution w = [w^1; ...; w^Nt]  (length 2*Nx*Nt)
          success_prob : ancilla success probability
          amp          : raw post-selected amplitudes (padded dilated space)
        """
        qc = self.build_circuit()
        amp, success_prob = self.extract_solution_statevector(qc)

        solution = self._amplitudes_to_solution(amp)
        return dict(solution=solution, success_prob=success_prob, amp=amp)

    def _amplitudes_to_solution(self, amp):
        """
        Post-selected amplitudes equal C * A^{-1} b / ||b||. Multiply by
        ||b|| / C to restore the true scale, drop the padding, and keep the
        second half of the dilated vector z = [0; w] (= the PDE solution w).
        """
        z = amp * self.params["norm_b"] / self.params["C"]
        z = z[: self.N_raw]
        return z[self.N_raw // 2:]

    def run_aer(self, shots=200000, noise_model=None, optimization_level=1, seed=None):
        """
        Shot-based run on a local AerSimulator (optionally with noise).

        Measures every qubit, post-selects on ancilla = 1 and clock = 0, and
        turns the surviving counts into |amplitudes|. Shot counts only give
        probabilities, so the SIGNS of the solution entries are lost: the
        result is |w|, not w. (The exact Statevector route in `solve()`
        keeps signs.)

        Parameters
        ----------
        shots              : number of shots.
        noise_model        : optional qiskit_aer NoiseModel.
        optimization_level : transpiler level (3 can be very slow for the
                             large controlled unitaries; 1 is a good default).
        seed               : seed for the simulator and transpiler.

        Returns a dict with:
          sys_counts   : post-selected counts per system basis state
          n_post       : number of shots that survived post-selection
          shots        : total shots
          solution_abs : |w| estimated from the counts (length 2*Nx*Nt)
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        qc = self.circuit if self.circuit is not None else self.build_circuit()
        backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)

        qc_meas = qc.copy()
        qc_meas.measure_all()
        tqc = transpile(qc_meas, backend, optimization_level=optimization_level,
                        seed_transpiler=seed)
        counts = backend.run(tqc, shots=shots).result().get_counts()

        nb, nl = self.nb, self.nl
        sys_counts = np.zeros(2 ** nb)
        for bitstring, n in counts.items():
            # Bitstring is MSB first: [anc | clock | sys]. Qubit i = bit i of
            # the integer, matching the Statevector index layout.
            idx = int(bitstring.replace(" ", ""), 2)
            anc_bit = (idx >> (nb + nl)) & 1
            clock_val = (idx >> nb) & (2 ** nl - 1)
            if anc_bit == 1 and clock_val == 0:
                sys_counts[idx & (2 ** nb - 1)] += n

        n_post = int(sys_counts.sum())
        if self.verbose:
            if n_post == 0:
                print("Aer: no shots survived post-selection.")
            else:
                print(f"Aer: {n_post}/{shots} shots survived "
                      f"({100 * n_post / shots:.2f}%)")

        # P(anc=1, clock=0, sys=j) = |amp_j|^2, so |amp_j| = sqrt(count_j/shots)
        # (note: divided by ALL shots, not just the post-selected ones).
        amp_abs = np.sqrt(sys_counts / shots)
        return dict(sys_counts=sys_counts, n_post=n_post, shots=shots,
                    solution_abs=np.abs(self._amplitudes_to_solution(amp_abs)))

    def classical_solution(self):
        """Reference solution w from a direct classical solve of A z = b."""
        z = np.linalg.solve(self.A_raw, self.b_raw)
        return z[self.N_raw // 2:]

    def compare(self, solution, magnitude=False):
        """
        Compare an HHL solution with the classical one. Returns
        (fidelity, relative_error). Fidelity (normalised overlap) is the
        natural HHL metric, since HHL fundamentally returns a quantum state.

        magnitude=True compares against |classical| instead; use it for the
        sign-free result of `run_aer`.
        """
        ref = self.classical_solution()
        if magnitude:
            ref = np.abs(ref)
        fid = abs(np.vdot(ref, solution)) / (np.linalg.norm(ref) * np.linalg.norm(solution))
        rel = np.linalg.norm(solution - ref) / np.linalg.norm(ref)
        return fid, rel

    def reshape_solution(self, w):
        """
        Split the stacked solution into u and v arrays of shape (Nt, Nx):
        u[n] = displacement at time t_{n+1}, v[n] = velocity at that time.
        Requires `block_dim` (set automatically by `from_pde_preper`).
        """
        assert self.block_dim is not None, "block_dim is required to reshape"
        Nx = self.block_dim // 2
        blocks = np.real_if_close(w, tol=1e6).reshape(-1, self.block_dim)
        return blocks[:, :Nx], blocks[:, Nx:]
