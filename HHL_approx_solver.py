import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PiecewiseLinearPauliRotations, RYGate
from qiskit.quantum_info import Statevector


class HHL_reciprocal:
    """
    Eigenvalue-inversion (reciprocal) rotation for HHL, approximated with a
    piecewise-linear rotation gate.

    Exact HHL applies, for every clock value k, a rotation RY(angle(k)) on
    the ancilla controlled on the clock register being exactly |k>. That
    needs O(2^nl) multi-controlled gates. Here angle(k) is replaced by a
    piecewise-linear function of k, which `PiecewiseLinearPauliRotations`
    implements with O(nl * num_breakpoints) gates.

    Typical usage
    -------------
        rec  = HHL_reciprocal(nl, C, t, num_breakpoints=12)
        # or: rec = HHL_reciprocal.from_hhl(hhl, num_breakpoints=12)
        gate = rec.build_reciprocal_gate()     # clock + target (+ ancillas)
        rec.validate_reciprocal_gate()         # exact vs piecewise fidelity

    Qubit order of the gate: [clock (nl) | target (1) | work ancillas ...].
    The work ancillas are returned to |0> by the gate and must be provided
    by the caller (their number is `rec.num_work_ancillas` after building).
    """

    def __init__(self, nl, C, t, num_breakpoints=12, concentrate_near_zero=True,
                 bit_reversed=False, verbose=True):
        """
        Parameters
        ----------
        nl    : int
            Number of clock qubits.
        C     : float
            HHL rotation constant (C <= smallest |eigenvalue|), e.g.
            `hhl.params["C"]`.
        t     : float
            QPE evolution time, e.g. `hhl.params["t"]`.
        num_breakpoints : int
            Number of breakpoints of the piecewise-linear fit. More
            breakpoints = better fit, more gates.
        concentrate_near_zero : bool
            Place more breakpoints at small register values, where the angle
            changes fastest (the region of the smallest eigenvalues). A
            uniform grid would waste breakpoints on the flat high-k tail.
        bit_reversed : bool
            Read the clock register with its bits reversed. This was needed
            in the original setup (`diagnose_qpe_bit_order` reported
            reversed=True). The manual QPE in `HHL_solver` (inverse QFT WITH
            swaps) gives the register in normal order, so there it must be
            False. Check this whenever the QPE implementation changes.
        verbose : bool
            Print gate-size information when the gate is built.
        """
        self.nl, self.C, self.t = nl, C, t
        self.num_breakpoints = num_breakpoints
        self.concentrate_near_zero = concentrate_near_zero
        self.bit_reversed = bit_reversed
        self.verbose = verbose

        self.gate = None                  # set by build_reciprocal_gate
        self.breakpoints = None
        self.num_work_ancillas = None

    @classmethod
    def from_hhl(cls, hhl, **kwargs):
        """Take nl, C and t from an `HHL_solver` instance."""
        return cls(hhl.nl, hhl.params["C"], hhl.params["t"], **kwargs)

    # ------------------------------------------------------------------
    # 1. Exact mapping: register integer -> rotation angle
    # ------------------------------------------------------------------
    @staticmethod
    def bit_reverse(x, nbits):
        """Reverse the order of the lowest `nbits` bits of x."""
        return int(format(x, f"0{nbits}b")[::-1], 2)

    def raw_to_angle(self, raw):
        """
        Exact mapping from a clock-register integer to the ancilla rotation
        angle 2*arcsin(C / lambda).

        Steps: (optionally bit-reverse) -> two's-complement signed value k ->
        phase phi = k / 2^nl -> eigenvalue estimate lambda = phi*2*pi/t ->
        angle. raw = 0 (lambda = 0) gets no rotation, and |C/lambda| > 1
        (an under-resolved eigenvalue) is clipped to +-pi.
        """
        nl = self.nl
        if raw == 0:
            return 0.0
        k = self.bit_reverse(raw, nl) if self.bit_reversed else raw
        k_signed = k - 2 ** nl if k >= 2 ** (nl - 1) else k
        phi = k_signed / 2 ** nl
        lam_est = phi * 2 * np.pi / self.t
        if lam_est == 0:
            return 0.0
        ratio = self.C / lam_est
        return np.sign(ratio) * np.pi if abs(ratio) > 1 else 2 * np.arcsin(ratio)

    # ------------------------------------------------------------------
    # 2. Piecewise-linear gate
    # ------------------------------------------------------------------
    def build_reciprocal_gate(self):
        """
        Fit angle(k) with a piecewise-linear function and return the gate
        implementing it (RY by that angle on the target qubit).

        Returns a `PiecewiseLinearPauliRotations` gate on
        [clock (nl) | target (1) | work ancillas].
        """
        nl = self.nl
        full_angles = np.array([self.raw_to_angle(k) for k in range(2 ** nl)])

        # --- breakpoints (always start at 0, strictly increasing) ---
        if self.concentrate_near_zero:
            # geometric spacing: dense near 0, sparse near 2^nl - 1
            frac = np.geomspace(1, 2 ** nl, self.num_breakpoints) / (2 ** nl)
            breakpoints = sorted(set((frac * (2 ** nl - 1)).astype(int)))
            if breakpoints[0] != 0:
                breakpoints = [0] + breakpoints
        else:
            breakpoints = sorted(set(np.linspace(0, 2 ** nl - 1, self.num_breakpoints, dtype=int)))

        # --- one line segment per interval: angle ~ slope * k + offset ---
        slopes, offsets = [], []
        for x0, x1 in zip(breakpoints[:-1], breakpoints[1:]):
            y0, y1 = full_angles[x0], full_angles[x1]
            slope = (y1 - y0) / (x1 - x0)
            slopes.append(slope)
            offsets.append(y0 - slope * x0)
        # After the last breakpoint the angle is held constant.
        slopes.append(0.0)
        offsets.append(full_angles[breakpoints[-1]])

        pw_gate = PiecewiseLinearPauliRotations(
            num_state_qubits=nl,
            breakpoints=breakpoints,
            slopes=slopes,
            offsets=offsets,
            basis="Y",
        )

        self.gate = pw_gate
        self.breakpoints = breakpoints
        self.num_work_ancillas = pw_gate.num_qubits - nl - 1
        if self.verbose:
            print(f"Gate needs {pw_gate.num_qubits} total qubits "
                  f"({nl} state + 1 target + {self.num_work_ancillas} ancilla)")
        return pw_gate

    # ------------------------------------------------------------------
    # 3. Validation against the exact multi-controlled-rotation loop
    # ------------------------------------------------------------------
    def validate_reciprocal_gate(self):
        """
        Compare the piecewise-linear gate with the exact loop (one
        multi-controlled RY per register value) on the clock + target
        qubits alone, starting from a uniform superposition over the clock
        register. This isolates the reciprocal's own error from QPE and
        Hamiltonian-evolution errors.

        Returns the fidelity |<exact|approx>|^2 (1.0 = identical).
        """
        nl = self.nl
        ql = QuantumRegister(nl, "clock")
        qa = QuantumRegister(1, "anc")

        # --- exact version ---
        qc_exact = QuantumCircuit(ql, qa)
        qc_exact.h(ql[:])
        for raw in range(1, 2 ** nl):
            angle = self.raw_to_angle(raw)
            # Flip the clock qubits that are 0 in |raw> so an all-ones
            # control fires exactly when the clock equals raw.
            bits = format(raw, f"0{nl}b")                 # MSB first
            flip = [ql[nl - 1 - i] for i, bit in enumerate(bits) if bit == "0"]
            if flip:
                qc_exact.x(flip)
            qc_exact.append(RYGate(angle).control(nl), ql[:] + [qa[0]])
            if flip:
                qc_exact.x(flip)

        # --- piecewise version (needs extra work ancillas) ---
        pw_gate = self.build_reciprocal_gate()
        work = QuantumRegister(self.num_work_ancillas, "work") if self.num_work_ancillas else None
        qc_approx = QuantumCircuit(ql, qa, *( [work] if work else [] ))
        qc_approx.h(ql[:])
        qc_approx.append(pw_gate, ql[:] + qa[:] + (work[:] if work else []))

        sv_exact = Statevector(qc_exact).data
        sv_approx = Statevector(qc_approx).data

        # Work ancillas are the highest qubits: with them in |0> the state
        # lives in the first 2^(nl+1) amplitudes. Anything outside means the
        # gate failed to uncompute its ancillas.
        n_main = 2 ** (nl + 1)
        leak = float(np.linalg.norm(sv_approx[n_main:]) ** 2)
        fidelity = abs(np.vdot(sv_exact, sv_approx[:n_main])) ** 2
        print(f"Reciprocal-only fidelity (exact vs piecewise, "
              f"{self.num_breakpoints} breakpoints): {fidelity:.6f}")
        if leak > 1e-9:
            print(f"WARNING: {leak:.2e} probability left in the work ancillas.")
        return fidelity
