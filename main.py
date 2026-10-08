import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate, RYGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

try:
    from qiskit.circuit.library import QFTGate

    def _qft(n, inverse=False):
        gate = QFTGate(n)
        return gate.inverse() if inverse else gate
except ImportError:  # older Qiskit
    from qiskit.circuit.library import QFT

    def _qft(n, inverse=False):
        return QFT(n, inverse=inverse)


if __name__ == "__main__":
    from PDE_preparation import PDE_preper
    from HHL_solver import HHL_solver
    # --- inputs ---
    c = 1.0
    Lx, Nx = 1.0, 2
    T, Nt = 1.0, 2
    u0_func = lambda x: np.sin(np.pi * x / Lx)
    v0_func = lambda x: np.zeros_like(x)
    f_func = lambda x, t: np.zeros_like(x)      # no forcing: standing wave

    prep = PDE_preper(c, u0_func, v0_func, f_func, Lx, Nx, T, Nt)
    hhl = HHL_solver.from_pde_preper(prep)
    out = hhl.solve()

    fid, rel = hhl.compare(out["solution"])
    u, v = hhl.reshape_solution(out["solution"])
    print(f"success probability = {out['success_prob']:.4f}")
    print(f"fidelity vs classical = {fid:.6f}, relative error = {rel:.4f}")
    print("u (rows = time steps):\n", u)
