import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


class PDE_postprocessor:
    """
    Turns the stacked solution vector from `HHL_solver` (or from a classical
    solve) into u(x,t), v(x,t) histories, interpolated callables, and
    classical-vs-quantum comparison plots.

    It is built from a `PDE_preper` instance, so the grids, the domain and
    the initial conditions always match the system that was solved.

    Typical usage
    -------------
        post = PDE_postprocessor(prep)
        u_hist, v_hist = post.reshape_to_uv(w)            # w from HHL_solver
        u_of_xt, v_of_xt = post.make_uv_of_xt(u_hist, v_hist)
        post.plot_comparison(u_classical, u_quantum)

    Conventions
    -----------
    - Histories have shape (Nt, Nx): row n is time t_{n+1}, column j is the
      interior point x_{j+1}. The initial condition (t = 0) and the Dirichlet
      boundary points (x = 0, Lx) are NOT in the histories; the "full" grids
      built here add them.
    - The stacked vector layout is [u^1, v^1, u^2, v^2, ...], i.e. each
      time step contributes a block [u_1..u_Nx, v_1..v_Nx] of size
      block_dim = 2*Nx (matches `PDE_preper.discretize_wave_pde`).
    """

    def __init__(self, prep):
        """
        Parameters
        ----------
        prep : PDE_preper
            The instance used to build the system. Provides Lx, Nx, T, Nt,
            the interior grids, block_dim and the initial-condition
            functions u0_func, v0_func.
        """
        self.prep = prep
        self.Lx, self.Nx = prep.Lx, prep.Nx
        self.T, self.Nt = prep.T, prep.Nt
        self.block_dim = prep.block_dim
        self.u0_func, self.v0_func = prep.u0_func, prep.v0_func

        # Interior grids (the unknowns)
        self.x_grid, self.t_grid = prep.x_grid, prep.t_grid

        # Full grids: add the Dirichlet boundary points and the t = 0 level
        self.x_full = np.concatenate([[0.0], self.x_grid, [self.Lx]])
        self.t_full = np.concatenate([[0.0], self.t_grid])

    # ------------------------------------------------------------------
    # 1. Vector -> (u, v) histories
    # ------------------------------------------------------------------
    def reshape_to_uv(self, w):
        """
        Split the stacked solution vector w (length Nt*block_dim, i.e. the
        M^{-1} rhs half of the dilated solution, exactly what
        `HHL_solver.solve()["solution"]` returns) into u and v histories of
        shape (Nt, Nx). Tiny imaginary parts from the simulation are dropped.
        """
        w = np.real_if_close(np.asarray(w), tol=1e6)
        assert w.size == self.Nt * self.block_dim, (
            f"expected a vector of length {self.Nt * self.block_dim}, got {w.size}")
        w_hist = np.real(w).reshape(self.Nt, self.block_dim)
        u_hist = w_hist[:, :self.Nx]
        v_hist = w_hist[:, self.Nx:2 * self.Nx]
        return u_hist, v_hist

    def apply_signs(self, w_abs, sign_reference):
        """
        Restore signs on a magnitude-only solution.

        Shot counts only give |amplitude|^2, so `HHL_solver.run_aer` returns
        |w|. This multiplies by the sign of `sign_reference` (e.g. the exact
        statevector solution or the classical solution, same length as w).

        NOTE: this borrows the signs from another solution, so it is a
        diagnostic aid, not an independent quantum result. Report the
        sign-free comparison too if you need a strict claim.
        """
        w_abs = np.asarray(w_abs, dtype=float)
        signs = np.sign(np.real(np.asarray(sign_reference)))
        assert signs.shape == w_abs.shape, "sign_reference must match w_abs in length"
        signs[signs == 0] = 1.0
        return w_abs * signs

    # ------------------------------------------------------------------
    # 2. Full grids (with initial condition and boundaries)
    # ------------------------------------------------------------------
    def build_full_u(self, u_hist):
        """
        (Nt+1, Nx+2) displacement grid: prepends the t=0 initial condition
        u0(x) and pads x=0, Lx with the Dirichlet boundary value u = 0.
        Ready for pcolormesh or interpolation.
        """
        u_full = np.vstack([self.u0_func(self.x_grid)[None, :], u_hist])
        zeros = np.zeros((self.Nt + 1, 1))
        return np.hstack([zeros, u_full, zeros])

    def build_full_v(self, v_hist):
        """
        (Nt+1, Nx+2) velocity grid: prepends v0(x) at t=0. Velocity has no
        boundary condition here, so the edge columns repeat the nearest
        interior value (flat extrapolation).
        """
        v_full = np.vstack([self.v0_func(self.x_grid)[None, :], v_hist])
        return np.hstack([v_full[:, :1], v_full, v_full[:, -1:]])

    # ------------------------------------------------------------------
    # 3. Continuous u(x,t), v(x,t)
    # ------------------------------------------------------------------
    def _interpolator(self, full_grid):
        """Linear interpolant over (t, x); extrapolates outside the grid."""
        return RegularGridInterpolator((self.t_full, self.x_full), full_grid,
                                       bounds_error=False, fill_value=None)

    @staticmethod
    def _evaluate(interp, x, t):
        """Evaluate an interpolant at broadcast-compatible x and t."""
        x, t = np.atleast_1d(x), np.atleast_1d(t)
        pts = np.stack(np.broadcast_arrays(t, x), axis=-1)
        return interp(pts)

    def make_uv_of_xt(self, u_hist, v_hist):
        """
        Wrap the discrete histories into callables u_of_xt(x, t) and
        v_of_xt(x, t), valid for x in [0, Lx] and t in [0, T]. They use the
        known initial condition at t = 0 and u = 0 at x = 0, Lx.

        Returns (u_of_xt, v_of_xt).
        """
        u_interp = self._interpolator(self.build_full_u(u_hist))
        v_interp = self._interpolator(self.build_full_v(v_hist))

        def u_of_xt(x, t):
            return self._evaluate(u_interp, x, t)

        def v_of_xt(x, t):
            return self._evaluate(v_interp, x, t)

        return u_of_xt, v_of_xt

    # ------------------------------------------------------------------
    # 4. Comparison metrics and plots
    # ------------------------------------------------------------------
    @staticmethod
    def error_summary(u_ref, u_test):
        """
        Compare two (Nt, Nx) histories. Returns
          max_abs : largest pointwise absolute error
          rel_l2  : ||u_test - u_ref|| / ||u_ref||
        """
        diff = np.asarray(u_test) - np.asarray(u_ref)
        return dict(max_abs=float(np.abs(diff).max()),
                    rel_l2=float(np.linalg.norm(diff) / np.linalg.norm(u_ref)))

    def plot_comparison(self, u_hist_classical, u_hist_quantum, dense=False,
                        n_dense=80, show=True, save_path=None):
        """
        Three-panel heatmap in (x, t): classical, quantum (HHL), difference.

        dense=False : the raw (Nt+1) x (Nx+2) grid. Honest, but blocky when
                      Nx and Nt are small.
        dense=True  : linearly interpolated onto an n_dense x n_dense grid.
                      Smoother-looking, but it is the same data resampled; it
                      adds no information.
        show        : call plt.show().
        save_path   : if given, save the figure there (e.g. "uxt.png").

        Returns the matplotlib Figure.
        """
        U_c = self.build_full_u(u_hist_classical)
        U_q = self.build_full_u(u_hist_quantum)

        if not dense:
            X_plot, T_plot = self.x_full, self.t_full
        else:
            interp_c, interp_q = self._interpolator(U_c), self._interpolator(U_q)
            X_plot = np.linspace(0, self.Lx, n_dense)
            T_plot = np.linspace(0, self.T, n_dense)
            Xg, Tg = np.meshgrid(X_plot, T_plot)
            U_c = self._evaluate(interp_c, Xg.ravel(), Tg.ravel()).reshape(Xg.shape)
            U_q = self._evaluate(interp_q, Xg.ravel(), Tg.ravel()).reshape(Xg.shape)

        U_diff = U_c - U_q

        # Same colour scale for classical and quantum so they are comparable;
        # the difference gets its own symmetric scale.
        vmax = max(np.abs(U_c).max(), np.abs(U_q).max())
        diff_max = max(np.abs(U_diff).max(), 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        im0 = axes[0].pcolormesh(X_plot, T_plot, U_c, shading="auto",
                                 cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title("Classical u(x,t)")
        im1 = axes[1].pcolormesh(X_plot, T_plot, U_q, shading="auto",
                                 cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("Quantum (HHL) u(x,t)")
        im2 = axes[2].pcolormesh(X_plot, T_plot, U_diff, shading="auto",
                                 cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
        axes[2].set_title("Difference (classical − quantum)")

        for ax in axes:
            ax.set_xlabel("x")
        axes[0].set_ylabel("t")
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig


class IBM_QPU_helper:
    """
    Post-processing for results that come back as measurement COUNTS
    (IBM hardware, or any shot-based run): fetch the counts from a job,
    turn them into the PDE solution, and plot them against the classical
    solution.

    Built from an `HHL_solver` (circuit layout and scaling constants) and a
    `PDE_postprocessor` (grids, boundary/initial conditions, interpolation).

    Typical usage
    -------------
        qpu    = IBM_QPU_helper(hhl, post)
        counts = qpu.get_counts_from_job(job)          # waits for the job
        ibm    = qpu.counts_to_solution_abs(counts)    # |w| from the counts
        qpu.plot_runs(u_classical, {"IBM hardware": u_ibm})
    """

    def __init__(self, hhl, post):
        """
        Parameters
        ----------
        hhl  : HHL_solver
            Provides nb, nl (qubit layout), params (||b||, C) and N_raw.
        post : PDE_postprocessor
            Provides grids, build_full_u and make_uv_of_xt for plotting.
        """
        self.hhl = hhl
        self.post = post

    # ------------------------------------------------------------------
    # 1. Counts from a Sampler job
    # ------------------------------------------------------------------
    @staticmethod
    def get_counts_from_job(job):
        """
        Block until the Sampler job is done and return its counts as a dict
        {bitstring: count}. measure_all() names the classical register
        "meas"; if that is missing, fall back to the first register that
        has counts.
        """
        data = job.result()[0].data               # first (only) circuit
        register = getattr(data, "meas", None)
        if register is None:
            name = next(f for f in dir(data)
                        if not f.startswith("_") and hasattr(getattr(data, f), "get_counts"))
            register = getattr(data, name)
        return register.get_counts()

    # ------------------------------------------------------------------
    # 2. Counts -> solution
    # ------------------------------------------------------------------
    def counts_to_solution_abs(self, counts):
        """
        Same post-processing as `HHL_solver.run_aer`, but for any counts
        dict (here: counts from IBM hardware).

        Keeps shots with ancilla = 1 and clock = 0, converts them to
        |amplitudes| = sqrt(count / total_shots) and rescales by ||b|| / C.
        Counts only give probabilities, so signs are lost (see
        `PDE_postprocessor.apply_signs`).

        Returns a dict with:
          solution_abs : |w| (length Nt*block_dim)
          n_post       : number of shots that survived post-selection
          shots        : total shots
          sys_counts   : post-selected counts per system basis state
        """
        hhl = self.hhl
        shots = sum(counts.values())
        nb, nl = hhl.nb, hhl.nl

        sys_counts = np.zeros(2 ** nb)
        for bitstring, n in counts.items():
            # Bitstring is MSB first: [anc | clock | sys]; qubit i = bit i of idx.
            idx = int(bitstring.replace(" ", ""), 2)
            anc_bit = (idx >> (nb + nl)) & 1
            clock_val = (idx >> nb) & (2 ** nl - 1)
            if anc_bit == 1 and clock_val == 0:
                sys_counts[idx & (2 ** nb - 1)] += n

        n_post = int(sys_counts.sum())
        # P(anc=1, clock=0, sys=j) = |amp_j|^2, divided by ALL shots.
        amp_abs = np.sqrt(sys_counts / shots)
        z = amp_abs * hhl.params["norm_b"] / hhl.params["C"]   # solution of A z = b
        z = z[: hhl.N_raw]                                      # drop padding
        return dict(solution_abs=np.abs(z[hhl.N_raw // 2:]),    # z = [0; w] -> w
                    n_post=n_post, shots=shots, sys_counts=sys_counts)

    # ------------------------------------------------------------------
    # 3. Plot classical vs any number of runs
    # ------------------------------------------------------------------
    def plot_runs(self, u_classical, runs, dense=True, n_dense=80,
                  show=True, save_path=None):
        """
        Compare the classical solution with any number of quantum runs.

        Top row    : classical, then each run (shared colour scale).
        Bottom row : difference (classical - run) for each run, on one
                     shared symmetric scale so the errors are comparable.

        Parameters
        ----------
        u_classical : (Nt, Nx) classical history.
        runs        : dict {label: (Nt, Nx) history}, e.g.
                      {"Aer": u_q, "IBM hardware": u_ibm}.
        dense       : interpolate onto an n_dense x n_dense grid (same data,
                      just smoother-looking) instead of the raw grid.
        show        : call plt.show().
        save_path   : if given, save the figure there.

        Returns the matplotlib Figure.
        """
        post = self.post

        def field(u_hist):
            if not dense:
                return post.x_full, post.t_full, post.build_full_u(u_hist)
            u_of_xt, _ = post.make_uv_of_xt(u_hist, np.zeros_like(u_hist))
            X = np.linspace(0, post.Lx, n_dense)
            T = np.linspace(0, post.T, n_dense)
            Xg, Tg = np.meshgrid(X, T)
            return X, T, u_of_xt(Xg.ravel(), Tg.ravel()).reshape(Xg.shape)

        X, T, U_c = field(u_classical)
        fields = {label: field(u)[2] for label, u in runs.items()}
        diffs = {label: U_c - U for label, U in fields.items()}

        vmax = max(np.abs(U_c).max(), *(np.abs(U).max() for U in fields.values()))
        diff_max = max(max(np.abs(d).max() for d in diffs.values()), 1e-12)

        ncol = 1 + len(runs)
        fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 8), sharey=True, squeeze=False)

        # Top row: solutions
        im = axes[0, 0].pcolormesh(X, T, U_c, shading="auto", cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax)
        axes[0, 0].set_title("Classical u(x,t)")
        for k, (label, U) in enumerate(fields.items(), start=1):
            im = axes[0, k].pcolormesh(X, T, U, shading="auto", cmap="RdBu_r",
                                       vmin=-vmax, vmax=vmax)
            axes[0, k].set_title(f"{label} u(x,t)")
        fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.9)

        # Bottom row: differences (first cell is left empty)
        axes[1, 0].axis("off")
        for k, (label, d) in enumerate(diffs.items(), start=1):
            imd = axes[1, k].pcolormesh(X, T, d, shading="auto", cmap="RdBu_r",
                                        vmin=-diff_max, vmax=diff_max)
            axes[1, k].set_title(f"Classical − {label}")
        fig.colorbar(imd, ax=axes[1, 1:].tolist(), shrink=0.9)

        for ax in list(axes[0, :]) + list(axes[1, 1:]):
            ax.set_xlabel("x")
        for ax in axes[:, 0]:
            ax.set_ylabel("t")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig
