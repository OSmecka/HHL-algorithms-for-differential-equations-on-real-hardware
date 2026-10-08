import numpy as np


class PDE_preper:
    """
    Prepares the linear system for the 1D wave equation, ready to be handed
    to a linear solver (classical, or HHL after the Hermitian dilation).

    Problem being discretized
    -------------------------
        u_tt = c^2 * u_xx + f(x, t)      on  x in [0, Lx],  t in [0, T]
        u(0, t) = u(Lx, t) = 0           (homogeneous Dirichlet BCs)
        u(x, 0) = u0(x),  u_t(x, 0) = v0(x)   (initial conditions)

    Method
    ------
    1. Space : central finite differences on Nx interior points.
    2. Time  : the second-order equation is rewritten as a first-order
               system in w = [u; v] with v = du/dt, then stepped with
               Crank-Nicolson (second-order accurate, unconditionally stable
               for this kind of system).
    3. All Nt time steps are stacked into ONE big linear system M w = rhs
       (a "time-all-at-once" formulation), which is what a quantum linear
       solver needs.

    Layout of the stacked unknown vector (length 2 * Nx * Nt)
    ---------------------------------------------------------
        W = [ w^1, w^2, ..., w^Nt ]            one block per time step
        w^n = [ u^n (Nx entries), v^n (Nx entries) ]

    Typical usage
    -------------
        prep = PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt)
        M, rhs, x, t, block_dim = prep.discretize_wave_pde()
        A, b = prep.hermitian_dilation(M, rhs)
    """

    def __init__(self, c, u0_func, v0_func, f_func, Lx, Nx, T, Nt):
        """
        Store the problem definition and precompute grids and step sizes.

        Parameters
        ----------
        c       : float
            Wave speed.
        u0_func : callable, u0_func(x) -> ndarray
            Initial displacement u(x, 0). Receives the array of interior
            grid points and must return an array of the same length.
        v0_func : callable, v0_func(x) -> ndarray
            Initial velocity u_t(x, 0). Same calling convention as u0_func.
        f_func  : callable, f_func(x, t) -> ndarray
            Source term f(x, t). Receives the interior grid points and a
            scalar time, returns an array of the same length as x.
        Lx      : float
            Length of the spatial domain [0, Lx].
        Nx      : int
            Number of INTERIOR spatial points (the two boundary points are
            eliminated because u = 0 there).
        T       : float
            Final time.
        Nt      : int
            Number of time steps. Must be a power of 2 so the stacked system
            (and its dilation) has a size that maps cleanly onto qubits.
        """
        # Nt > 0 and a power of 2  <=>  Nt has exactly one bit set.
        assert Nt > 0 and (Nt & (Nt - 1)) == 0, "Nt must be a power of 2"

        # --- problem definition ---
        self.c = c
        self.u0_func = u0_func
        self.v0_func = v0_func
        self.f_func = f_func
        self.Lx, self.Nx = Lx, Nx
        self.T, self.Nt = T, Nt

        # --- step sizes ---
        # Nx interior points split [0, Lx] into Nx + 1 equal intervals.
        self.dx = Lx / (Nx + 1)
        self.dt = T / Nt

        # --- grids ---
        # Spatial grid: interior points only (excludes x = 0 and x = Lx).
        self.x_grid = np.linspace(self.dx, Lx - self.dx, Nx)
        # Temporal grid: t_1, ..., t_Nt (excludes t = 0, which is the
        # initial condition and is not an unknown).
        self.t_grid = np.linspace(self.dt, T, Nt)

        # Each time step carries both u and v, so each block has 2*Nx entries.
        self.block_dim = 2 * Nx

    # ------------------------------------------------------------------
    # 1. Spatial operators
    # ------------------------------------------------------------------
    @staticmethod
    def spatial_laplacian(Nx, dx):
        """
        Discrete second derivative d^2/dx^2 with homogeneous Dirichlet BCs.

        Uses the standard 3-point stencil
            u_xx(x_i) ~ ( u_{i-1} - 2 u_i + u_{i+1} ) / dx^2.
        The boundary values u_0 = u_{Nx+1} = 0 are known, so they drop out
        of the unknowns and only affect the first/last rows (which simply
        lose one neighbour).

        Parameters
        ----------
        Nx : int
            Number of interior points.
        dx : float
            Grid spacing.

        Returns
        -------
        D_xx : (Nx, Nx) ndarray
            Tridiagonal matrix with -2 on the diagonal and +1 on the first
            sub-/super-diagonals, all divided by dx^2. It is symmetric and
            negative definite.
        """
        main = -2.0 * np.ones(Nx)   # diagonal entries
        off = np.ones(Nx - 1)       # sub- and super-diagonal entries
        D_xx = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx**2
        return D_xx

    @staticmethod
    def wave_generator(c, D_xx):
        """
        Generator L of the first-order-in-time system.

        With w = [u; v] and v = du/dt, the wave equation becomes
            du/dt = v
            dv/dt = c^2 * D_xx * u + f
        i.e.
            dw/dt = L w + [0; f],
            L = [[ 0,         I ],
                 [ c^2 * D_xx, 0 ]].

        Parameters
        ----------
        c    : float
            Wave speed.
        D_xx : (Nx, Nx) ndarray
            Spatial Laplacian from `spatial_laplacian`.

        Returns
        -------
        L : (2*Nx, 2*Nx) ndarray
            Block matrix above. It is NOT symmetric, which is why the
            resulting time-stepping matrix needs a Hermitian dilation
            before being used with HHL.
        """
        Nx = D_xx.shape[0]
        Z, I = np.zeros((Nx, Nx)), np.eye(Nx)   # zero block, identity block
        L = np.block([[Z, I], [c**2 * D_xx, Z]])
        return L

    # ------------------------------------------------------------------
    # 2. PDE discretization: Crank-Nicolson, block-bidiagonal system
    # ------------------------------------------------------------------
    def discretize_wave_pde(self):
        """
        Assemble the stacked Crank-Nicolson system  M W = rhs.

        One Crank-Nicolson step averages the right-hand side between time
        levels n and n+1:
            (I - dt/2 L) w^{n+1} = (I + dt/2 L) w^n + dt/2 (f^n + f^{n+1}).
        Moving the w^n term to the left-hand side and writing all Nt steps
        together gives a block-bidiagonal matrix:

            [ D          ] [w^1]   [ -S w^0 + dt/2 (f^0 + f^1) ]
            [ S  D       ] [w^2]   [  dt/2 (f^1 + f^2)         ]
            [    S  D    ] [w^3] = [  dt/2 (f^2 + f^3)         ]
            [       ...  ] [...]   [  ...                      ]

        with D = I - dt/2 L  (the "diagonal" block, multiplies w^{n+1})
        and  S = -(I + dt/2 L) (the "sub-diagonal" block, multiplies w^n).

        Returns
        -------
        M         : (2*Nx*Nt, 2*Nx*Nt) ndarray
            Block-bidiagonal, non-symmetric system matrix.
        rhs       : (2*Nx*Nt,) ndarray
            Right-hand side (initial conditions and source terms).
        x_grid    : (Nx,) ndarray
            Interior spatial grid points.
        t_grid    : (Nt,) ndarray
            Time levels t_1 ... t_Nt.
        block_dim : int
            Size of each per-time-step block (= 2*Nx). Use it to slice the
            solution: block n is W[n*block_dim:(n+1)*block_dim], whose first
            Nx entries are u and last Nx entries are v.
        """
        # Local aliases to keep the formulas below readable.
        Nx, Nt, dt = self.Nx, self.Nt, self.dt
        block_dim = self.block_dim
        x_grid, t_grid = self.x_grid, self.t_grid

        # Build the spatial operator and the first-order generator.
        D_xx = self.spatial_laplacian(Nx, self.dx)
        L = self.wave_generator(self.c, D_xx)

        # The two blocks that repeat along the diagonals of M.
        I_block = np.eye(block_dim)
        diag_block = I_block - (dt / 2) * L      # multiplies w^{n+1}
        sub_block = -(I_block + (dt / 2) * L)    # multiplies w^n (moved to LHS)

        # Allocate the full system.
        M = np.zeros((Nt * block_dim, Nt * block_dim))
        rhs = np.zeros(Nt * block_dim)

        # Initial state w^0 = [u0; v0], evaluated on the interior grid.
        w0 = np.concatenate([self.u0_func(x_grid), self.v0_func(x_grid)])

        # Source term at the "previous" time level. It only acts on the
        # velocity equation, so the u-part is zero. Starts at t = 0.
        f_prev = np.concatenate([np.zeros(Nx), self.f_func(x_grid, 0.0)])

        # Fill M and rhs one time step (block row) at a time.
        for n in range(Nt):
            # Row range of block n in the big system.
            r0, r1 = n * block_dim, (n + 1) * block_dim

            # Diagonal block: coefficient of the unknown w^{n+1}.
            M[r0:r1, r0:r1] = diag_block

            # Source term at the "next" time level t_{n+1} (= t_grid[n]).
            f_next = np.concatenate([np.zeros(Nx), self.f_func(x_grid, t_grid[n])])

            if n > 0:
                # Sub-diagonal block: couples step n to the previous unknown.
                M[r0:r1, r0 - block_dim:r0] = sub_block
            else:
                # First step: w^0 is KNOWN, so its contribution
                # (sub_block @ w0) cannot sit in M. Move it to the RHS
                # with a sign flip.
                rhs[r0:r1] += -sub_block @ w0

            # Trapezoidal average of the source over the step.
            rhs[r0:r1] += (dt / 2) * (f_prev + f_next)

            # Reuse f_next as f_prev on the next iteration (saves one call).
            f_prev = f_next

        return M, rhs, x_grid, t_grid, block_dim

    # ------------------------------------------------------------------
    # 3. Hermitian dilation (needed for HHL)
    # ------------------------------------------------------------------
    @staticmethod
    def hermitian_dilation(M, rhs):
        """
        Embed the non-symmetric system M y = rhs in a Hermitian one.

        HHL requires a Hermitian matrix. For a real non-symmetric M, use
            A = [[ 0,   M ],
                 [ M^T, 0 ]],      b = [ rhs; 0 ].
        Solving A [x; y] = b gives
            M y   = rhs    (upper block row)
            M^T x = 0      (lower block row)
        so (for invertible M) x = 0 and y is the solution of the original
        system. In other words: the answer lives in the SECOND half of the
        dilated solution vector.

        Parameters
        ----------
        M   : (N, N) ndarray
            Original system matrix.
        rhs : (N,) ndarray
            Original right-hand side.

        Returns
        -------
        A     : (2N, 2N) ndarray
            Symmetric (Hermitian) dilated matrix.
        b_vec : (2N,) ndarray
            Dilated right-hand side, zero-padded.
        """
        N = M.shape[0]
        A = np.block([[np.zeros((N, N)), M], [M.T, np.zeros((N, N))]])
        b_vec = np.concatenate([rhs, np.zeros(N)])
        return A, b_vec

    # ------------------------------------------------------------------
    # 4. Convenience wrapper
    # ------------------------------------------------------------------
    def build(self):
        """
        Run the full pipeline: discretize the PDE, then apply the Hermitian
        dilation.

        Returns
        -------
        A, b_vec  : dilated Hermitian matrix and right-hand side
        x_grid    : interior spatial grid
        t_grid    : time levels
        block_dim : per-time-step block size (2*Nx), for slicing u and v
                    out of the solution (remember the solution of the
                    original system is the second half of the dilated one)
        """
        M, rhs, x_grid, t_grid, block_dim = self.discretize_wave_pde()
        A, b_vec = self.hermitian_dilation(M, rhs)
        return A, b_vec, x_grid, t_grid, block_dim
