"""
convergence.py - empirical vs. theoretical convergence rates for the
HHL wave-equation pipeline.

Three independent error sources are studied, each against its theoretical
rate:

  1. Discretization error (classical, no quantum part)
       spatial_convergence, temporal_convergence, joint_convergence
       expected: second order in dx and in dt (central differences and
       Crank-Nicolson).

  2. HHL algorithmic error (statevector, noise-free)
       hhl_clock_convergence    error vs. number of clock qubits nl
       hhl_trotter_convergence  error vs. Trotter repetitions
       expected: error ~ kappa / 2^nl  (phase-estimation resolution),
                 Trotter error ~ reps^(-order).

  3. Sampling error (finite shots)
       sampling_convergence
       expected: error ~ shots^(-1/2). The script also evaluates the
       closed-form prediction, not just the slope.

The reference problem is the standing wave
    u(x, t) = sin(m*pi*x/Lx) * cos(m*pi*c*t/Lx),
which is the default input of main.py (mode m = 1, no forcing).

Parts 2 and 3 need Qiskit (they import HHL_solver); part 1 needs only
numpy / scipy / matplotlib.

Usage
-----
    python convergence.py                    # everything + summary plot
    python convergence.py --skip-quantum     # discretization only
    python convergence.py --no-plot          # tables only, no figure
    python convergence.py --nx 2 --nt 2 --save-dir figures

A combined summary figure of all studies is drawn at the end (toggle with
run_all(..., plot=False) or --no-plot). With a save directory, the summary
and one PNG per study are also written to disk.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from PDE_preparation import PDE_preper


# ======================================================================
# Shared helpers
# ======================================================================

def standing_wave(c=1.0, Lx=1.0, mode=1):
    """
    Exact standing-wave solution and the matching problem inputs.

    Returns (u_exact, v_exact, u0, v0, f), where u_exact(x, t) and
    v_exact(x, t) broadcast over x and t, and u0, v0, f have the calling
    conventions PDE_preper expects.
    """
    k = mode * np.pi / Lx
    om = c * k
    u_exact = lambda x, t: np.sin(k * np.asarray(x)) * np.cos(om * t)
    v_exact = lambda x, t: -om * np.sin(k * np.asarray(x)) * np.sin(om * t)
    u0 = lambda x: np.sin(k * x)
    v0 = lambda x: np.zeros_like(x)
    f = lambda x, t: np.zeros_like(x)
    return u_exact, v_exact, u0, v0, f


def make_prep(c=1.0, Lx=1.0, Nx=2, T=1.0, Nt=2, mode=1):
    """PDE_preper for the standing-wave problem."""
    _, _, u0, v0, f = standing_wave(c, Lx, mode)
    return PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt)


def fit_order(h, err, floor=1e-13):
    """
    Least-squares slope of log(err) against log(h), i.e. the observed
    convergence order. Points at or below `floor` (round-off) are ignored.
    """
    h, err = np.asarray(h, float), np.asarray(err, float)
    ok = np.isfinite(err) & (err > floor)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[ok]), np.log(err[ok]), 1)[0])


def pairwise_orders(h, err):
    """Observed order between each pair of successive refinements."""
    h, err = np.asarray(h, float), np.asarray(err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def _result(h, err, expected, **extra):
    """Bundle a convergence study into a dict."""
    return dict(h=np.asarray(h, float), err=np.asarray(err, float),
                order=fit_order(h, err), pairwise=pairwise_orders(h, err),
                expected=expected, **extra)


def print_table(title, header, rows):
    """Print a plain-text table."""
    print(f"\n{title}")
    print("  " + "  ".join(f"{h:>13}" for h in header))
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:>13d}")
            elif isinstance(v, str):
                cells.append(f"{v:>13}")
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append(f"{'-':>13}")
            else:
                cells.append(f"{v:>13.4e}")
        print("  " + "  ".join(cells))


def plot_loglog(x, curves, xlabel, title, expected_slope=None,
                expected_label=None, invert_x=False, save_path=None,
                show=False, ax=None):
    """
    Log-log plot of one or more error curves against x, with an optional
    dashed reference line of a given slope anchored at the first point of
    the first curve.

    If `ax` is given the curves are drawn on that axis (used by
    `plot_summary`) and nothing is saved, shown or closed; otherwise a new
    figure is created.
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
    x = np.asarray(x, float)
    for label, y in curves.items():
        ax.loglog(x, y, "o-", label=label)
    if expected_slope is not None:
        first = next(iter(curves.values()))
        y0 = np.asarray(first, float)[0]
        ax.loglog(x, y0 * (x / x[0]) ** expected_slope, "k--", alpha=0.6,
                  label=expected_label or f"slope {expected_slope:g}")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if own:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
    return fig


def _panel_specs(results):
    """
    Describe one plot panel per study present in `results` (the dict
    built by `run_all`): a list of dicts with the arguments for
    `plot_loglog`. Studies that were not run are skipped.
    """
    specs = []
    disc = results.get("discretization", {})
    for key, hname in [("spatial", "dx"), ("temporal", "dt"), ("joint", "dx")]:
        if key not in disc:
            continue
        r = disc[key]
        curves = {f"observed (order {r['order']:.2f})": r["err"]}
        if key == "joint":
            curves[f"v error (order {r['order_v']:.2f})"] = r["err_v"]
        specs.append(dict(name=key, x=r["h"], curves=curves, xlabel=hname,
                          title=f"{key.capitalize()} convergence",
                          expected_slope=2, expected_label="slope 2 (theory)"))
    if "clock" in results:
        r = results["clock"]
        specs.append(dict(
            name="clock", x=r["h"],
            curves={f"rel. error (order {r['order']:.2f})": r["err"],
                    "kappa / 2^nl": r["scale"]},
            xlabel="2^-nl  (clock resolution)",
            title="HHL error vs clock qubits", expected_slope=1,
            expected_label="slope 1 (theory)", invert_x=True))
    if "trotter" in results:
        r = results["trotter"]
        specs.append(dict(
            name="trotter", x=r["h"],
            curves={f"Trotter error (order {r['order']:.2f})": r["err"]},
            xlabel="1 / reps", title="Trotter error vs exact HHL",
            expected_slope=r["expected"],
            expected_label=f"slope {r['expected']:g} (theory)"))
    if "sampling" in results:
        r = results["sampling"]
        specs.append(dict(
            name="sampling", x=r["h"],
            curves={f"sampling (order {r['order']:.2f})": r["err"],
                    "predicted": r["err_predicted"],
                    "total (vs classical)": r["err_total"]},
            xlabel="1 / shots", title="Sampling error", expected_slope=0.5,
            expected_label="slope 1/2 (theory)"))
    return specs


def plot_summary(results, save_path=None, show=True):
    """
    One figure with a log-log panel for every study in `results`, each with
    its theoretical slope as a dashed line.

    Parameters
    ----------
    results   : dict as returned by `run_all`. Keys used: "discretization"
                (with "spatial", "temporal", "joint"), "clock", "trotter",
                "sampling". Missing keys are skipped.
    save_path : if given, save the figure there.
    show      : call plt.show().
    """
    specs = _panel_specs(results)
    if not specs:
        print("plot_summary: nothing to plot.")
        return None
    ncol = min(3, len(specs))
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow),
                             squeeze=False)
    for ax, sp in zip(axes.ravel(), specs):
        plot_loglog(sp["x"], sp["curves"], sp["xlabel"], sp["title"],
                    expected_slope=sp.get("expected_slope"),
                    expected_label=sp.get("expected_label"),
                    invert_x=sp.get("invert_x", False), ax=ax)
    for ax in axes.ravel()[len(specs):]:
        ax.axis("off")
    fig.suptitle("Convergence summary: observed vs theoretical rates")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# ======================================================================
# 1. Discretization error (classical)
# ======================================================================
def spatial_convergence(c=1.0, Lx=1.0, T=1.0, mode=1, k_values=range(2, 9),
                        n_times=8):
    """
    Spatial order of the central-difference Laplacian.

    The semi-discrete system dw/dt = L w is solved EXACTLY in time (matrix
    exponential), so the only error left is the spatial one. It is
    compared with the continuous standing wave. Grid: Nx = 2^k - 1
    interior points, dx = Lx / 2^k.

    Error = max over x and the sampled times of |u_h - u_exact|.
    Expected order: 2.
    """
    u_ex, _, u0, v0, _ = standing_wave(c, Lx, mode)
    times = np.linspace(T / n_times, T, n_times)
    hs, errs = [], []
    for k in k_values:
        Nx = 2 ** k - 1
        dx = Lx / (Nx + 1)
        x = np.linspace(dx, Lx - dx, Nx)
        L = PDE_preper.wave_generator(c, PDE_preper.spatial_laplacian(Nx, dx))
        step = expm(L * (T / n_times))           # exact propagator
        w = np.concatenate([u0(x), v0(x)])
        err = 0.0
        for t in times:
            w = step @ w
            err = max(err, np.abs(w[:Nx] - u_ex(x, t)).max())
        hs.append(dx)
        errs.append(err)
    return _result(hs, errs, expected=2.0)


def temporal_convergence(c=1.0, Lx=1.0, T=1.0, mode=1, Nx=7,
                         k_values=range(1, 8)):
    """
    Temporal order of the stacked Crank-Nicolson system.

    The all-at-once system M W = rhs from PDE_preper is solved directly and
    compared with the exact solution of the SAME semi-discrete system
    (matrix exponential), so the spatial error cancels and only the time
    stepping error remains. Nt = 2^k steps, dt = T / Nt.

    Error = max over time steps of the max-norm of (u, v) differences.
    Expected order: 2.
    """
    _, _, u0, v0, f = standing_wave(c, Lx, mode)
    hs, errs = [], []
    for k in k_values:
        Nt = 2 ** k
        prep = PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt)
        M, rhs, x, _, block_dim = prep.discretize_wave_pde()
        W = np.linalg.solve(M, rhs).reshape(Nt, block_dim)

        L = PDE_preper.wave_generator(
            c, PDE_preper.spatial_laplacian(Nx, prep.dx))
        step = expm(L * prep.dt)
        w = np.concatenate([u0(x), v0(x)])
        err = 0.0
        for n in range(Nt):
            w = step @ w
            err = max(err, np.abs(W[n] - w).max())
        hs.append(prep.dt)
        errs.append(err)
    return _result(hs, errs, expected=2.0)


def joint_convergence(c=1.0, Lx=1.0, T=1.0, mode=1, k_values=range(2, 6)):
    """
    Total discretization error of the full system, refining space and time
    together (Nx = 2^k - 1, Nt = 2^k, so dt/dx stays constant), against
    the continuous standing wave. This is exactly the system HHL receives.

    The system is solved densely, so keep k_values small (size 2*Nx*Nt).
    Returns the u error as the main curve and the v error as `err_v`.
    Expected order: 2 in h = dx.
    """
    u_ex, v_ex, u0, v0, f = standing_wave(c, Lx, mode)
    hs, err_u, err_v = [], [], []
    for k in k_values:
        Nx, Nt = 2 ** k - 1, 2 ** k
        prep = PDE_preper(c, u0, v0, f, Lx, Nx, T, Nt)
        M, rhs, x, t, block_dim = prep.discretize_wave_pde()
        W = np.linalg.solve(M, rhs).reshape(Nt, block_dim)
        X, TT = x[None, :], t[:, None]
        err_u.append(np.abs(W[:, :Nx] - u_ex(X, TT)).max())
        err_v.append(np.abs(W[:, Nx:] - v_ex(X, TT)).max())
        hs.append(prep.dx)
    res = _result(hs, err_u, expected=2.0)
    res["err_v"] = np.asarray(err_v)
    res["order_v"] = fit_order(hs, err_v)
    return res


def discretization_convergence(c=1.0, Lx=1.0, T=1.0, mode=1, verbose=True):
    """Run the spatial, temporal and joint studies; print a summary."""
    out = dict(
        spatial=spatial_convergence(c, Lx, T, mode),
        temporal=temporal_convergence(c, Lx, T, mode),
        joint=joint_convergence(c, Lx, T, mode),
    )
    if verbose:
        for name, key, hname in [("Spatial (exact in time)", "spatial", "dx"),
                                 ("Temporal (Crank-Nicolson)", "temporal", "dt"),
                                 ("Joint (dt/dx fixed)", "joint", "dx")]:
            r = out[key]
            rows = [(float(h), float(e), None if i == 0 else float(p))
                    for i, (h, e, p) in enumerate(
                        zip(r["h"], r["err"],
                            np.concatenate([[np.nan], r["pairwise"]])))]
            print_table(f"{name}: fitted order = {r['order']:.3f} "
                        f"(theory {r['expected']:g})",
                        [hname, "max error", "local order"], rows)
    return out


# ======================================================================
# 2. HHL algorithmic error (statevector, noise-free)
# ======================================================================
def hhl_clock_convergence(prep, nl_values=None, verbose=True, **hhl_kwargs):
    """
    HHL error vs. the number of clock qubits nl.

    Theory: the QPE register resolves phases in steps of 1/2^nl, so the
    relative error of the smallest eigenvalue is at most about
    kappa / 2^nl (with t = pi / lambda_max). The error in the solution
    therefore falls like 2^-nl once nl exceeds the minimum, so the fitted
    order against h = 2^-nl should be about 1. It is only a rough trend:
    eigenphases fall between clock bins in an irregular way, so individual
    points are not monotone, and the curve saturates at the
    Trotter/numerical floor.

    Parameters
    ----------
    prep        : PDE_preper
    nl_values   : iterable of clock-qubit counts. Default: from one below
                  the minimum up to two above it. Runtime and memory grow
                  quickly with nl (2^nl multi-controlled rotations), so
                  extend this range with care.
    hhl_kwargs  : forwarded to HHL_solver (e.g. use_exact=False).

    Returns a dict with h = 2^-nl, err (relative error vs the classical
    solution), 1 - fidelity, success probability, kappa and the
    kappa / 2^nl scale.
    """
    from HHL_solver import HHL_solver

    probe = HHL_solver.from_pde_preper(prep, verbose=False, **hhl_kwargs)
    nl_min, kappa = probe.nl_min, probe.params["kappa"]
    if nl_values is None:
        nl_values = range(max(1, nl_min - 1), nl_min + 3)
    nl_values = list(nl_values)

    errs, infid, succ = [], [], []
    for nl in nl_values:
        hhl = HHL_solver.from_pde_preper(prep, n_clock_qubits=nl,
                                         verbose=False, **hhl_kwargs)
        out = hhl.solve()
        fid, rel = hhl.compare(out["solution"])
        errs.append(rel)
        infid.append(1.0 - fid)
        succ.append(out["success_prob"])

    nl_arr = np.array(nl_values)
    h = 2.0 ** (-nl_arr.astype(float))
    res = _result(h, errs, expected=1.0, nl=nl_arr, infidelity=np.array(infid),
                  success_prob=np.array(succ), kappa=kappa, nl_min=nl_min,
                  scale=kappa * h)
    if verbose:
        rows = [(int(n), float(e), float(i), float(s), float(sc))
                for n, e, i, s, sc in zip(nl_arr, res["err"], res["infidelity"],
                                          res["success_prob"], res["scale"])]
        print_table(f"HHL error vs clock qubits (kappa = {kappa:.2f}, "
                    f"nl_min = {nl_min}); fitted order in 2^-nl = "
                    f"{res['order']:.3f} (theory ~1)",
                    ["nl", "rel. error", "1 - fidelity", "P(success)",
                     "kappa/2^nl"], rows)
    return res


def hhl_trotter_convergence(prep, reps_list=(1, 2, 4, 8), n_clock_qubits=4,
                            order=2, verbose=True):
    """
    Trotter error of the controlled evolution e^{iAt}.

    Each Trotterized solution is compared with the exact-exponential HHL
    solution at the SAME number of clock qubits, so only the Trotter error
    is measured. Theory: error ~ reps^-order for an order-`order`
    Suzuki-Trotter formula (h = 1 / reps), until it hits round-off.

    Cost grows with reps and with nl (the power 2^j multiplies reps), so
    the default is a small fixed nl = 4. That is below the minimum clock
    size for many systems and HHL_solver then prints a resolution warning;
    it is harmless here because both solutions share the same clock
    register. Pass n_clock_qubits=None to use the minimum instead (slow).
    """
    from HHL_solver import HHL_solver

    ref = HHL_solver.from_pde_preper(prep, n_clock_qubits=n_clock_qubits,
                                     use_exact=True, verbose=False)
    w_ref = ref.solve()["solution"]
    nl = ref.nl

    errs = []
    for reps in reps_list:
        hhl = HHL_solver.from_pde_preper(prep, n_clock_qubits=nl,
                                         use_exact=False, trotter_reps=reps,
                                         trotter_order=order, verbose=False)
        w = hhl.solve()["solution"]
        errs.append(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref))

    h = 1.0 / np.asarray(reps_list, float)
    res = _result(h, errs, expected=float(order), reps=np.asarray(reps_list),
                  nl=nl)
    if verbose:
        rows = [(int(r), float(e)) for r, e in zip(reps_list, res["err"])]
        print_table(f"Trotter error vs exact HHL (nl = {nl}, order {order}); "
                    f"fitted order = {res['order']:.3f} (theory {order})",
                    ["reps", "rel. error"], rows)
    return res


# ======================================================================
# 3. Sampling error (finite shots)
# ======================================================================
def predicted_sampling_error(p_sel, shots):
    """
    Closed-form expected relative L2 error of |w| estimated from `shots`
    shots.

    Each post-selected probability p_j is estimated by a binomial
    frequency with variance p_j (1 - p_j) / shots, so the amplitude
    sqrt(p_j) has standard deviation about sqrt((1 - p_j) / (4 shots)).
    Summing over the solution entries and dividing by the norm sqrt(sum
    p_j) gives
        sqrt( sum_j (1 - p_j) / (4 shots) ) / sqrt( sum_j p_j ).
    Valid when shots * p_j >> 1. It scales as shots^(-1/2).
    """
    p_sel = np.asarray(p_sel, float)
    shots = np.asarray(shots, float)
    return np.sqrt(np.sum(1.0 - p_sel) / (4.0 * shots)) / np.sqrt(p_sel.sum())


def sampling_convergence(hhl, shots_list=None, repeats=20, method="multinomial",
                         seed=0, verbose=True):
    """
    Error of the shot-based solution vs. the number of shots.

    Two errors are reported:
      err_sampling : error against the EXACT statevector HHL magnitudes
                     |w_HHL|. This isolates sampling noise; theory says
                     it falls like shots^-1/2, and `err_predicted` gives
                     the closed-form value (see predicted_sampling_error).
      err_total    : error against the classical |w|. This flattens at the
                     HHL algorithmic floor (clock resolution, Trotter),
                     which is the part sampling cannot remove.

    Parameters
    ----------
    hhl        : HHL_solver instance (circuit built or not).
    shots_list : shot counts. Default: 10^3 ... 10^7 (multinomial).
    repeats    : independent repetitions averaged per shot count (rms).
    method     : "multinomial" - ideal sampling drawn from the exact
                 statevector probabilities (fast, noise-free; it is what a
                 perfect simulator returns), or "aer" - hhl.run_aer, which
                 re-transpiles the circuit on every call and is slow.
    seed       : RNG seed.
    """
    rng = np.random.default_rng(seed)
    if shots_list is None:
        shots_list = ([10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
                      if method == "multinomial"
                      else [10 ** 3, 10 ** 4, 10 ** 5])
    shots_list = [int(s) for s in shots_list]

    out = hhl.solve()
    amp = out["amp"]
    w_hhl = np.abs(out["solution"])                    # statevector |w|
    p = np.abs(amp) ** 2                               # P(anc=1, clock=0, sys=j)
    p_sel = p[: hhl.N_raw][hhl.N_raw // 2:]            # entries that form w
    probs = np.append(p, max(0.0, 1.0 - p.sum()))
    probs /= probs.sum()

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    e_samp, e_tot, e_pred = [], [], []
    for shots in shots_list:
        s_err, t_err = [], []
        for r in range(repeats):
            if method == "multinomial":
                counts = rng.multinomial(shots, probs)[:-1]
                amp_abs = np.sqrt(counts / shots)
                sol = np.abs(hhl._amplitudes_to_solution(amp_abs))
            elif method == "aer":
                sol = hhl.run_aer(shots=shots, seed=seed + r)["solution_abs"]
            else:
                raise ValueError("method must be 'multinomial' or 'aer'")
            s_err.append(rel(sol, w_hhl))
            t_err.append(hhl.compare(sol, magnitude=True)[1])
        e_samp.append(np.sqrt(np.mean(np.square(s_err))))
        e_tot.append(np.sqrt(np.mean(np.square(t_err))))
        e_pred.append(predicted_sampling_error(p_sel, shots))

    h = 1.0 / np.asarray(shots_list, float)
    res = _result(h, e_samp, expected=0.5, shots=np.asarray(shots_list),
                  err_total=np.asarray(e_tot), err_predicted=np.asarray(e_pred),
                  success_prob=float(p.sum()))
    if verbose:
        rows = [(int(s), float(a), float(b), float(c_))
                for s, a, b, c_ in zip(shots_list, e_samp, e_pred, e_tot)]
        print_table(f"Sampling error ({method}, {repeats} repeats, "
                    f"P(success) = {res['success_prob']:.4f}); fitted order "
                    f"in 1/shots = {res['order']:.3f} (theory 0.5)",
                    ["shots", "err sampling", "err predicted", "err total"],
                    rows)
    return res


# ======================================================================
# Driver
# ======================================================================
def run_all(c=1.0, Lx=1.0, T=1.0, mode=1, Nx=2, Nt=2, skip_quantum=False,
            save_dir=None, plot=True):
    """
    Run every study and print the tables.

    Parameters
    ----------
    c, Lx, T, mode : standing-wave problem (mode 1 = main.py default).
    Nx, Nt         : grid for the HHL runs (parts 2 and 3); Nt a power of 2.
    skip_quantum   : only run the classical discretization study.
    save_dir       : if given, save one PNG per study plus the summary.
    plot           : True (default) draws the combined summary figure at
                     the end; False skips it (tables only).

    Returns a dict with the results of each study.
    """
    def path(name):
        if save_dir is None:
            return None
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, name)

    results = {}

    # --- 1. discretization ---
    results["discretization"] = discretization_convergence(c, Lx, T, mode)

    if not skip_quantum:
        # --- 2. HHL algorithmic error ---
        prep = make_prep(c, Lx, Nx, T, Nt, mode)
        results["clock"] = hhl_clock_convergence(prep)
        results["trotter"] = hhl_trotter_convergence(prep)

        # --- 3. sampling error ---
        from HHL_solver import HHL_solver
        hhl = HHL_solver.from_pde_preper(prep, verbose=False)
        results["sampling"] = sampling_convergence(hhl)

    # --- one PNG per study (only when a save directory is given) ---
    if save_dir is not None:
        for sp in _panel_specs(results):
            plot_loglog(sp["x"], sp["curves"], sp["xlabel"], sp["title"],
                        expected_slope=sp.get("expected_slope"),
                        expected_label=sp.get("expected_label"),
                        invert_x=sp.get("invert_x", False),
                        save_path=path(f"convergence_{sp['name']}.png"))

    # --- summary figure at the end ---
    if plot:
        plot_summary(results, save_path=path("convergence_summary.png"),
                     show=True)
    return results


def build_parser(add_help=True):
    """
    Command-line options shared by convergence.py and main.py. main.py
    reuses it as a parent parser (add_help=False) and adds its own flags.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 add_help=add_help)
    ap.add_argument("--c", type=float, default=1.0, help="wave speed")
    ap.add_argument("--Lx", type=float, default=1.0, help="domain length")
    ap.add_argument("--T", type=float, default=1.0, help="final time")
    ap.add_argument("--mode", type=int, default=1,
                    help="standing-wave mode: u0 = sin(mode*pi*x/Lx)")
    ap.add_argument("--nx", type=int, default=2,
                    help="interior points for the HHL runs")
    ap.add_argument("--nt", type=int, default=2,
                    help="time steps for the HHL runs (power of 2)")
    ap.add_argument("--skip-quantum", action="store_true",
                    help="convergence study: classical discretization only")
    ap.add_argument("--save-dir", default=None,
                    help="convergence study: directory for PNG figures")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip all figures")
    return ap


def parse_args(argv=None):
    """Parse the shared command-line options."""
    return build_parser().parse_args(argv)

def parse_args(argv=None):
    """Shared options from convergence.py plus the flags specific to main."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 parents=[build_parser(add_help=False)])
    ap.add_argument("--shots-aer", type=int, default=200000,
                    help="shots for the local Aer run")
    ap.add_argument("--convergence", action="store_true",
                    help="also run the convergence study (convergence.py)")
    ap.add_argument("--ibm", action="store_true",
                    help="connect to IBM Quantum and estimate the QPU time")
    ap.add_argument("--submit", action="store_true",
                    help="with --ibm: actually submit the job (spends QPU time)")
    ap.add_argument("--job-id", default=None,
                    help="with --ibm: reload this finished job instead of submitting")
    ap.add_argument("--shots-ibm", type=int, default=10000,
                    help="shots for the IBM run")
    return ap.parse_args(argv)
