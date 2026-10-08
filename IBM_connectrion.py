from typing import Optional

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2 as Sampler,
)


def IBM_instance_key(token: str, instance: str, opl: int = 1,
                     backend_name: str = "ibm_phoenix", n_show: int = 5):
    """
    Save the IBM Cloud credentials, connect, and prepare a transpiler for one
    backend.

    Parameters
    ----------
    token        : IBM Cloud API key. Avoid hard-coding it in scripts; read it
                   from an environment variable instead, e.g.
                   os.environ["IBM_TOKEN"].
    instance     : IBM Cloud instance (CRN or name) to run jobs on.
    opl          : transpiler optimization level (0-3) for the pass manager.
    backend_name : backend to target. Must be available to your instance.
    n_show       : how many available backends to list as a sanity check.

    Returns
    -------
    service : QiskitRuntimeService (needed later to re-load saved jobs)
    backend : the selected backend
    isa_pm  : pass manager that converts circuits to the backend's native
              gates (ISA circuits), which hardware primitives require
    """
    # Store the credentials on disk (overwrites any previously saved account)
    # and make this account the default for QiskitRuntimeService().
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=token,
        instance=instance,
        overwrite=True,
        set_as_default=True,
    )

    # Verify the credentials by connecting and listing backends.
    service = QiskitRuntimeService()
    backends = service.backends()
    backend = service.backend(backend_name)
    print(f"Account OK. {len(backends)} backend(s) available:")
    for b in backends[:n_show]:
        print(f"  {b.name} ({b.num_qubits} qubits)")

    # Pass manager targeting the chosen backend (no scheduling needed here;
    # scheduling is only used for the duration estimate in Sampler_RUN).
    isa_pm = generate_preset_pass_manager(backend=backend, optimization_level=opl)
    return service, backend, isa_pm


def Sampler_RUN(qc, backend, isa_pm, service, IBM_label: str,
                SHOTS: int = 10000, SUBMIT_JOB: bool = False,
                JOB_ID: Optional[str] = None, estimate_time: bool = True,
                opl: int = 1, Qiskit_scheduling_method: str = "alap"):
    """
    Estimate the QPU time of a circuit and then submit it (or reload a
    previously submitted job) with the Sampler primitive.

    Parameters
    ----------
    qc            : circuit WITHOUT measurements; measure_all() is added to a
                    copy here.
    backend, isa_pm, service : as returned by `IBM_instance_key`.
    IBM_label     : job tag, makes the job easy to find in the IBM dashboard.
    SHOTS         : number of shots.
    SUBMIT_JOB    : must be True to actually submit. Default False, so a
                    stray call never spends QPU time.
    JOB_ID        : if given, reload that saved job instead of submitting a
                    new one (takes priority over SUBMIT_JOB).
    estimate_time : print the per-shot and total QPU time estimate first.
    opl           : optimization level used for the scheduled estimate
                    circuit.
    Qiskit_scheduling_method : "alap" or "asap", used for the estimate only.

    Returns
    -------
    (job, job_id) : the runtime job and its id, or (None, None) if nothing
                    was submitted or reloaded. Call job.result() to get data.
    """
    # Measure all qubits on a copy so the caller's circuit is untouched.
    qc_meas = qc.copy()
    qc_meas.measure_all()

    sampler = Sampler(mode=backend)
    sampler.options.environment.job_tags = [IBM_label]

    # ------------------------------------------------------------------
    # 1. QPU time estimate: per shot = circuit + overhead + repetition delay
    # ------------------------------------------------------------------
    if estimate_time:
        # Separate pass manager WITH scheduling, so the circuit has a duration.
        pm_sched = generate_preset_pass_manager(
            target=backend.target,
            optimization_level=opl,
            scheduling_method=Qiskit_scheduling_method,
        )
        scheduled_qc = pm_sched.run(qc_meas)

        dt = backend.target.dt                       # seconds per sample
        # A scheduled circuit reports its duration in dt units.
        circuit_duration = (scheduled_qc.duration * dt
                            if scheduled_qc.unit == "dt" else scheduled_qc.duration)
        init_overhead = dt                           # tiny; kept from the original formula
        rep_delay = (sampler.options.execution.rep_delay
                     or getattr(backend, "default_rep_delay", None))
        if rep_delay is None:
            print("Warning: backend has no default rep_delay; assuming 0 s.")
            rep_delay = 0.0

        per_shot_time = circuit_duration + init_overhead + rep_delay
        print(f"Per-shot time: {per_shot_time * 1e3:.1f} ms")
        print(f"Base QPU time per execution group: {per_shot_time * SHOTS:.1f} s")

    # ------------------------------------------------------------------
    # 2. Reload a saved job, submit a new one, or do nothing
    # ------------------------------------------------------------------
    if JOB_ID is not None:
        job = service.job(JOB_ID)
        print(f"Re-using saved sampler job: {JOB_ID}")
    elif SUBMIT_JOB:
        isa_qc = isa_pm.run(qc_meas)                 # native-gate (ISA) circuit
        job = sampler.run([isa_qc], shots=SHOTS)
        JOB_ID = job.job_id()
        print(f"Submitted sampler job: {JOB_ID}")
    else:
        print("Set SUBMIT_JOB=True to submit.")
        return None, None

    return job, JOB_ID
