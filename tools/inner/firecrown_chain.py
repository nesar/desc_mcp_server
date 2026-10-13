"""Inner script: Cobaya MCMC over a firecrown TwoPointExperiment in PURE_CCL
mode (no Cobaya theory block; CCL computes everything) - env-kernel.

params (JSON):
  experiment_yaml : str   TwoPointExperiment YAML (ccl_factory.creation_mode
                          must be pure_ccl_mode)
  fixed           : dict  parameter values held fixed (firecrown names);
                          anything not in `priors` and not fixed takes the
                          firecrown default
  priors          : dict  {name: {"min": lo, "max": hi, "ref": x, "proposal": w}}
                          uniform priors for the sampled parameters
  max_samples     : int   cobaya mcmc max_samples
  rminus1_stop    : float Gelman-Rubin stopping criterion
  work_dir        : str   where the chain files are written (job CWD remotely)
  chain_prefix    : str   file prefix inside work_dir
  seed            : int | None
  resume          : bool  continue from the Cobaya checkpoint of a previous run
                          with the same prefix (walltime-bounded chains); the
                          sampler settings must match that run
  file_locking    : bool  Cobaya's portalocker output lock (default True). Set
                          False on filesystems without POSIX locks - NERSC CFS
                          raises OSError errno 524 - the wrapper does so for
                          every facility run.

returns: {"chain_files": [...], "n_samples", "n_chains", "acceptance_rate",
          "summary": {name: {"mean", "std", "lower68", "upper68"}},
          "sampled": [...], "converged": bool|None, "Rminus1": float|None}

Multi-node (contract R9, hep-genesis make-tools-dispatchable): dispatched
with nodes > 1 the engine launches nodes x DISPATCH_MPI tasks_per_node
copies of main(); each MPI process runs ONE Cobaya chain of max_samples,
R-1 is the real cross-chain Gelman-Rubin statistic, and rank 0 returns the
summary of all chains combined (first third of each chain skipped). Ranks
other than 0 return {"rank", "n_samples", "acceptance_rate"}. A resume must
use the same chain count (Cobaya checkpoints one chain per process).
"""

import os

# One MPI process per Cobaya chain, 8 cores each (CCL/CAMB are partly
# OpenMP-threaded; a DES-Y1 3x2pt likelihood is memory-light). Conservative
# on purpose: 4 chains fill a Polaris node (32 cores) and a quarter of a
# Perlmutter CPU node; run_pack_kernel's tasks_per_node/cpus_per_task
# override it. The wrapper mirrors this into DISPATCH_KERNELS.
DISPATCH_MPI = {"main": {"tasks_per_node": 4, "cpus_per_task": 8}}

from .firecrown_loglike import load_experiment, materialize_experiment, required_defaults

_FACTORY_SRC = '''"""Firecrown likelihood factory written by desc-mcp-server (firecrown_run_chain)."""
from firecrown.likelihood.factories import TwoPointExperiment
from firecrown.modeling_tools import ModelingTools


def build_likelihood(build_parameters):
    exp = TwoPointExperiment.load_from_yaml(build_parameters.get_string("likelihood_config"))
    return exp.make_likelihood(), ModelingTools(ccl_factory=exp.ccl_factory)
'''


def _progress_line(progress_path: str, max_samples: int) -> str | None:
    """'N accepted samples of M, acceptance 0.25, R-1 0.048' from Cobaya's
    <prefix>.progress table (None until the first checkpoint)."""
    try:
        with open(progress_path, encoding="utf-8") as fh:
            rows = [ln.split() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError:
        return None
    if not rows:
        return None
    last = rows[-1]
    try:
        n, acc, r = int(float(last[0])), float(last[2]), float(last[3])
    except (IndexError, ValueError):
        return None
    return f"{n} accepted samples of {max_samples}, acceptance {acc:.2f}, R-1 {r:.3g}"


def _start_progress_writer(progress_path: str, out_path: str, max_samples: int, period_s: float = 60.0):
    """Background thread: every period, rewrite out_path with the latest
    one-liner. Returns a stop() that writes the final line."""
    import threading

    halt = threading.Event()

    def write_once() -> None:
        line = _progress_line(progress_path, max_samples) or f"sampling (max {max_samples} samples)"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def loop() -> None:
        write_once()
        while not halt.wait(period_s):
            write_once()

    t = threading.Thread(target=loop, name="chain-progress", daemon=True)
    t.start()

    def stop() -> None:
        halt.set()
        t.join(timeout=5)
        write_once()

    return stop


def _mpi_world():
    """(comm, rank, size) when this process runs under the engine's MPI
    runner, which initialises mpi4py before importing the kernel; (None, 0, 1)
    otherwise. Never imports mpi4py itself: a bare import outside an MPI
    launch hangs on some facilities (Perlmutter IRI job steps, 2026-10-12)."""
    import sys

    mod = sys.modules.get("mpi4py.MPI")
    if mod is None:
        return None, 0, 1
    comm = mod.COMM_WORLD
    return comm, int(comm.Get_rank()), int(comm.Get_size())


def _chain_count(work_dir: str, prefix: str) -> int:
    """Chains <prefix>.<k>.txt a previous run of this prefix left in work_dir."""
    import re

    pat = re.compile(re.escape(prefix) + r"\.(\d+)\.txt$")
    try:
        return sum(1 for f in os.listdir(work_dir) if pat.match(f))
    except OSError:
        return 0


def _weighted_summary(names, values, weights) -> dict:
    import numpy as np

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = {}
    for j, name in enumerate(names):
        x = np.asarray(values[:, j], dtype=float)
        mean = float((w * x).sum())
        std = float(np.sqrt(max((w * (x - mean) ** 2).sum(), 0.0)))
        order = np.argsort(x)
        cdf = np.cumsum(w[order])
        lo = float(x[order][np.searchsorted(cdf, 0.16)]) if len(x) > 2 else mean
        hi = float(x[order][min(np.searchsorted(cdf, 0.84), len(x) - 1)]) if len(x) > 2 else mean
        out[name] = {"mean": mean, "std": std, "lower68": lo, "upper68": hi}
    return out


def main(params: dict) -> dict:
    import warnings

    import numpy as np

    experiment_yaml = ""
    fixed = dict(params.get("fixed") or {})
    priors = dict(params.get("priors") or {})
    max_samples = int(params.get("max_samples", 200))
    rminus1_stop = float(params.get("rminus1_stop", 0.05))
    work_dir = os.path.abspath(params.get("work_dir") or os.getcwd())
    prefix = str(params.get("chain_prefix") or "chain")
    seed = params.get("seed")
    resume = bool(params.get("resume", False))
    if not params.get("file_locking", True):
        # must be set before cobaya.output is imported (read at lock time)
        os.environ["COBAYA_USE_FILE_LOCKING"] = "False"
    os.makedirs(work_dir, exist_ok=True)
    comm, rank, size = _mpi_world()
    use_mpi = size > 1

    # Rank 0 materialises the experiment (inline sacc/yaml are written into
    # the job CWD: one writer) and learns the required names; the others
    # receive both. Cobaya then builds its own likelihood on every rank.
    defaults: dict = {}
    if rank == 0:
        experiment_yaml = os.path.abspath(materialize_experiment(params))
        like, tools, _ = load_experiment(experiment_yaml)
        defaults = required_defaults(like, tools)
    if comm is not None:
        experiment_yaml, defaults = comm.bcast((experiment_yaml, defaults) if rank == 0 else None, root=0)
    unknown = sorted(k for k in list(fixed) + list(priors) if k not in defaults)
    if unknown:
        raise ValueError(f"parameters not used by this likelihood: {unknown}; "
                         f"required names: {sorted(defaults)}")
    if not priors:
        raise ValueError("priors is empty: give at least one sampled parameter "
                         "as {name: {min, max}}.")

    if resume:
        have = _chain_count(work_dir, prefix)
        if have and have != size:
            raise ValueError(
                f"resume=True with {size} chain(s), but the previous run of {prefix} left {have} chain "
                f"file(s) in {work_dir}: Cobaya resumes one chain per MPI process. Use the same nodes x "
                f"chains per node ({have} chains) or start over without resume.")

    factory_path = os.path.join(work_dir, f"{prefix}_factory.py")
    if rank == 0:
        with open(factory_path, "w", encoding="utf-8") as fh:
            fh.write(_FACTORY_SRC)
    if comm is not None:
        comm.Barrier()

    cobaya_params: dict = {}
    for name, default in defaults.items():
        if name in priors:
            spec = priors[name]
            lo, hi = float(spec["min"]), float(spec["max"])
            ref = float(spec.get("ref", fixed.get(name, default)))
            ref = min(max(ref, lo), hi)
            prop = float(spec.get("proposal", (hi - lo) / 20.0))
            cobaya_params[name] = {"prior": {"min": lo, "max": hi}, "ref": ref,
                                   "proposal": prop, "latex": name.replace("_", r"\_")}
        elif name in fixed:
            cobaya_params[name] = float(fixed[name])
        else:
            cobaya_params[name] = float(default) if not isinstance(default, list) else default

    from firecrown.connector.cobaya.likelihood import LikelihoodConnector

    info = {
        "likelihood": {"firecrown": {"external": LikelihoodConnector,
                                     "firecrownIni": factory_path,
                                     "build_parameters": {"likelihood_config": experiment_yaml}}},
        "params": cobaya_params,
        "sampler": {"mcmc": {"max_samples": max_samples, "Rminus1_stop": rminus1_stop,
                             "burn_in": 0, "max_tries": 10 * max(1, len(priors)) * 100}},
        "output": os.path.join(work_dir, prefix),
    }
    # resume continues from <prefix>.checkpoint (Cobaya refuses mismatched
    # settings); force restarts over any previous files of this prefix.
    info["resume" if resume else "force"] = True
    if seed is not None:
        info["sampler"]["mcmc"]["seed"] = int(seed)

    from cobaya.run import run

    # Human one-liner for whoever is waiting on this job (the dispatch engine
    # relays progress.txt from the job directory as its heartbeat; locally
    # firecrown_chain_status reads the same file). Cobaya knowledge stays
    # HERE, in the kernel, never in the client.
    # Cobaya writes <prefix>.progress from its main process only.
    stop = (_start_progress_writer(os.path.join(work_dir, f"{prefix}.progress"),
                                   os.path.join(os.getcwd(), "progress.txt"), max_samples)
            if rank == 0 else (lambda: None))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # no_mpi=True outside an MPI launch: Cobaya would otherwise import
            # mpi4py on its own (see _mpi_world).
            _updated, sampler = run(info, no_mpi=not use_mpi)
    finally:
        stop()

    own_n = int(len(sampler.collection))
    try:
        own_acc = float(sampler.get_acceptance_rate())
    except Exception:  # noqa: BLE001
        own_acc = None
    # combined=True concatenates every rank's chain (a collective call: all
    # ranks make it); the first third of each chain is dropped as burn-in
    # before concatenation, the Cobaya recommendation for combined chains.
    skip = 0.33 if use_mpi else 0
    sample = sampler.products(combined=use_mpi, skip_samples=skip)["sample"]
    chain_samples = comm.gather(own_n, root=0) if comm is not None else [own_n]
    chain_acc = comm.gather(own_acc, root=0) if comm is not None else [own_acc]
    if rank != 0:
        return {"rank": rank, "n_samples": own_n, "acceptance_rate": own_acc}
    sampled = list(priors)
    # all weighted samples (cobaya already removed burn_in=0)
    data = sample.data if hasattr(sample, "data") else sample
    weights = np.asarray(data["weight"], dtype=float)
    values = np.column_stack([np.asarray(data[n], dtype=float) for n in sampled])
    summary = _weighted_summary(sampled, values, weights)
    try:  # getdist means/stds when available (same numbers, canonical tool)
        gd = sample.to_getdist()
        means = gd.getMeans()
        stds = np.sqrt(gd.getVars())
        for j, name in enumerate(gd.getParamNames().list()):
            if name in summary:
                summary[name]["getdist_mean"] = float(means[j])
                summary[name]["getdist_std"] = float(stds[j])
        used_getdist = True
    except Exception:  # noqa: BLE001 - getdist optional
        used_getdist = False

    n_samples = int(len(weights))
    accs = [a for a in chain_acc if a is not None]
    acc = float(sum(accs) / len(accs)) if accs else None
    rminus1 = None
    converged = None
    try:
        rminus1 = float(sampler.Rminus1_last) if hasattr(sampler, "Rminus1_last") else None
        converged = bool(sampler.converged) if hasattr(sampler, "converged") else None
    except Exception:  # noqa: BLE001
        pass

    chain_files = sorted(os.path.join(work_dir, f) for f in os.listdir(work_dir)
                         if f.startswith(prefix + ".") or f == os.path.basename(factory_path))
    loglike_col = None
    for col in ("minuslogpost", "minuslogprior", "chi2"):
        if col in data:
            loglike_col = col
            break
    best = None
    if "minuslogpost" in data:
        i = int(np.argmin(np.asarray(data["minuslogpost"])))
        best = {n: float(data[n].iloc[i]) if hasattr(data[n], "iloc") else float(data[n][i])
                for n in sampled}
        best["minuslogpost"] = float(np.asarray(data["minuslogpost"])[i])
    return {"chain_files": chain_files, "n_samples": n_samples, "n_weighted": float(weights.sum()),
            "n_chains": size, "chain_samples": [int(n) for n in chain_samples], "skip_samples": skip,
            "acceptance_rate": acc, "chain_acceptance": chain_acc, "summary": summary, "sampled": sampled,
            "fixed": {k: v for k, v in cobaya_params.items() if not isinstance(v, dict)},
            "converged": converged, "Rminus1": rminus1, "used_getdist": used_getdist,
            "best_fit": best, "loglike_column": loglike_col, "factory_file": factory_path}
