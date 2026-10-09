---
name: hpc-dispatch-handoff
description: Run this server's heavy kernels (Fisher forecasts, likelihood scans/chains, TXPipe pipelines, large CCL grids) on ALCF Polaris or NERSC Perlmutter through the hep-genesis engine - export the pack, let the CLIENT's facility tools run it under the user's own credentials and chosen environment
---

# HPC dispatch handoff (client-side execution)

Use this when the user wants a computation from THIS server executed on a
DOE facility. This server holds no facility credentials, project, workdir or
environment settings by design - those live with the client's hep-genesis
harness, per user. All facility access goes through IRI (job submission) and
Globus or the IRI filesystem API (staging); nothing here writes batch scripts
or opens SSH sessions.

## Two kinds of kernels in the pack

| Kind | Examples | Node-side needs | How the client runs it |
|---|---|---|---|
| **pip-kernel** | `ccl.kernels.*` (pure pyccl) | `pip_deps` from the manifest (`pyccl`, `camb` wheels) | `run_pack_kernel(function=<kernel>, args={...}, pip_deps=[...])` |
| **env-kernel** | firecrown loglike/scan/chain, augur forecast, TXPipe pipeline | a facility environment containing the DESC stack, named by the user in `env_setup` | `run_pack_kernel(function="envkernel.run_in_env", args={"env_setup": "...", "inner": "<name>", "params": {...}}, pip_deps=[])` |

firecrown and numcosmo are conda-only (not on PyPI), which is why env-kernels
exist: the engine's venv + pip bootstrap cannot build them on a node.

## Steps

1. Call `export_dispatch_pack` (this server). The client saves the kernel
   files locally and shows you `pack_path` plus the `kernels` manifest and
   `env_kernel.env_setup_candidates`.
2. Pick the kernel for the task from the manifest. For an env-kernel, ask the
   user which facility environment to use, offering the candidates:
   - NERSC: `desc-python` (`source /global/common/software/lsst/common/miniconda/setup_current_python.sh`;
     firecrown 1.15, pyccl, sacc, tjpcov) or `desc-cosmology`
     (`source $CFS/lsst/groups/MCP/setup-cosmology.sh`). For TXPipe: the
     user's own TXPipe env + checkout (`module load python; conda activate <env>; cd <checkout>`).
   - ALCF: no DESC-maintained stack exists; the user supplies the activation
     of an env they built from `desc-cosmology-env` (see ENVIRONMENT.md).
   Never guess an `env_setup`; never reuse one user's choice for another.
3. On the CLIENT's facility server (hep-genesis-alcf / hep-genesis-nersc),
   call `run_pack_kernel` once per run:
   - `pack` = the exact `pack_path` (ends in `/tools`) - never a run/output directory.
   - pip-kernel: `function` and `pip_deps` (`base_pip_deps` + the kernel's `pip_deps`) from the manifest.
   - env-kernel: `function="envkernel.run_in_env"`, `args={"env_setup": ..., "inner": <manifest inner>, "params": <the kernel's params>}`, `pip_deps=[]`.
   - `duration` >= the manifest's `duration_hint_s`; `artifact_dir` = the client run directory.
4. Read the result: for env-kernels the kernel result is under
   `result["result"]` and `result["env_check"]` lists the versions found in the
   chosen environment - if `firecrown`/`pyccl` are `null` there, the
   `env_setup` did not activate what was expected; fix it before anything else.
   A result string starting with `Error` is a kernel failure with the traceback;
   do NOT blind-retry.
5. Verify the result's `host` field names a compute node before telling the
   user it ran on the facility.

## Files and plots

This server cannot read the client's files, so never pass client-side paths
into this server's plotting/inspection tools. Env-kernels return numbers (and
small sacc files base64-inline) in `results.json`; the CLIENT saves them.
Because the kernels are deterministic, recomputing the same call LOCALLY on
this server reproduces facility numbers - do that for figures and say so.

## Server-side mode

`set_dispatch` on this server is only for deployments running on the user's
own machine with the hep-genesis backend installed (check `auth_status`).
Then the heavy tools dispatch themselves; env-kernel tools take `env_setup`
as an argument. It never affects, and is never affected by, other servers'
dispatch state.
