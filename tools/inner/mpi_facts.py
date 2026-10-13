"""Inner script: facts check for multi-node MPI dispatch (phase 0 of the
multi-node plan) — runs INSIDE a 2-node facility job under the pack's lock
env and reports what the node really supports. stdlib only.

params: {"ranks": 2, "build_mpi4py": true, "timeout_s": 900}
returns: {"site", "host", "python", "probes": {name: {"cmd", "rc", "out"}},
          "mpi4py_import": {...}, "hello": [...], "gpus": ...}
"""

import os
import signal
import time
import platform
import subprocess
import sys


# run-time prelude for anything that loads the Cray-built MPI.so: the module
# sets CRAY_LD_LIBRARY_PATH but not LD_LIBRARY_PATH (Polaris 7737509)
_MODS = ('module load cray-mpich 2>/dev/null; '
         'export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:+$CRAY_LD_LIBRARY_PATH:}$LD_LIBRARY_PATH";')


# library dirs for the build: every Cray PE dir plus the SYSTEM libfabric
# dir. Polaris 7737558: python's own RPATH ($ORIGIN/../lib) is consulted for
# every transitive load, so Cray libmpi picked up the conda env's libfabric
# (shipped with the lock's Open MPI) -> "fi_getinfo() failed: No data
# available" whenever python ran from the env; baking the Cray libfabric dir
# into MPI.so's RPATH makes that lookup win (loader chain beats the executable).
_LIBS = ('FABRIC_SO=$(ldconfig -p 2>/dev/null | awk \'/libfabric\\.so\\.1 /{print $NF; exit}\'); '
         'FABRIC_DIR=${FABRIC_SO:+$(dirname "$FABRIC_SO")}; '
         '[ -n "$FABRIC_DIR" ] && [ -d "$FABRIC_DIR" ] || FABRIC_DIR=$(ls -d /opt/cray/libfabric/*/lib64 2>/dev/null | tail -1); '
         'CRAY_LIBS="${FABRIC_DIR:+$FABRIC_DIR:}$CRAY_LD_LIBRARY_PATH"; echo "CRAY_LIBS=$CRAY_LIBS";')


def _clean_env() -> dict:
    """The environment the facility toolchain expects: the lock env's bin
    dirs OFF the PATH and its pkg-config/compiler variables unset. Loading
    Cray modules after a conda activation broke on Perlmutter ("Error
    invoking pkg-config!", job 59680714); the engine's run.sh will therefore
    load modules and resolve the Cray binaries BEFORE activating the env.
    Python is still reached by its absolute path."""
    env = os.environ.copy()
    prefix = sys.prefix.rstrip("/")
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":")
                           if p and not p.startswith(prefix + "/") and "/hep-genesis-mm/" not in p)
    # keep the Cray PE's pkg-config dirs (the cc driver resolves loaded modules
    # through them: darshan, cudatoolkit... - Perlmutter 59681422 failed with
    # "Error invoking pkg-config!" when the whole variable was dropped)
    if env.get("PKG_CONFIG_PATH"):
        env["PKG_CONFIG_PATH"] = ":".join(p for p in env["PKG_CONFIG_PATH"].split(":") if p and not p.startswith(prefix + "/"))
    for k in ("CC", "CXX", "FC", "LD", "AR", "RANLIB", "NM", "STRIP", "OBJCOPY", "CFLAGS", "CXXFLAGS",
              "LDFLAGS", "CPPFLAGS", "LDSHARED", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_BUILD_SYSROOT",
              "GSETTINGS_SCHEMA_DIR"):
        env.pop(k, None)
    return env


_STAGE = {"name": "start"}


def _stage(name: str) -> None:
    """Mark the current probe on stdout AND in progress.txt (contract R7), so
    a job that dies at the walltime still says where it was (Perlmutter
    59687887 hung for 25 min with nothing after "Running runner.py...")."""
    _STAGE["name"] = name
    print(f"[mpi_facts] {name} ({time.strftime('%H:%M:%S')})", flush=True)
    try:
        with open("progress.txt", "w", encoding="utf-8") as fh:
            fh.write(f"mpi facts: {name}\n")
    except OSError:
        pass


def _run(cmd: str, timeout: int = 300, env: dict | None = None) -> dict:
    """Run *cmd* under a login bash in its own process group; on timeout kill
    the WHOLE group (a hung grandchild - python under srun, pip - would
    otherwise hold the pipes open and subprocess.run would block forever
    after killing bash alone: that is how 59687887 reached the walltime)."""
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env or _clean_env(), start_new_session=True)
    try:
        so, se = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            so, se = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            so, se = "", ""
        tail = (so.strip()[-1200:] + ("\n[stderr]\n" + se.strip()[-600:] if se.strip() else "")).strip()
        return {"cmd": cmd, "rc": -1, "out": f"timed out after {timeout}s" + (f"; partial output:\n{tail}" if tail else "")}
    # stdout keeps its own budget: a chatty stderr (login-shell noise) must
    # not push the probe's answer out of the record
    out = (so.strip()[-3200:] + ("\n[stderr]\n" + se.strip()[-800:] if se.strip() else "")).strip()
    return {"cmd": cmd, "rc": p.returncode, "out": out}


_HELLO = (
    "import os, socket, sys\n"
    "keys = ['SLURM_PROCID','SLURM_NTASKS','SLURM_NODEID','PALS_RANKID','PALS_LOCAL_RANKID','PALS_NODEID',"
    "'PMI_RANK','PMI_SIZE','PMI_LOCAL_RANK','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']\n"
    "info = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}\n"
    "try:\n"
    "    from mpi4py import MPI\n"
    "    c = MPI.COMM_WORLD; info['mpi_rank'] = c.Get_rank(); info['mpi_size'] = c.Get_size()\n"
    "    info['mpi_lib'] = MPI.Get_library_version().strip().splitlines()[0][:80]\n"
    "except Exception as e:\n"
    "    info['mpi4py_error'] = repr(e)[:200]\n"
    "info['host'] = socket.gethostname(); info['python'] = sys.executable\n"
    "try:\n"
    "    info['libfabric'] = next((l.split()[-1] for l in open('/proc/self/maps') if 'libfabric.so' in l), None)\n"
    "except Exception:\n"
    "    pass\n"
    "print('HELLO ' + repr(info), flush=True)\n"
)


def main(params: dict) -> dict:
    _stage("main entered")
    ranks = int(params.get("ranks", 2))
    timeout = int(params.get("timeout_s", 900))
    site = "perlmutter" if os.environ.get("SLURM_JOB_ID") else ("polaris" if os.environ.get("PBS_JOBID") else "unknown")
    out: dict = {"site": site, "host": platform.node(), "python": sys.executable, "prefix": sys.prefix,
                 "probes": {}, "hello": [], "gpus": None, "mpi4py_import": None}
    P = out["probes"]
    _stage("modules")
    P["modules"] = _run("module list 2>&1 | head -40; echo ---; module avail cray-mpich 2>&1 | head -10")
    # The lock env is first on PATH and ships its own (conda-forge Open MPI) cc
    # and mpiexec, so the Cray toolchain must be resolved EXPLICITLY, never by
    # bare name (Polaris facts job 7737068: conda mpiexec rejected --ppn).
    resolve = (
        'module load cray-mpich 2>/dev/null; module load PrgEnv-gnu 2>/dev/null; '
        f'CRAY_CC=$(which -a cc 2>/dev/null | grep -v "{sys.prefix}" | grep -m1 "/opt/cray"); '
        f'LAUNCHER=$(which -a mpiexec srun 2>/dev/null | grep -v "{sys.prefix}" | grep -m1 -E "pals|slurm|/usr/bin/srun|/opt/cray"); '
        'echo "CRAY_CC=$CRAY_CC"; echo "LAUNCHER=$LAUNCHER"; echo "---which -a cc:"; which -a cc 2>/dev/null; '
        'echo "---which -a mpiexec srun:"; which -a mpiexec srun 2>/dev/null; "$CRAY_CC" --version 2>&1 | head -1'
    )
    _stage("load_mpich")
    P["load_mpich"] = _run(resolve)
    cray_cc = next((ln.split("=", 1)[1] for ln in P["load_mpich"]["out"].splitlines() if ln.startswith("CRAY_CC=")), "")
    launcher = next((ln.split("=", 1)[1] for ln in P["load_mpich"]["out"].splitlines() if ln.startswith("LAUNCHER=")), "")
    out["cray_cc"], out["launcher"] = cray_cc, launcher
    P["prgenv"] = _run("module list 2>&1 | grep -i prgenv; module load PrgEnv-gnu 2>&1 && echo PRGENV_GNU_OK")
    _stage("job_shape")
    P["job_shape"] = _run('echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"; '
                          'echo "SLURM_PROCID=${SLURM_PROCID:-unset} SLURM_STEP_ID=${SLURM_STEP_ID:-unset} SLURM_NTASKS=${SLURM_NTASKS:-unset} '
                          'SLURM_STEP_NUM_TASKS=${SLURM_STEP_NUM_TASKS:-unset} SLURM_LAUNCH_NODE_IPADDR=${SLURM_LAUNCH_NODE_IPADDR:-unset}"; '
                          '[ -n "${PBS_NODEFILE:-}" ] && echo "PBS nodes: $(wc -l < "$PBS_NODEFILE")" && cat "$PBS_NODEFILE"')
    out["gpus"] = _run("nvidia-smi -L 2>&1 | head -8; echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}")
    _stage("conda_mpi")
    # informational only (the lock's Open MPI), so a short leash: under a
    # one-task Slurm step its PMI probing may block
    P["conda_mpi"] = _run(f'"{sys.executable}" -c "import mpi4py, mpi4py.MPI as M; print(mpi4py.__version__, M.Get_library_version().splitlines()[0][:80])" 2>&1 | tail -2', timeout=60)

    # Build mpi4py against cray-mpich with the CRAY compiler driver into a
    # PROBE PREFIX (the shared lock env is not mutated by this facts check);
    # the hello below puts that prefix first on PYTHONPATH.
    probe = os.path.join(os.getcwd(), "mpi4py_probe")
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    probe_site = os.path.join(probe, "lib", pyver, "site-packages")
    if params.get("build_mpi4py", True) and cray_cc:
        # 1) can the Cray driver compile + link an MPI program at all (and as a
        #    shared object, which is what mpi4py needs)?
        probe_c = os.path.join(os.getcwd(), "mpi_probe.c")
        with open(probe_c, "w", encoding="utf-8") as fh:
            fh.write("#include <mpi.h>\n#include <stdio.h>\nint main(int c,char**v){MPI_Init(&c,&v);int r;"
                     "MPI_Comm_rank(MPI_COMM_WORLD,&r);printf(\"rank %d\\n\",r);MPI_Finalize();return 0;}\n")
        # Perlmutter 59687223: the pip link step `cc -shared _configtest.o -o
        # _configtest` died with "cannot find -lpmi/-lpmi2" although the same
        # driver linked mpi_probe.so one probe earlier -> give the linker every
        # Cray PE library dir explicitly (-L) as well as the RPATHs, and record
        # the verbose link lines under the overrides so the diff is visible.
        libdirs = '-L${CRAY_LIBS//:/ -L}'
        rpath = '-Wl,-rpath,${CRAY_LIBS//:/ -Wl,-rpath,}'
        # Polaris 7737546: RPATH worked, but the driver also linked
        # libsci_gnu_mpi (cray-libsci is loaded by default), which NEEDS a
        # libmpi_gnu_123.so.12 that exists nowhere on the node -> MPI.so could
        # not load. mpi4py needs no BLAS: unload cray-libsci for the build and
        # link --as-needed so only libraries actually used become NEEDED.
        mpild = f"{cray_cc} -shared -Wl,--as-needed {libdirs} {rpath}"
        overrides = (f'CC="{cray_cc}" MPICC="{cray_cc}" MPILD="{mpild}" LDSHARED="{mpild}" '
                     'LDFLAGS="" CRAY_ADD_RPATH=yes')
        _stage("cc_probe")
        P["cc_probe"] = _run(
            f"module load cray-mpich 2>/dev/null; module load PrgEnv-gnu 2>/dev/null; {_LIBS} "
            f'echo "--- cc -show:"; "{cray_cc}" -craype-verbose -c -o /dev/null "{probe_c}" 2>&1 | head -5; '
            f'"{cray_cc}" -o mpi_probe "{probe_c}" && echo LINK_OK && "{cray_cc}" -shared -fPIC -o mpi_probe.so "{probe_c}" && echo SHARED_OK; '
            f'echo "--- two-step link, verbose:"; "{cray_cc}" -c -fPIC -o mpi_probe.o "{probe_c}" && '
            f'"{cray_cc}" -shared -craype-verbose mpi_probe.o -o mpi_probe2.so 2>&1 | cut -c1-900 | head -4 && echo TWO_STEP_OK; '
            f'echo "--- two-step link under the build overrides:"; env {overrides} "{cray_cc}" -shared mpi_probe.o -o mpi_probe3.so 2>&1 | head -4 && echo TWO_STEP_OVERRIDES_OK; '
            'echo "--- pkg-config:"; pkg-config --libs mpich 2>&1 | cut -c1-300; pkg-config --libs cray-pmi 2>&1 | cut -c1-300; '
            'echo "--- env:"; env | grep -E "^(PE_ENV|CRAYPE|CRAY_MPICH|MPICH_DIR|PKG_CONFIG_PATH|CRAY_LD_LIBRARY_PATH|PE_PKGCONFIG|CRAY_PMI|PE_PMI)" | cut -c1-400 | head -14')
        # 2) the build. conda python's baked sysconfig LDSHARED carries
        #    "-B <env>/compiler_compat", which makes gcc link with the env's own
        #    old binutils ld (cannot resolve Cray libfabric/darshan symbols,
        #    Polaris 7737088); "cc -shared" as MPICC trips the Cray driver's
        #    argument parsing; CRAY_ADD_RPATH is not honoured for -shared so the
        #    extension carried no RPATH (Polaris 7737509). Hence: plain Cray
        #    driver for CC/MPICC, "$CRAY_CC -shared -L... -Wl,-rpath,..." for
        #    MPILD/LDSHARED (one word per variable: Polaris 7737536 had nested
        #    quotes), no LDFLAGS, darshan unloaded.
        pip_cmd = (f'PYTHONUNBUFFERED=1 "{sys.executable}" -m pip install -v --no-cache-dir --no-binary=mpi4py --no-deps '
                   f'--ignore-installed --prefix "{probe}" "mpi4py>=4"')
        report = ('rc=$?; echo "pip rc=$rc"; grep -n -i -E "MPI C compiler|MPI LD|checking for|success|failure|error:|undefined reference|ld returned" '
                  '{log} | head -30; echo "--- log tail:"; tail -15 {log}; exit $rc')
        build = (f"module load cray-mpich 2>/dev/null; module load PrgEnv-gnu 2>/dev/null; module unload darshan cray-libsci 2>/dev/null; {_LIBS} "
                 f"env {overrides} {pip_cmd} > mpi4py_build.log 2>&1; " + report.format(log="mpi4py_build.log"))
        _stage("build_mpi4py")
        P["build_mpi4py"] = _run(build, timeout=timeout)
        if P["build_mpi4py"]["rc"] != 0:
            # 3) fallback: only mpi4py's own knobs (no CC/LDSHARED/CRAY_ADD_RPATH
            #    in the environment), darshan left loaded, CFLAGS pinned
            build2 = (f"module load cray-mpich 2>/dev/null; module load PrgEnv-gnu 2>/dev/null; {_LIBS} "
                      f'env MPICC="{cray_cc}" MPILD="{mpild}" CFLAGS="-O2 -fPIC" LDFLAGS="" {pip_cmd} > mpi4py_build2.log 2>&1; '
                      + report.format(log="mpi4py_build2.log"))
            _stage("build_mpi4py_fallback")
            P["build_mpi4py_fallback"] = _run(build2, timeout=timeout)
        _stage("mpi4py_import")
        # Perlmutter 59692573: a direct `python -c "from mpi4py import MPI"`
        # HUNG for 300 s. The kernel itself runs inside the IRI's job step
        # (SLURM_STEP_ID=0, SLURM_NTASKS=2, PMI_* set), so Cray MPICH's
        # MPI_Init tried to join that 2-task step instead of a singleton.
        # Run the import as its own 1-task step (what the engine will do).
        # Polaris (PALS, no step env) imports directly and worked (7737568).
        one_task = 'srun --overlap -N 1 -n 1 --export=ALL ' if site == "perlmutter" else ''
        out["mpi4py_import"] = _run(
            f'{_MODS} PYTHONPATH="{probe_site}" {one_task}"{sys.executable}" -c '
            '"from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0]); print(MPI.COMM_WORLD.Get_size())"; '
            f'echo "--- MPI.so rpath + libs:"; so=$(ls "{probe_site}"/mpi4py/MPI.*.so 2>/dev/null | head -1); '
            'readelf -d "$so" 2>/dev/null | grep -E "RPATH|RUNPATH|NEEDED" | head -12; ldd "$so" 2>/dev/null | grep -E "mpi|fabric|pmi|sci|not found" | head -8')
    else:
        P["build_mpi4py"] = {"cmd": "", "rc": -1, "out": "skipped: no Cray cc found (see load_mpich)"}
        out["mpi4py_import"] = None

    hello_py = os.path.join(os.getcwd(), "mpi_hello_probe.py")
    with open(hello_py, "w", encoding="utf-8") as fh:
        fh.write(_HELLO)
    env_pp = f'PYTHONPATH="{probe_site}"'
    if site == "perlmutter":
        # --overlap: NERSC's IRI runs run.sh itself as a job step spanning all
        # nodes, so a plain nested srun waits forever for the resources
        # (59687619: "timed out after 240s")
        launch = (f'{_MODS} export {env_pp}; srun --overlap -N "${{SLURM_JOB_NUM_NODES:-1}}" -n {ranks} '
                  f'--ntasks-per-node=1 --cpus-per-task=4 --cpu-bind=cores --export=ALL "{sys.executable}" "{hello_py}"')
        gpu_launch = (f'{_MODS} export {env_pp}; srun --overlap -N "${{SLURM_JOB_NUM_NODES:-1}}" -n {ranks} '
                      f'--ntasks-per-node=1 --gpus-per-task=1 --export=ALL "{sys.executable}" "{hello_py}"')
    elif site == "polaris" and launcher:
        # PALS copies the executable to the nodes by default (sys.executable was
        # /var/run/palsd/.../files/0/python in 7737546): prefer --no-transfer,
        # fall back to the default if this PALS has no such flag
        launch = (f'{_MODS} export {env_pp}; "{launcher}" --no-transfer -n {ranks} --ppn 1 --hostfile "$PBS_NODEFILE" '
                  f'--depth 4 --cpu-bind depth --envall "{sys.executable}" "{hello_py}" || '
                  f'{{ echo "--- retry without --no-transfer"; "{launcher}" -n {ranks} --ppn 1 --hostfile "$PBS_NODEFILE" '
                  f'--depth 4 --cpu-bind depth --envall "{sys.executable}" "{hello_py}"; }}')
        gpu_launch = (f'{_MODS} export {env_pp}; "{launcher}" -n {ranks} --ppn 1 --hostfile "$PBS_NODEFILE" '
                      f'--depth 4 --cpu-bind depth --envall bash -c \'export CUDA_VISIBLE_DEVICES=$((PALS_LOCAL_RANKID % 4)); '
                      f'"{sys.executable}" "{hello_py}"\'')
    else:
        launch = gpu_launch = f'export {env_pp}; "{sys.executable}" "{hello_py}"'
    imp0 = out.get("mpi4py_import") or {}
    have_cray = imp0.get("rc") == 0 and ("MPICH" in imp0.get("out", "") or "CRAY" in imp0.get("out", "").upper())
    if not have_cray:
        # the conda Open MPI mpi4py hangs under srun (Perlmutter 59680714) and
        # runs as singletons under PALS (Polaris 7737080): not worth 10 minutes
        P["launch"] = {"cmd": launch, "rc": -1, "out": "skipped: no Cray-built mpi4py (see build_mpi4py)"}
        P["launch_gpu"] = {"cmd": gpu_launch, "rc": -1, "out": "skipped: no Cray-built mpi4py"}
        out["hello"], out["hello_gpu"] = [], []
    else:
        _stage("launch")
        P["launch"] = _run(launch, timeout=180)
        out["hello"] = [ln for ln in P["launch"]["out"].splitlines() if ln.startswith("HELLO ")]
        if "NVIDIA-SMI has failed" in (out["gpus"].get("out") or "") or "GPU 0" not in (out["gpus"].get("out") or ""):
            P["launch_gpu"] = {"cmd": gpu_launch, "rc": -1, "out": "skipped: no GPUs on these nodes (Perlmutter: constraint=gpu + the _g account)"}
            out["hello_gpu"] = []
        else:
            _stage("launch_gpu")
            P["launch_gpu"] = _run(gpu_launch, timeout=180)
            out["hello_gpu"] = [ln for ln in P["launch_gpu"]["out"].splitlines() if ln.startswith("HELLO ")]
    imp = out.get("mpi4py_import") or {}
    cray_lib = imp.get("rc") == 0 and ("MPICH" in imp.get("out", "") or "CRAY" in imp.get("out", "").upper())
    sizes = [ln.split("'mpi_size': ")[1].split(",")[0] for ln in out["hello"] if "'mpi_size': " in ln]
    out["summary"] = {
        "cray_cc": cray_cc, "launcher": launcher,
        "cray_mpich_loads": bool(cray_cc),
        # built AND it is the Cray library (the conda Open MPI one imports too but
        # cannot join srun/PALS: it ran as singletons on Polaris, job 7737080)
        "mpi4py_built": bool(cray_lib),
        "mpi4py_lib": imp.get("out", "").splitlines()[0][:80] if imp.get("out") else None,
        "ranks_see_each_other": bool(sizes) and all(s == str(ranks) for s in sizes),
        "hello_ranks": len(out["hello"]), "hello_gpu_ranks": len(out["hello_gpu"]),
        "distinct_hosts": len({ln.split("'host': '")[1].split("'")[0] for ln in out["hello"] if "'host': '" in ln}),
    }
    return out


# import-time marker: tells a timed-out job apart from one that never got here
print("[mpi_facts] module imported", flush=True)
