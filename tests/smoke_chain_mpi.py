"""Facility smoke test: a multi-node Cobaya chain (contract R9).

    .mcp-env/bin/python tests/smoke_chain_mpi.py perlmutter|polaris [NODES] [MAX_SAMPLES]

Builds the DES Y1 3x2pt likelihood locally, sets dispatch to the facility
and runs firecrown_run_chain with nodes=NODES (default 2): the engine
launches NODES x 4 MPI processes, each a Cobaya chain of MAX_SAMPLES
(default 300) over sigma8 and Omega_c; rank 0 returns the combined summary
and the cross-chain R-1. Checks n_chains, the per-chain sample counts, one
<prefix>.<k>.txt per chain and a finite R-1. Run it once with NODES=1 as the
baseline the multi-node means should agree with (same priors, more chains).

Costs allocation: NODES x ~30 min (the first run of a new lock builds the
environment and mpi4py; later runs reuse it). Needs the hep-genesis backend
in this env, its .env with the facility sign-in, and Globus Connect Personal.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(site: str, nodes: int, max_samples: int) -> int:
    from mcp_server.dispatch import set_dispatch
    from tools.firecrown_tools import (firecrown_build_likelihood, firecrown_list_examples,
                                       firecrown_run_chain)
    from tools.inner.firecrown_chain import DISPATCH_MPI

    sacc = ROOT / "firecrown" / "tests" / "sacc_data.hdf5"
    if not sacc.is_file():
        print(f"missing {sacc} (firecrown clone with tests)"); return 2
    out = Path.home() / "hep-genesis" / "artifacts" / f"chain_mpi_{site}_{nodes}n_{time.strftime('%y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    fiducial = firecrown_list_examples().metadata["des_y1_fiducial"]
    exp = firecrown_build_likelihood(output_dir=str(out), sacc_path=str(sacc)).metadata["experiment_yaml"]
    print(set_dispatch(site, artifact_dir=str(out)))
    per_node = int(DISPATCH_MPI["main"]["tasks_per_node"])
    expected = nodes * per_node
    print(f"[1] {site}: {nodes} node(s) x {per_node} chains = {expected} chains of {max_samples} samples; "
          f"artifacts -> {out}")
    t0 = time.time()
    r = firecrown_run_chain(output_dir=str(out), experiment_yaml=exp, nuisance=fiducial,
                            priors={"sigma8": {"min": 0.6, "max": 1.0, "ref": 0.8},
                                    "Omega_c": {"min": 0.15, "max": 0.4, "ref": 0.26}},
                            max_samples=max_samples, nodes=nodes, walltime_s=1800, seed=1, plot=False)
    dt = time.time() - t0
    m = r.metadata
    print(f"[2] finished in {dt:.0f}s on {m.get('computed_on')}: {r.message[:600]}")
    chains = sorted(f for f in m.get("chain_files", []) if f.rsplit(".", 2)[-1] == "txt" and f.rsplit(".", 2)[-2].isdigit())
    r1 = m.get("Rminus1")
    ok = True
    for label, cond in (
        (f"n_chains == {expected}", m.get("n_chains") == expected),
        (f"{expected} per-chain sample counts", len(m.get("chain_samples") or []) == expected),
        (f"{expected} chain files <prefix>.k.txt", len(chains) == expected),
        ("R-1 is a finite number", isinstance(r1, (int, float)) and math.isfinite(r1)),
        ("summary has both parameters", set(m.get("sampled") or []) == {"sigma8", "Omega_c"}),
    ):
        print(f"    {'ok' if cond else 'FAIL'}: {label}")
        ok = ok and bool(cond)
    print("[3] summary:", json.dumps(m.get("summary"), indent=1))
    print("PASS" if ok else "FAIL - see the checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "perlmutter"
    nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    sys.exit(main(site, nodes, max_samples))
