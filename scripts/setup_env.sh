#!/usr/bin/env bash
# Create the light "desc-mcp" Python environment that runs this server.
#
# Usage:  bash scripts/setup_env.sh [env-name]
#
# Holds pyccl + sacc + firecrown + augur + TJPCov + ceci (the parts of the DESC
# stack that coexist; see notes/survey_integration_env.md and ENVIRONMENT.md).
# TXPipe is deliberately NOT installed here: it pins firecrown 1.7.* and needs a
# compiled MPI/NaMaster/TreeCorr stack. TXPipe tools in this server compose
# pipelines and run them in a separate TXPipe env (local, optional) or on a
# facility through the hep-genesis dispatch engine.
#
# The upstream package clones (CCL/, CCLX/, TXPipe/, firecrown/, augur/) are
# never modified: augur is installed from a temporary copy of its clone.
set -euo pipefail

ENV_NAME="${1:-desc-mcp}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEP_GENESIS="${HEP_GENESIS_DIR:-$HOME/Projects/Tutorials/HEPKE/hep-genesis-agent}"

# conda-forge only (never mix with `defaults`); libmamba solver if available.
SOLVER_ARGS=()
if conda list -n base conda-libmamba-solver 2>/dev/null | grep -q conda-libmamba-solver; then
    SOLVER_ARGS=(--solver=libmamba)
fi

# idempotent: safe to rerun after a partial failure
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" -y "${SOLVER_ARGS[@]}" --override-channels -c conda-forge \
        "python=3.12" \
        "numpy>=2.0,<2.4" "scipy>=1.13" "camb<2.0" \
        "pyccl>=3.3.1" "sacc>=2.4" "firecrown>=1.16" \
        "tjpcov>=0.5" "lsstdesc-ceci>=2.4" "lsstdesc-crow" \
        "qp-prob" "healpy" "numdifftools" "jinja2" \
        "matplotlib-base" "h5py" "astropy" "pyyaml" "pip"
fi
ENV_PREFIX="$(conda info --base)/envs/$ENV_NAME"
PIP="$ENV_PREFIX/bin/pip"
PYTHON="$ENV_PREFIX/bin/python"

$PIP install --upgrade pip

# --- the MCP server's own needs + pip-only extras ------------------------
$PIP install "mcp[cli]>=1.27,<2" "pydantic>=2,<3" pytest anyio derivkit cobaya getdist isitgr

# --- augur from the clone (1.2.4 > conda-forge 1.2.3), WITHOUT touching it:
#     pip writes *.egg-info into a source tree it builds from, so build from
#     a throwaway copy. Rerun this script after pulling the clone to refresh.
AUGUR_BUILD="$(mktemp -d)/augur"
cp -R "$REPO_DIR/augur" "$AUGUR_BUILD"
rm -rf "$AUGUR_BUILD/.git"
$PIP install --no-deps "$AUGUR_BUILD"
rm -rf "$(dirname "$AUGUR_BUILD")"

# --- the server itself ---------------------------------------------------
$PIP install --no-deps -e "$REPO_DIR"

# --- HPC dispatch engine (optional; server-side dispatch mode only) -------
if [ -d "$HEP_GENESIS/backend" ]; then
    $PIP install -e "$HEP_GENESIS/backend[iri]" \
        || echo "WARNING: hep-genesis backend install failed; server-side dispatch unavailable (client-side handoff still works)."
else
    echo "NOTE: hep-genesis-agent not found at $HEP_GENESIS; server-side dispatch unavailable (client-side handoff still works)."
fi

echo
echo "Environment '$ENV_NAME' ready. Validate (from OUTSIDE the clone dirs) with:"
echo "  $PYTHON $REPO_DIR/tests/smoke_env.py"
