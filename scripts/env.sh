#!/usr/bin/env bash
# Build (or rebuild) this server's self-contained Python environment from the
# conda-lock file that ships inside the kernels package, using micromamba.
#
#   bash scripts/env.sh                # .micromamba/ + .mcp-env/ next to this repo, then smoke test
#   bash scripts/env.sh --relock       # re-solve tools/env/environment.yml -> conda-lock.yml first
#   bash scripts/env.sh --prefix DIR   # put the environment somewhere else
#   bash scripts/env.sh --no-smoke     # skip tests/smoke_env.py
#
# The SAME lock builds the environment on a compute node (the hep-genesis
# dispatch engine ships it with every job), so what runs here runs there.
# Server-agnostic on purpose: copy this file into any MCP server repo that
# keeps its lock at tools/env/conda-lock.yml (override with ENV_LOCK=...).
# Needs: bash, curl, tar. No conda. Works on macOS arm64 and Linux x86_64.
# The prefix is .mcp-env/ (NOT .env/: that name is the dotenv file hep-genesis
# and other tools look for, and a directory there shadowed it on 2026-10-09).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_LOCK="${ENV_LOCK:-$REPO_DIR/tools/env/conda-lock.yml}"
ENV_SRC="${ENV_SRC:-$REPO_DIR/tools/env/environment.yml}"
PREFIX="$REPO_DIR/.mcp-env"
MM_HOME="$REPO_DIR/.micromamba"
RELOCK=0
SMOKE=1
while [ $# -gt 0 ]; do
    case "$1" in
        --relock) RELOCK=1 ;;
        --prefix) PREFIX="$2"; shift ;;
        --no-smoke) SMOKE=0 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) PLAT=osx-arm64 ;;
    Darwin-x86_64) PLAT=osx-64 ;;
    Linux-x86_64) PLAT=linux-64 ;;
    Linux-aarch64) PLAT=linux-aarch64 ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 2 ;;
esac

# --- micromamba, private to this repo ----------------------------------------
export MAMBA_ROOT_PREFIX="$MM_HOME/root"
MM="$MM_HOME/bin/micromamba"
if [ ! -x "$MM" ]; then
    echo "Fetching micromamba ($PLAT) into $MM_HOME/bin ..."
    mkdir -p "$MM_HOME/bin"
    curl -fsSL --retry 3 -o "$MM.tmp" \
        "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$PLAT"
    chmod +x "$MM.tmp" && mv "$MM.tmp" "$MM"
fi
echo "micromamba $("$MM" --version)"

# --- optional: re-solve the lock from environment.yml ------------------------
if [ "$RELOCK" = 1 ]; then
    command -v uvx >/dev/null || { echo "--relock needs uv (https://docs.astral.sh/uv/)" >&2; exit 2; }
    # conda-lock drives a conda-compatible solver; micromamba 2.x is not
    # accepted by conda-lock 4 for cross-platform solves, so use conda/mamba
    # from PATH (SOLVER=... overrides). The lock itself needs only micromamba.
    SOLVER="${SOLVER:-$(command -v mamba || command -v conda || true)}"
    [ -n "$SOLVER" ] || { echo "--relock needs conda or mamba on PATH (SOLVER=/path/to/conda)" >&2; exit 2; }
    echo "Re-solving $ENV_SRC for linux-64 + osx-arm64 with $SOLVER (minutes) ..."
    # libmamba makes the solve minutes instead of an hour (classic conda solver)
    CONDA_SOLVER=libmamba uvx conda-lock lock --conda "$SOLVER" -f "$ENV_SRC" -p osx-arm64 -p linux-64 --lockfile "$ENV_LOCK"
fi
[ -f "$ENV_LOCK" ] || { echo "no lock at $ENV_LOCK (run with --relock)" >&2; exit 2; }

# --- the environment, from the lock (exact versions, no solve) ---------------
if [ -f "$PREFIX/.lock-sha" ] && [ "$(cat "$PREFIX/.lock-sha")" = "$(shasum -a 1 "$ENV_LOCK" | cut -c1-40)" ]; then
    echo "Environment $PREFIX already matches the lock."
else
    echo "Creating $PREFIX from $(basename "$ENV_LOCK") ..."
    rm -rf "$PREFIX"
    "$MM" create -y -p "$PREFIX" -f "$ENV_LOCK"
    shasum -a 1 "$ENV_LOCK" | cut -c1-40 > "$PREFIX/.lock-sha"
fi

# --- this server (not a dependency: not in the lock) -------------------------
"$PREFIX/bin/python" -m pip install --quiet --no-deps -e "$REPO_DIR"
if [ -n "${HEP_GENESIS_DIR:-}" ] && [ -d "$HEP_GENESIS_DIR/backend" ]; then
    # optional: server-side HPC dispatch from this host (the client-side
    # handoff needs nothing here)
    "$PREFIX/bin/python" -m pip install --quiet -e "$HEP_GENESIS_DIR/backend[iri]"
fi

echo
echo "Ready: $PREFIX/bin/python -m mcp_server [--transport streamable-http --port N]"
if [ "$SMOKE" = 1 ] && [ -f "$REPO_DIR/tests/smoke_env.py" ]; then
    (cd "$REPO_DIR" && "$PREFIX/bin/python" tests/smoke_env.py)
fi
