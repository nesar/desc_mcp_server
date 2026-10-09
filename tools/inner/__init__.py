"""Inner scripts for env-kernels (see tools/envkernel.py).

Keep this __init__ EMPTY and every module here free of server-only imports
(pydantic, matplotlib, mcp): these files run under a facility environment the
user chose, which only guarantees the DESC stack (pyccl, sacc, firecrown,
augur, ...) plus numpy. Each module exposes ``main(params: dict) -> dict``
with JSON-safe input and output; the same function is called in-process when
the server runs locally.
"""
