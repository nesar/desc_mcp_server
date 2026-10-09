"""Smoke-test inner script: echoes its params (used by tests and auth checks)."""
import platform


def main(params: dict) -> dict:
    return {"echo": params, "platform": platform.platform(), "host": platform.node()}
