"""Run registry: every result is a JSON file with config hash, git SHA and machine."""

from __future__ import annotations

import datetime as dt
import json
import platform
import socket
import subprocess
from pathlib import Path

from engram_core.space import digest_hex

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v2"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_run(name: str, config: dict, metrics: dict, extra: dict | None = None) -> Path:
    now = dt.datetime.now(dt.timezone.utc)
    cfg_hash = digest_hex(config, "engram.run.v2", n_bytes=6)
    run_id = f"{now:%Y%m%dT%H%M%S}-{name}-{cfg_hash}"
    rec = {
        "run_id": run_id, "name": name, "utc": now.isoformat(),
        "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "branch": _git("branch", "--show-current"),
        "host": socket.gethostname(), "platform": platform.platform(),
        "config_hash": cfg_hash, "config": config, "metrics": metrics, "extra": extra or {},
    }
    out = RESULTS / f"{now:%Y-%m-%d}" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str))
    return out
