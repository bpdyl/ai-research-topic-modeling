"""
Configuration loading, path resolution, run directories and seeding.

Every script starts here. Nothing else in the package should hard-code a path,
a hyperparameter or a seed — if it is tunable, it belongs in
config/default.yaml so the run snapshot captures it.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# repo_root/src/nrtm/config.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """dict with attribute access and repo-relative path resolution."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a `paths.<key>` entry against the repository root."""
        return REPO_ROOT / self["paths"][key]


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config. `path` defaults to config/default.yaml."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["_config_path"] = str(cfg_path)
    return Config(data)


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on. numpy is imported lazily so the BLAS caps in
    __init__.py are always applied first."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str:
    """Short commit hash, so a run can be tied to the exact code that made it.
    Returns 'unknown' outside a git checkout rather than raising."""
    if shutil.which("git") is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@dataclass
class RunDir:
    """One experiment run's output directory.

    Layout: results/runs/<RUN_ID>/ containing config.yaml (the exact settings
    used), meta.json (timestamp + commit) and whatever artefacts the script
    writes. This is what marking criterion 2.4 asks for — "evidence of running
    the experiments provided in appendices".
    """

    path: Path
    run_id: str

    def file(self, name: str) -> Path:
        return self.path / name

    def write_json(self, name: str, obj: Any) -> Path:
        target = self.file(name)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        return target


def new_run(cfg: Config, tag: str) -> RunDir:
    """Create results/runs/<timestamp>_<tag>/ and snapshot the config into it."""
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    path = cfg.path("results_dir") / run_id
    path.mkdir(parents=True, exist_ok=True)

    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(snapshot, fh, sort_keys=False, allow_unicode=True)

    meta = {
        "run_id": run_id,
        "tag": tag,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source_config": cfg.get("_config_path"),
    }
    with open(path / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return RunDir(path=path, run_id=run_id)
