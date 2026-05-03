"""Config helpers shared by training/evaluation scripts.

Design goal:
- Keep experiment knobs in JSON for reproducibility and W&B logging.
- Allow selective CLI overrides for quick classroom experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path | None) -> dict[str, Any]:
    """Load a training/evaluation config.

    JSON is intentionally used instead of Python so the exact knobs used in a
    run can be copied into W&B config and shared with students.
    """

    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def cli_or_config(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    """Return CLI value when present, otherwise config value, otherwise default."""

    cli_value = getattr(args, key.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return config.get(key.replace("-", "_"), config.get(key, default))


def bool_cli_or_config(args: argparse.Namespace, config: dict[str, Any], key: str, default: bool = False) -> bool:
    """Resolve boolean settings with CLI priority."""
    cli_value = getattr(args, key.replace("-", "_"), None)
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get(key.replace("-", "_"), config.get(key, default)))


def dump_effective_config(out_dir: Path, config: dict[str, Any]) -> Path:
    """Persist the fully-resolved config used for this run.

    This file is important for debugging and grading because it records the
    exact hyperparameters used when generating a submitted model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "effective_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
