"""Helpers for loading student submission bundles.

A submission directory is expected to contain:
- policy.py: model loader entry point
- policy.pt: weights/checkpoint
- metadata.json: identifiers and environment compatibility
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LoadedPolicy:
    """Normalized submission object used by evaluators."""

    student_id: str
    policy: Any
    metadata: dict[str, Any]
    path: Path


def load_policy_from_dir(path: str | Path, device: str = "cpu") -> LoadedPolicy:
    """Import and instantiate a policy from one submission directory.

    Loading priority:
    1) module.load_policy(model_path, device=...)
    2) module.Policy()
    3) module.TeamPolicy()
    """
    submission_dir = Path(path)
    metadata_path = submission_dir / "metadata.json"
    policy_path = submission_dir / "policy.py"
    model_path = submission_dir / "policy.pt"

    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy.py in {submission_dir}")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    module_name = f"student_policy_{submission_dir.name}_{abs(hash(submission_dir))}"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {policy_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(submission_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path

    if hasattr(module, "load_policy"):
        policy = module.load_policy(str(model_path), device=device)
    elif hasattr(module, "Policy"):
        policy = module.Policy()
    elif hasattr(module, "TeamPolicy"):
        policy = module.TeamPolicy()
    else:
        raise AttributeError("policy.py must define load_policy(), Policy, or TeamPolicy")

    return LoadedPolicy(
        student_id=str(metadata.get("student_id", submission_dir.name)),
        policy=policy,
        metadata=metadata,
        path=submission_dir,
    )
