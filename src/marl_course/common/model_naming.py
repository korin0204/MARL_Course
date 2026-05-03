"""Model-name helper used by training scripts and visualizers."""
from __future__ import annotations

import re


MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_[0-9]+$")


def default_model_name(student_id: object, algo: str, episodes: object) -> str:
    """Build a classroom-friendly model name such as `alice_dqn_10000`."""
    base = _safe_token(str(student_id), fallback="student")
    algo_token = _safe_token(algo, fallback="model")
    return f"{base}_{algo_token}_{int(episodes)}"


def validate_model_name(name: object) -> str:
    """Validate naming rule: `<name>_<algo>_<episodes>`.

    Examples:
    - `alice_dqn_10000`
    - `team01_qmix_5000`
    """
    value = str(name)
    if not MODEL_NAME_RE.fullmatch(value):
        raise ValueError(
            "model_name must look like '<name>_<algo>_<episodes>', "
            "for example 'alice_dqn_10000'. Use ASCII letters, digits, and underscores."
        )
    return value


def _safe_token(value: str, fallback: str) -> str:
    """Convert arbitrary text into an ASCII token for model naming."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token or fallback
