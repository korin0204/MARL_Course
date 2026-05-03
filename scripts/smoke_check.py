"""Quick end-to-end smoke checks for CI/local validation.

The goal is not score quality; we only verify that key pipelines execute
without crashing and return expected result keys.
"""
from __future__ import annotations

from marl_course.evaluation.bomber import run_bomber_tournament
from marl_course.evaluation.coop import run_coop_evaluation, run_coop_zero_shot_evaluation


def main() -> None:
    """Run tiny evaluation jobs and assert essential outputs exist."""
    bomber = run_bomber_tournament(episodes=1, seed=7)
    assert bomber["leaderboard"], "Bomber leaderboard is empty"
    coop = run_coop_evaluation(episodes=1, seed=7)
    assert coop["leaderboard"], "Coop leaderboard is empty"
    zero = run_coop_zero_shot_evaluation(episodes=1, seed=7)
    assert "heldout_family" in zero, "Zero-shot result missing heldout bucket"
    print("smoke_check: ok")
    print("bomber top:", bomber["leaderboard"][0])
    print("coop top:", coop["leaderboard"][0])
    print("zero-shot buckets:", ", ".join(zero))


if __name__ == "__main__":
    main()
