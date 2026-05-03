# Tests for evaluation entry points.
from marl_course.evaluation.bomber import run_bomber_tournament
from marl_course.evaluation.coop import run_coop_evaluation, run_coop_zero_shot_evaluation


def test_evaluators_return_leaderboards():
    """All evaluator entry points should return non-empty ranking outputs."""
    bomber = run_bomber_tournament(episodes=1, seed=3)
    coop = run_coop_evaluation(episodes=1, seed=3)
    zero = run_coop_zero_shot_evaluation(episodes=1, seed=3)
    assert bomber["leaderboard"]
    assert coop["leaderboard"]
    assert "heldout_family" in zero
