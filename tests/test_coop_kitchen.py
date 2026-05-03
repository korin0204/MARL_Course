# Tests for Coop Kitchen environment contracts.
from marl_course.envs.coop_kitchen import CoopGreedyTeamPolicy, CoopKitchenEnv, generate_layout
from marl_course.evaluation.coop import make_team_obs


def test_coop_greedy_delivers_on_open_layout():
    """Greedy baseline should achieve at least one delivery on easy layout."""
    env = CoopKitchenEnv()
    obs, _ = env.reset(seed=0)
    policy = CoopGreedyTeamPolicy()
    for _ in range(120):
        team_obs = make_team_obs(obs)
        actions = policy.act(team_obs, team_obs["action_mask"])
        result = env.step({f"agent_{idx}": actions[idx] for idx in range(4)})
        obs = result.observations
    assert env.delivered_soups >= 1


def test_zero_shot_layout_padded_observation():
    """Generated layout observations should respect fixed padded dimensions."""
    layout = generate_layout(10, family="heldout_island")
    env = CoopKitchenEnv(layout=layout)
    obs, _ = env.reset(seed=10)
    assert len(obs["agent_0"]["valid_cell_mask"]) == env.config.max_height
    assert len(obs["agent_0"]["valid_cell_mask"][0]) == env.config.max_width
    assert len(obs["agent_0"]["action_mask"]) == 6
