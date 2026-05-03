# Tests for Bomber Arena environment contracts.
from marl_course.envs.bomber_arena import BomberArenaEnv, BomberRuleBasedPolicy
from marl_course.algos.bomber_torch import ActorCriticNet, DQNNet, bomber_obs_to_tensor, infer_obs_shapes


def test_bomber_rule_based_runs_to_end():
    """Rule-based match should progress and terminate/truncate cleanly."""
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=1)
    policies = [BomberRuleBasedPolicy(seed=idx) for idx in range(4)]
    for _ in range(400):
        actions = {
            f"agent_{idx}": policies[idx].act(obs[f"agent_{idx}"], obs[f"agent_{idx}"]["action_mask"])
            for idx in range(4)
        }
        result = env.step(actions)
        obs = result.observations
        if any(result.terminations.values()) or any(result.truncations.values()):
            break
    assert env.step_count > 0
    assert env.last_winner is not None or env.step_count >= env.config.max_steps


def test_bomber_observation_shape_contract():
    """Observation keys/dimensions should stay stable for student code."""
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=2)
    agent_obs = obs["agent_0"]
    assert len(agent_obs["action_mask"]) == 6
    assert "danger" in agent_obs["grid"]
    assert len(agent_obs["alive_mask"]) == 4


def test_bomber_enemy_eliminated_event_has_killer_and_target():
    """Kill event should identify who earned the elimination reward."""
    env = BomberArenaEnv()
    env.reset(seed=3)
    victim_pos = env.positions["agent_1"]
    env.flames[victim_pos] = env.config.flame_life
    env.flame_owners[victim_pos] = "agent_0"

    env._kill_agents_in_flames()

    kill_events = [event for event in env.events if event.name == "enemy_eliminated"]
    assert kill_events
    assert kill_events[0].actor == "agent_0"
    assert kill_events[0].target == "agent_1"


def test_bomber_cnn_models_accept_flat_observation():
    """CNN mode should split flat observation into grid and stats internally."""
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=4)
    obs_dim, grid_shape, stats_dim = infer_obs_shapes(obs["agent_0"])
    obs_vec = bomber_obs_to_tensor(obs["agent_0"]).unsqueeze(0)

    dqn = DQNNet(obs_dim=obs_dim, hidden_dim=64, use_cnn=True, grid_shape=grid_shape, stats_dim=stats_dim)
    ac = ActorCriticNet(obs_dim=obs_dim, hidden_dim=64, use_cnn=True, grid_shape=grid_shape, stats_dim=stats_dim)

    assert dqn(obs_vec).shape == (1, 6)
    logits, value = ac(obs_vec)
    assert logits.shape == (1, 6)
    assert value.shape == (1,)


def test_bomber_observation_danger_has_no_side_effects():
    """danger plane生成は木箱やeventsを変更してはいけない。"""
    env = BomberArenaEnv()
    obs, _ = env.reset(seed=5)
    env.step({"agent_0": 5, "agent_1": 0, "agent_2": 0, "agent_3": 0})
    board_before = [row[:] for row in env.board]
    powerups_before = dict(env.powerups)
    events_before = list(env.events)

    _ = env._observations()

    assert env.board == board_before
    assert env.powerups == powerups_before
    assert env.events == events_before


def test_bomber_block_destroyed_event_has_owner():
    """木箱破壊報酬を学生本人にだけ付けられるよう、actorを記録する。"""
    env = BomberArenaEnv()
    env.reset(seed=6)
    env.board[1][2] = env.WOOD

    env._blast_cells((1, 1), blast=2, apply_effects=True, owner="agent_0")

    block_events = [event for event in env.events if event.name == "block_destroyed"]
    assert block_events
    assert block_events[0].actor == "agent_0"
