from __future__ import annotations

from marl_course.algos.reward_examples import BomberShapingReward
from marl_course.common.rewards import Event
from marl_course.envs.bomber_arena import BomberArenaEnv
from scripts.train_bomber import _student_reward as qlearning_reward
from scripts.train_bomber_actor_critic import student_reward as actor_critic_reward
from scripts.train_bomber_dqn import student_reward as dqn_reward


def test_bomber_train_rewards_penalize_agent0_when_eliminated_by_enemy() -> None:
    """全Bomber学習ループで、敵に倒された時の即時ペナルティを揃える。"""

    env = BomberArenaEnv()
    env.reset(seed=10)
    events = [Event("agent_eliminated", actor="agent_1", target="agent_0")]

    assert qlearning_reward(env, events) == -0.4
    assert dqn_reward(env, events) == -0.4
    assert actor_critic_reward(env, events) == -0.4


def test_bomber_shaping_reward_penalizes_eliminated_target() -> None:
    """参照報酬クラスでも、倒した側と倒された側の報酬を分けて扱う。"""

    reward_fn = BomberShapingReward(eliminate=0.4, eliminated=-0.4)
    rewards = reward_fn(
        transition=None,
        events=[
            Event("enemy_eliminated", actor="agent_1", target="agent_0"),
            Event("agent_eliminated", actor="agent_1", target="agent_0"),
        ],
        info={"agents": [f"agent_{idx}" for idx in range(4)]},
    )

    assert rewards["agent_1"] == 0.4
    assert rewards["agent_0"] == -0.4
