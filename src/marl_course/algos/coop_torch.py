"""Neural MAPPO and QMIX building blocks for Coop Kitchen.

このファイルはOvercooked系課題の「完全版」学習ループで使う共通実装。
観測は各エージェントの盤面チャネルをCNNで処理し、MAPPOは集中critic、
QMIXはmonotonic mixerでチームQ値を作る。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


COOP_PLANE_KEYS = [
    "wall",
    "floor",
    "onion",
    "dish",
    "pot",
    "delivery",
    "counter",
    "out_of_bounds",
    "onion_item",
    "dish_item",
    "soup_item",
    "pot_onions",
    "pot_ready",
    "pot_cooking",
    "agent_0",
    "agent_1",
    "agent_2",
    "agent_3",
]


def coop_agent_obs_to_tensor(agent_obs: dict[str, list[list[float]]], device: str = "cpu") -> torch.Tensor:
    """1エージェント分の盤面チャネルを `[C,H,W]` tensorへ変換する。"""

    planes = []
    for key in COOP_PLANE_KEYS:
        planes.append(torch.tensor(agent_obs[key], dtype=torch.float32, device=device))
    return torch.stack(planes, dim=0)


def coop_team_obs_to_tensor(team_obs: dict[str, Any], device: str = "cpu") -> torch.Tensor:
    """4エージェント分の観測を `[4,C,H,W]` tensorへ変換する。"""

    return torch.stack([coop_agent_obs_to_tensor(obs, device=device) for obs in team_obs["agent_obs"]], dim=0)


def infer_coop_shapes(team_obs: dict[str, Any]) -> tuple[tuple[int, int, int], int]:
    """観測shapeとflattenしたteam state次元を推定する。"""

    one = coop_agent_obs_to_tensor(team_obs["agent_obs"][0])
    team = coop_team_obs_to_tensor(team_obs)
    return tuple(one.shape), int(team.numel())


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """非合法行動を非常に小さいlogitにしてサンプリング対象から外す。"""

    return logits.masked_fill(masks <= 0, -1.0e9)


class CoopCNNEncoder(nn.Module):
    """各エージェントの局所盤面観測を埋め込みへ変換する共有CNN。"""

    def __init__(self, obs_shape: tuple[int, int, int], hidden_dim: int = 128):
        super().__init__()
        channels, height, width = obs_shape
        self.obs_shape = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MAPPOActorCritic(nn.Module):
    """Parameter-sharing actor + centralized critic for 4-agent MAPPO."""

    def __init__(self, obs_shape: tuple[int, int, int], hidden_dim: int = 128, n_actions: int = 6):
        super().__init__()
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.actor_encoder = CoopCNNEncoder(obs_shape, hidden_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic_encoder = CoopCNNEncoder(obs_shape, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, agent_obs: torch.Tensor) -> torch.Tensor:
        """`[B,4,C,H,W]` or `[4,C,H,W]` から各agent logitsを返す。"""

        squeeze_batch = agent_obs.dim() == 4
        if squeeze_batch:
            agent_obs = agent_obs.unsqueeze(0)
        bsz, n_agents, channels, height, width = agent_obs.shape
        flat = agent_obs.reshape(bsz * n_agents, channels, height, width)
        logits = self.actor(self.actor_encoder(flat)).reshape(bsz, n_agents, self.n_actions)
        return logits.squeeze(0) if squeeze_batch else logits

    def forward_value(self, team_obs: torch.Tensor) -> torch.Tensor:
        """集中criticとして、4人の観測全体からチーム価値を返す。"""

        squeeze_batch = team_obs.dim() == 4
        if squeeze_batch:
            team_obs = team_obs.unsqueeze(0)
        bsz, n_agents, channels, height, width = team_obs.shape
        flat = team_obs.reshape(bsz * n_agents, channels, height, width)
        encoded = self.critic_encoder(flat).reshape(bsz, n_agents * self.hidden_dim)
        values = self.critic(encoded).squeeze(-1)
        return values.squeeze(0) if squeeze_batch else values


class QMIXNet(nn.Module):
    """Shared agent Q-network + monotonic mixing network."""

    def __init__(self, obs_shape: tuple[int, int, int], state_dim: int, hidden_dim: int = 128, mixing_dim: int = 32, n_actions: int = 6):
        super().__init__()
        self.obs_shape = obs_shape
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.mixing_dim = mixing_dim
        self.n_actions = n_actions
        self.agent_encoder = CoopCNNEncoder(obs_shape, hidden_dim)
        self.agent_q = nn.Linear(hidden_dim, n_actions)
        self.hyper_w1 = nn.Linear(state_dim, 4 * mixing_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_dim), nn.ReLU(), nn.Linear(mixing_dim, 1))

    def agent_q_values(self, agent_obs: torch.Tensor) -> torch.Tensor:
        squeeze_batch = agent_obs.dim() == 4
        if squeeze_batch:
            agent_obs = agent_obs.unsqueeze(0)
        bsz, n_agents, channels, height, width = agent_obs.shape
        flat = agent_obs.reshape(bsz * n_agents, channels, height, width)
        q = self.agent_q(self.agent_encoder(flat)).reshape(bsz, n_agents, self.n_actions)
        return q.squeeze(0) if squeeze_batch else q

    def mix(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """各agentの選択Q `[B,4]` とstateからチームQ `[B]` を作る。"""

        if agent_qs.dim() == 1:
            agent_qs = agent_qs.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        bsz = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).reshape(bsz, 4, self.mixing_dim)
        b1 = self.hyper_b1(state).reshape(bsz, 1, self.mixing_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).reshape(bsz, self.mixing_dim, 1)
        b2 = self.hyper_b2(state).reshape(bsz, 1, 1)
        return (torch.bmm(hidden, w2) + b2).reshape(bsz)


@dataclass
class MAPPOPolicy:
    """教師評価で使うMAPPO推論ラッパー。"""

    model: MAPPOActorCritic
    device: str = "cpu"

    def act(self, team_obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        masks = torch.tensor(action_mask or team_obs["action_mask"], dtype=torch.float32, device=self.device)
        obs_tensor = coop_team_obs_to_tensor(team_obs, device=self.device)
        with torch.no_grad():
            logits = masked_logits(self.model.forward_actor(obs_tensor), masks)
            if deterministic:
                return [int(x) for x in torch.argmax(logits, dim=-1).detach().cpu().tolist()]
            probs = torch.softmax(logits, dim=-1)
            return [int(x) for x in torch.distributions.Categorical(probs=probs).sample().detach().cpu().tolist()]


@dataclass
class QMIXPolicy:
    """教師評価で使うQMIX推論ラッパー。"""

    model: QMIXNet
    device: str = "cpu"

    def act(self, team_obs: dict[str, Any], action_mask: list[list[int]] | None = None, deterministic: bool = True) -> list[int]:
        masks = torch.tensor(action_mask or team_obs["action_mask"], dtype=torch.float32, device=self.device)
        obs_tensor = coop_team_obs_to_tensor(team_obs, device=self.device)
        with torch.no_grad():
            q_values = self.model.agent_q_values(obs_tensor)
            q_values = q_values.masked_fill(masks <= 0, -1.0e9)
            if deterministic:
                return [int(x) for x in torch.argmax(q_values, dim=-1).detach().cpu().tolist()]
            probs = torch.softmax(q_values, dim=-1)
            return [int(x) for x in torch.distributions.Categorical(probs=probs).sample().detach().cpu().tolist()]


def load_mappo_policy(model_path: str | Path, device: str = "cpu") -> MAPPOPolicy:
    """MAPPO checkpointを読み、評価可能なpolicyを返す。"""

    payload = torch.load(str(model_path), map_location=device)
    model = MAPPOActorCritic(
        obs_shape=tuple(payload["obs_shape"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
        n_actions=int(payload.get("n_actions", 6)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return MAPPOPolicy(model=model, device=device)


def load_qmix_policy(model_path: str | Path, device: str = "cpu") -> QMIXPolicy:
    """QMIX checkpointを読み、評価可能なpolicyを返す。"""

    payload = torch.load(str(model_path), map_location=device)
    model = QMIXNet(
        obs_shape=tuple(payload["obs_shape"]),
        state_dim=int(payload["state_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
        mixing_dim=int(payload.get("mixing_dim", 32)),
        n_actions=int(payload.get("n_actions", 6)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return QMIXPolicy(model=model, device=device)
