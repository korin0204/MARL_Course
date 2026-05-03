"""PyTorch model definitions and inference wrappers for Bomber policies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PLANE_KEYS = [
    "wall",
    "wood",
    "bomb_timer",
    "bomb_blast",
    "flame",
    "bomb_up",
    "fire_up",
    "danger",
    "agent_0",
    "agent_1",
    "agent_2",
    "agent_3",
]


def infer_obs_dim(example_obs: dict[str, Any]) -> int:
    """Infer flattened observation dimensionality from one sample."""
    return bomber_obs_to_tensor(example_obs, device="cpu").numel()


def infer_obs_shapes(example_obs: dict[str, Any]) -> tuple[int, tuple[int, int, int], int]:
    """Infer flattened dim, grid shape, and non-spatial feature dim.

    The flattened vector is laid out as:
    `[grid planes flattened, agent stats/self id features]`.
    CNN models split this vector back into spatial and non-spatial parts.
    """

    grid = example_obs["grid"]
    channels = len(PLANE_KEYS)
    height = len(grid["wall"])
    width = len(grid["wall"][0])
    obs_dim = infer_obs_dim(example_obs)
    grid_dim = channels * height * width
    return obs_dim, (channels, height, width), obs_dim - grid_dim


def bomber_obs_to_tensor(obs: dict[str, Any], device: str = "cpu") -> torch.Tensor:
    """Flatten Bomber observation into a stable dense feature vector."""

    features: list[float] = []
    grid = obs["grid"]
    for key in PLANE_KEYS:
        plane = grid[key]
        for row in plane:
            features.extend(float(value) for value in row)
    stats = obs["stats"]
    h = len(grid["wall"])
    w = len(grid["wall"][0])
    for idx in range(4):
        agent = stats[f"agent_{idx}"]
        pos_r, pos_c = agent["position"]
        features.extend(
            [
                1.0 if agent["alive"] else 0.0,
                float(agent["ammo"]) / 6.0,
                float(agent["blast"]) / 8.0,
                float(pos_r) / max(1, h - 1),
                float(pos_c) / max(1, w - 1),
            ]
        )
    self_id = int(obs.get("self_id", 0))
    features.extend(1.0 if idx == self_id else 0.0 for idx in range(4))
    return torch.tensor(features, dtype=torch.float32, device=device)


class BomberFeatureEncoder(nn.Module):
    """Encode Bomber observations with either MLP-only or CNN+MLP.

    `use_cnn=False`: the full flattened vector goes through an MLP.
    `use_cnn=True`: board planes go through CNN, stats/self-id go through MLP,
    then both embeddings are concatenated. This matches the observation
    semantics: board channels are spatial, while ammo/blast/alive/self-id are
    compact non-spatial features.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        use_cnn: bool = False,
        grid_shape: tuple[int, int, int] | None = None,
        stats_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_cnn = use_cnn
        self.grid_shape = grid_shape
        self.stats_dim = stats_dim
        if not use_cnn:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            return

        if grid_shape is None or stats_dim is None:
            raise ValueError("CNN Bomber encoder requires grid_shape and stats_dim.")
        channels, height, width = grid_shape
        if channels * height * width + stats_dim != obs_dim:
            raise ValueError("grid_shape/stats_dim do not match obs_dim.")
        self.grid_dim = channels * height * width
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width, hidden_dim),
            nn.ReLU(),
        )
        stats_hidden = max(32, hidden_dim // 2)
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_dim, stats_hidden),
            nn.ReLU(),
            nn.Linear(stats_hidden, stats_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + stats_hidden, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        if not self.use_cnn:
            return self.encoder(obs_vec)
        channels, height, width = self.grid_shape or (0, 0, 0)
        grid_flat = obs_vec[:, : self.grid_dim]
        stats = obs_vec[:, self.grid_dim :]
        grid = grid_flat.reshape(obs_vec.shape[0], channels, height, width)
        return self.fusion(torch.cat([self.cnn(grid), self.stats_encoder(stats)], dim=1))


class DQNNet(nn.Module):
    """Q-network used by DQN training."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 6,
        hidden_dim: int = 256,
        use_cnn: bool = False,
        grid_shape: tuple[int, int, int] | None = None,
        stats_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = BomberFeatureEncoder(obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim)
        self.q_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(obs_vec))


class ActorCriticNet(nn.Module):
    """Actor-critic network with optional CNN observation encoder."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 6,
        hidden_dim: int = 256,
        use_cnn: bool = False,
        grid_shape: tuple[int, int, int] | None = None,
        stats_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = BomberFeatureEncoder(obs_dim, hidden_dim=hidden_dim, use_cnn=use_cnn, grid_shape=grid_shape, stats_dim=stats_dim)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs_vec)
        return self.actor_head(hidden), self.critic_head(hidden).squeeze(-1)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Mask illegal actions by sending logits to a very negative value."""

    return logits.masked_fill(action_mask <= 0, -1.0e9)


@dataclass
class DQNTorchPolicy:
    """Inference-time wrapper around a trained DQN model."""

    model: DQNNet
    device: str = "cpu"

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        mask = action_mask or obs["action_mask"]
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            obs_vec = bomber_obs_to_tensor(obs, device=self.device).unsqueeze(0)
            q_values = self.model(obs_vec).squeeze(0)
            q_values = q_values.masked_fill(mask_tensor <= 0, -1.0e9)
            if deterministic:
                return int(torch.argmax(q_values).item())
            probs = torch.softmax(q_values, dim=-1)
            return int(torch.multinomial(probs, 1).item())


@dataclass
class ActorCriticTorchPolicy:
    """Inference-time wrapper around a trained actor-critic model."""

    model: ActorCriticNet
    device: str = "cpu"

    def act(self, obs: dict[str, Any], action_mask: list[int] | None = None, deterministic: bool = True) -> int:
        mask = action_mask or obs["action_mask"]
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            obs_vec = bomber_obs_to_tensor(obs, device=self.device).unsqueeze(0)
            logits, _value = self.model(obs_vec)
            logits = masked_logits(logits.squeeze(0), mask_tensor)
            if deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())


def load_dqn_policy(model_path: str | Path, device: str = "cpu") -> DQNTorchPolicy:
    """Load DQN checkpoint and return ready-to-act policy object."""
    payload = torch.load(str(model_path), map_location=device)
    model = DQNNet(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload.get("n_actions", 6)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        use_cnn=bool(payload.get("use_cnn", False)),
        grid_shape=tuple(payload["grid_shape"]) if payload.get("grid_shape") else None,
        stats_dim=int(payload["stats_dim"]) if payload.get("stats_dim") is not None else None,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return DQNTorchPolicy(model=model, device=device)


def load_actor_critic_policy(model_path: str | Path, device: str = "cpu") -> ActorCriticTorchPolicy:
    """Load actor-critic checkpoint and return ready-to-act policy object."""
    payload = torch.load(str(model_path), map_location=device)
    model = ActorCriticNet(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload.get("n_actions", 6)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        use_cnn=bool(payload.get("use_cnn", False)),
        grid_shape=tuple(payload["grid_shape"]) if payload.get("grid_shape") else None,
        stats_dim=int(payload["stats_dim"]) if payload.get("stats_dim") is not None else None,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return ActorCriticTorchPolicy(model=model, device=device)
