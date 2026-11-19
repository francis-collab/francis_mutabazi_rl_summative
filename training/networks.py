"""Neural networks used with TorchRL modules."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNet(nn.Module):
    """Q-network for DQN (outputs one value per discrete action)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class ActorCriticNet(nn.Module):
    """Shared trunk with policy logits and state-value head (A2C / PPO)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

    def dist(self, obs: torch.Tensor) -> Categorical:
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)


class PolicyNet(nn.Module):
    """Stochastic policy only (REINFORCE)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.mlp(obs))
