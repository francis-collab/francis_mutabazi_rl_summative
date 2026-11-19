"""
DQN training with TorchRL.

Uses:
  - torchrl.envs.GymWrapper around RwandaCropGuardEnv
  - TensorDictReplayBuffer
  - TorchRL-style epsilon-greedy collection + DQN TD update
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import trange

from torchrl.envs import GymWrapper
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

from environment.custom_env import RwandaCropGuardEnv
from training.networks import QNet


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 12_000
    target_update_freq: int = 250
    train_start: int = 1_000
    total_steps: int = 40_000
    hidden: int = 128
    seed: int = 42
    device: str = "cpu"
    exp_name: str = "exp01"
    save_dir: str = "models/dqn"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_torchrl_env(seed: int | None = None):
    """Wrap the custom Gymnasium env with TorchRL GymWrapper when compatible.

    TorchRL <0.7 does not support gymnasium>=1.0; in that case we still use
    TorchRL's TensorDictReplayBuffer but step the base Gymnasium env.
    """
    base = RwandaCropGuardEnv(seed=seed)
    try:
        env = GymWrapper(base, categorical_action_encoding=True)
        env._base_gym = base
        return env
    except Exception as exc:
        print(f"[warn] GymWrapper unavailable ({exc}); using base Gymnasium env with TorchRL buffers.")
        base._base_gym = base
        return base


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    t = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def train_dqn(cfg: DQNConfig) -> Dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    wrapped = make_torchrl_env(cfg.seed)
    env = getattr(wrapped, "_base_gym", None) or getattr(wrapped, "_env", None) or wrapped
    if not isinstance(env, RwandaCropGuardEnv):
        # GymWrapper path
        env = RwandaCropGuardEnv(seed=cfg.seed)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    online = QNet(obs_dim, n_actions, cfg.hidden).to(device)
    target = QNet(obs_dim, n_actions, cfg.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=cfg.lr)

    # TorchRL replay buffer (TensorDict storage)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer_size),
        batch_size=cfg.batch_size,
    )

    episode_returns: List[float] = []
    losses: List[float] = []
    ep_ret = 0.0
    obs, _ = env.reset(seed=cfg.seed)

    for step in trange(1, cfg.total_steps + 1, desc=f"DQN {cfg.exp_name}", leave=False):
        eps = epsilon_by_step(step, cfg)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)

        if random.random() < eps:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                action = int(online(obs_t).argmax(dim=-1).item())

        next_obs_np, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        ep_ret += float(reward)

        transition = TensorDict(
            {
                "obs": torch.as_tensor(np.asarray(obs), dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.int64),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_obs": torch.as_tensor(np.asarray(next_obs_np), dtype=torch.float32),
                "done": torch.tensor(done, dtype=torch.bool),
            },
            batch_size=[],
        )
        rb.add(transition)

        obs = next_obs_np
        if done:
            episode_returns.append(ep_ret)
            ep_ret = 0.0
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        if step >= cfg.train_start and len(rb) >= cfg.batch_size:
            batch = rb.sample()
            b_obs = batch["obs"].to(device)
            b_act = batch["action"].to(device)
            b_rew = batch["reward"].to(device)
            b_next = batch["next_obs"].to(device)
            b_done = batch["done"].to(device).float()

            q = online(b_obs).gather(1, b_act.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                next_q = target(b_next).max(dim=1).values
                y = b_rew + cfg.gamma * next_q * (1.0 - b_done)
            loss = F.smooth_l1_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if step % cfg.target_update_freq == 0:
            target.load_state_dict(online.state_dict())

    out_dir = Path(cfg.save_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": online.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden": cfg.hidden,
            "config": asdict(cfg),
        },
        out_dir / "dqn_model.pth",
    )

    results = {
        "algorithm": "dqn",
        "config": asdict(cfg),
        "episode_returns": episode_returns,
        "mean_return_last_20": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
        "mean_return_all": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "best_return": float(np.max(episode_returns)) if episode_returns else 0.0,
        "num_episodes": len(episode_returns),
        "losses": losses[-200:],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_returns(episode_returns, out_dir / "training_plot.png", title=f"DQN {cfg.exp_name}")
    env.close()
    return results


def _plot_returns(returns: List[float], path: Path, title: str) -> None:
    if not returns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(returns, alpha=0.35, label="episode return")
    if len(returns) >= 10:
        kernel = np.ones(10) / 10
        smooth = np.convolve(returns, kernel, mode="valid")
        ax.plot(range(9, len(returns)), smooth, label="10-ep average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def dqn_hyperparameter_grid(device: str, total_steps: int) -> List[DQNConfig]:
    """10 DQN configurations (formative-style axes)."""
    base = dict(device=device, total_steps=total_steps, save_dir="models/dqn")
    specs = [
        ("exp01", dict(lr=1e-3, gamma=0.99, batch_size=64, eps_decay_steps=12_000)),
        ("exp02", dict(lr=3e-4, gamma=0.99, batch_size=64, eps_decay_steps=12_000)),
        ("exp03", dict(lr=1e-4, gamma=0.99, batch_size=64, eps_decay_steps=12_000)),
        ("exp04", dict(lr=1e-3, gamma=0.95, batch_size=64, eps_decay_steps=12_000)),
        ("exp05", dict(lr=1e-3, gamma=0.99, batch_size=128, eps_decay_steps=12_000)),
        ("exp06", dict(lr=1e-3, gamma=0.99, batch_size=32, eps_decay_steps=12_000)),
        ("exp07", dict(lr=1e-3, gamma=0.99, batch_size=64, eps_end=0.01, eps_decay_steps=12_000)),
        ("exp08", dict(lr=1e-3, gamma=0.99, batch_size=64, eps_decay_steps=6_000)),
        ("exp09", dict(lr=1e-3, gamma=0.99, batch_size=64, eps_decay_steps=20_000)),
        ("exp10", dict(lr=3e-4, gamma=0.99, batch_size=128, eps_decay_steps=12_000)),
    ]
    return [DQNConfig(exp_name=name, **base, **kw) for name, kw in specs]
