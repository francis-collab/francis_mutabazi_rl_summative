"""Greedy (deterministic) evaluation of saved models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from environment.custom_env import RwandaCropGuardEnv
from training.networks import ActorCriticNet, PolicyNet, QNet


def load_and_evaluate(model_path: str, n_episodes: int = 10, deterministic: bool = True) -> Dict:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    hidden = ckpt.get("hidden", 128)
    algo = ckpt.get("algorithm", "dqn")

    if "dqn" in Path(model_path).name or algo == "dqn" or "state_dict" in ckpt and "q" not in algo:
        # Heuristic: DQN checkpoints from our saver
        if "algorithm" not in ckpt or ckpt.get("algorithm") == "dqn" or Path(model_path).name.startswith("dqn"):
            net = QNet(obs_dim, n_actions, hidden)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            kind = "dqn"
        else:
            kind = ckpt["algorithm"]
            if kind == "reinforce":
                net = PolicyNet(obs_dim, n_actions, hidden)
            else:
                net = ActorCriticNet(obs_dim, n_actions, hidden)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
    else:
        kind = ckpt.get("algorithm", "ppo")
        if kind == "reinforce":
            net = PolicyNet(obs_dim, n_actions, hidden)
        else:
            net = ActorCriticNet(obs_dim, n_actions, hidden)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

    # Fix kind for dqn files
    if Path(model_path).name.startswith("dqn") or "dqn" in str(Path(model_path).parent):
        kind = "dqn"
        net = QNet(obs_dim, n_actions, hidden)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

    env = RwandaCropGuardEnv()
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < env.season_length + 5:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if kind == "dqn":
                    action = int(net(obs_t).argmax(dim=-1).item())
                elif kind == "reinforce":
                    dist = net(obs_t)
                    action = int(dist.probs.argmax(dim=-1).item()) if deterministic else int(dist.sample().item())
                else:
                    logits, _ = net(obs_t)
                    action = int(logits.argmax(dim=-1).item()) if deterministic else int(
                        torch.distributions.Categorical(logits=logits).sample().item()
                    )
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    env.close()
    return {
        "model": model_path,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    res = load_and_evaluate(args.model, n_episodes=args.episodes)
    print(json.dumps(res, indent=2))
    out = Path(args.model).parent / "greedy_eval.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
