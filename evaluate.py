"""Greedy evaluation + live play of saved models (2D GUI + Three.js 3D trajectory)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from environment.custom_env import RwandaCropGuardEnv
from training.networks import ActorCriticNet, PolicyNet, QNet


def _load_network(model_path: str) -> Tuple[torch.nn.Module, str, int]:
    """Load checkpoint and return (net, kind, n_actions)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    hidden = ckpt.get("hidden", 128)
    algo = ckpt.get("algorithm", "dqn")

    path = Path(model_path)
    if path.name.startswith("dqn") or "dqn" in str(path.parent).lower() or algo == "dqn":
        kind = "dqn"
        net = QNet(obs_dim, n_actions, hidden)
    elif algo == "reinforce" or "reinforce" in path.name.lower():
        kind = "reinforce"
        net = PolicyNet(obs_dim, n_actions, hidden)
    else:
        kind = algo if algo in ("a2c", "ppo") else "ppo"
        net = ActorCriticNet(obs_dim, n_actions, hidden)

    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, kind, n_actions


def _select_action(net, kind: str, obs, deterministic: bool = True) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if kind == "dqn":
            return int(net(obs_t).argmax(dim=-1).item())
        if kind == "reinforce":
            dist = net(obs_t)
            if deterministic:
                return int(dist.probs.argmax(dim=-1).item())
            return int(dist.sample().item())
        logits, _ = net(obs_t)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())


def load_and_evaluate(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict:
    """Silent greedy evaluation → mean / std returns."""
    net, kind, _ = _load_network(model_path)
    env = RwandaCropGuardEnv()
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < env.season_length + 5:
            action = _select_action(net, kind, obs, deterministic=deterministic)
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


def _snapshot(env: RwandaCropGuardEnv, action: int | None, reward: float, total: float) -> dict:
    """Serialise one environment frame for the Three.js viewer."""
    return {
        "day": int(env._step_count),
        "plot": int(env._current_plot),
        "disease": [float(x) for x in env._disease],
        "moisture": [float(x) for x in env._moisture],
        "chemical": int(env._chemical),
        "organic": int(env._organic),
        "labour": int(env._labour),
        "weather": float(env._weather),
        "yield_risk": float(env._yield_risk),
        "action": int(action) if action is not None else None,
        "action_name": env.ACTIONS[action] if action is not None else "RESET",
        "reward": float(reward),
        "total": float(total),
    }


def play_agent(
    model_path: str,
    n_episodes: int = 1,
    deterministic: bool = True,
    delay: float = 0.35,
    max_steps: int = 70,
    viz: str = "2d",
) -> None:
    """
    Live step-by-step rollout.

    viz:
      - none : terminal ANSI only
      - 2d   : live matplotlib 2D farm GUI
      - 3d   : record trajectory + open Three.js 3D viewer (advanced)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

    net, kind, _ = _load_network(model_path)
    env = RwandaCropGuardEnv(render_mode="rgb_array")

    print("=" * 70)
    print(f"PLAY  |  model = {model_path}")
    print(f"       algorithm = {kind}  |  deterministic = {deterministic}")
    print(f"       visualisation = {viz}")
    print("=" * 70)

    trajectory: List[dict] = []
    show_gui = viz in ("2d", "3d")

    if show_gui:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        title = "RwandaCropGuard – 2D Live Farm View" if viz == "2d" else "RwandaCropGuard – recording for 3D viewer"
        fig.canvas.manager.set_window_title(title)

    def draw_2d(env, ax, total_reward: float, last_action_name: str = ""):
        ax.clear()
        ax.set_xlim(-0.3, env.num_plots + 0.8)
        ax.set_ylim(0, 4.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (-0.3, 0), env.num_plots + 1.1, 4.4,
            boxstyle="round,pad=0.02", facecolor="#e8f5e9", edgecolor="none", zorder=0
        ))
        ax.set_title(
            f"Rwanda CropGuard  |  Day {env._step_count}/{env.season_length}  "
            f"|  YieldRisk {env._yield_risk:.2f}  |  Return {total_reward:.1f}\n"
            f"Last action: {last_action_name}",
            fontsize=11, color="#1a5f2a", pad=6, fontweight="bold"
        )
        for i in range(env.num_plots):
            moist = float(env._moisture[i])
            sev = float(env._disease[i])
            soil = (0.35 + 0.3 * moist, 0.55 + 0.25 * moist, 0.18)
            rect = Rectangle((i + 0.12, 0.65), 0.76, 2.0,
                             facecolor=soil, edgecolor="#3e2723", lw=2, zorder=2)
            ax.add_patch(rect)
            if sev > 0.02:
                drect = Rectangle(
                    (i + 0.12, 0.65), 0.76, 2.0 * sev,
                    facecolor=(0.85, 0.12, 0.08, 0.70), edgecolor="none", zorder=3
                )
                ax.add_patch(drect)
            ax.text(i + 0.5, 0.42, f"Plot {i}", ha="center", fontsize=9,
                    fontweight="bold", color="#333")
            ax.text(i + 0.5, 0.18, f"D={sev:.2f}  M={moist:.2f}",
                    ha="center", fontsize=7.5, color="#444")
            if i == env._current_plot:
                circ = Circle((i + 0.5, 3.15), 0.24, facecolor="#ffc107",
                              edgecolor="#e65100", lw=2, zorder=6)
                ax.add_patch(circ)
                ax.text(i + 0.5, 3.55, "AGENT", ha="center", fontsize=8,
                        fontweight="bold", color="#e65100")
        ax.text(
            0.15, 4.05,
            f"  Chem {env._chemical}    Org {env._organic}    Labour {env._labour}  ",
            fontsize=9.5, color="#1b5e20",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#c8e6c9",
                      edgecolor="#2e7d32", lw=1.5)
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        print(f"\n----- Episode {ep + 1}/{n_episodes} -----")
        print(env._render_ansi())
        print("-" * 55)

        total = 0.0
        done = False
        steps = 0
        last_name = "RESET"
        trajectory = [_snapshot(env, None, 0.0, 0.0)]

        if show_gui:
            draw_2d(env, ax, total, last_name)

        while not done and steps < max_steps:
            action = _select_action(net, kind, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
            last_name = env.ACTIONS[action]
            trajectory.append(_snapshot(env, action, reward, total))

            print(
                f"t={steps:02d}  action={last_name:16s}  "
                f"reward={reward:7.2f}  total={total:8.2f}"
            )
            print(env._render_ansi())
            print("-" * 55)

            if show_gui:
                draw_2d(env, ax, total, last_name)

            if delay > 0:
                time.sleep(delay)

            if done:
                print(f"Episode finished | info keys = {list(info.keys())}")
                break

        print(f">>> Episode return: {total:.2f}\n")

    env.close()

    # Save trajectory for Three.js viewer
    out_dir = Path("assets/trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "latest_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump({"frames": trajectory, "model": model_path}, f, indent=2)
    print(f"Trajectory saved → {traj_path}")

    if viz == "3d":
        viewer = Path("visualization/viewer_3d.html").resolve()
        print("\n" + "=" * 70)
        print("3D VIEWER (Three.js – advanced simulation library)")
        print("=" * 70)
        print(f"1. Open this file in a browser:  {viewer}")
        print(f"2. It loads:  {traj_path.resolve()}")
        print("3. Press Play to watch the agent walk the Rwandan plots in 3D.")
        print("=" * 70)
        try:
            import webbrowser
            webbrowser.open(viewer.as_uri())
        except Exception:
            pass

    if show_gui:
        print("\nClose the matplotlib window when you finish recording.")
        plt.ioff()
        plt.show()
    print("Play finished.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate or play a saved RL model")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["eval", "play"],
        default="eval",
        help="eval = silent numeric scores; play = live demo",
    )
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument(
        "--viz",
        choices=["none", "2d", "3d"],
        default="2d",
        help="none | 2d (matplotlib live) | 3d (Three.js advanced viewer)",
    )
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    deterministic = not args.stochastic

    if args.mode == "play":
        play_agent(
            args.model,
            n_episodes=args.episodes,
            deterministic=deterministic,
            delay=args.delay,
            viz=args.viz,
        )
    else:
        res = load_and_evaluate(
            args.model,
            n_episodes=args.episodes,
            deterministic=deterministic,
        )
        print(json.dumps(res, indent=2))
        out = Path(args.model).parent / "greedy_eval.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
