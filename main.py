"""
RwandaCropGuard entry point.

Examples:
  uv run main.py --mode demo
  uv run main.py --mode train_dqn --total-steps 80000 --device cpu
  uv run main.py --mode train_pg --algorithm ppo --total-steps 80000
  uv run main.py --mode evaluate --model models/dqn/best_model/dqn_model.pth
  uv run main.py --mode play --model models/dqn/best_model/dqn_model.pth --episodes 1 --viz 2d
  uv run main.py --mode play --model models/dqn/best_model/dqn_model.pth --episodes 1 --viz 3d
"""

from __future__ import annotations

import argparse

from environment.custom_env import RwandaCropGuardEnv


def run_demo(n_steps: int = 40) -> None:
    """Random-policy walkthrough (does not import TorchRL)."""
    from environment.rendering import save_episode_frames

    env = RwandaCropGuardEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("RwandaCropGuard demo – random policy")
    print("Observation shape:", obs.shape)
    print(env.render())
    total = 0.0
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        name = env.ACTIONS[action]
        print(f"t={t:02d} action={name:16s} reward={reward:7.2f} total={total:8.2f}")
        if terminated or truncated:
            print("Episode ended:", info)
            break
    try:
        path = save_episode_frames(env_seed=0, out_path="assets/env_preview.png")
        print("Saved environment preview →", path)
    except Exception as exc:
        print("Preview save skipped:", exc)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="RwandaCropGuard RL summative")
    parser.add_argument(
        "--mode",
        choices=["demo", "train_dqn", "train_pg", "evaluate", "play"],
        default="demo",
    )
    parser.add_argument("--algorithm", default="ppo", choices=["reinforce", "a2c", "ppo"])
    parser.add_argument("--exp-name", default="exp01")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--total-steps", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Pause between steps in play mode (seconds)",
    )
    parser.add_argument(
        "--viz",
        choices=["none", "2d", "3d"],
        default="2d",
        help="Play visualisation: none | 2d (matplotlib) | 3d (Three.js)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
        return

    if args.mode == "train_dqn":
        from training.dqn_training import DQNConfig, train_dqn

        cfg = DQNConfig(
            lr=args.lr or 1e-3,
            gamma=args.gamma,
            total_steps=args.total_steps,
            exp_name=args.exp_name,
            seed=args.seed,
            device=args.device,
        )
        results = train_dqn(cfg)
        print(
            "DQN finished:",
            {k: results[k] for k in ["mean_return_last_20", "best_return", "num_episodes"]},
        )

    elif args.mode == "train_pg":
        from training.pg_training import PGConfig, train_pg

        cfg = PGConfig(
            algorithm=args.algorithm,
            lr=args.lr or 3e-4,
            gamma=args.gamma,
            total_steps=args.total_steps,
            exp_name=f"{args.algorithm}_{args.exp_name}",
            seed=args.seed,
            device=args.device,
        )
        results = train_pg(cfg)
        print(
            f"{args.algorithm.upper()} finished:",
            {k: results[k] for k in ["mean_return_last_20", "best_return", "num_episodes"]},
        )

    elif args.mode == "evaluate":
        from evaluate import load_and_evaluate

        if not args.model:
            print("Please provide --model path")
            return
        print(load_and_evaluate(args.model, n_episodes=args.episodes))

    elif args.mode == "play":
        from evaluate import play_agent

        if not args.model:
            print("Please provide --model path")
            return
        play_agent(
            args.model,
            n_episodes=args.episodes,
            deterministic=True,
            delay=args.delay,
            viz=args.viz,
        )


if __name__ == "__main__":
    main()
