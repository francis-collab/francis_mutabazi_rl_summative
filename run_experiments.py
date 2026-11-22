"""
Run the full hyper-parameter study: 10 configs × 4 algorithms = 40 experiments.

All algorithms use the same RwandaCropGuardEnv (including multi-plot visitation
shaping: stronger MOVE when other plots are sick, higher untreated-disease cost).

Trains directly against the Gymnasium-compatible RwandaCropGuardEnv.

uv run run_experiments.py --device cpu --total-steps 80000  # Runs all 40 experiments on CPU 

Optional: only one family
  uv run run_experiments.py --device cpu --total-steps 80000 --only dqn
  uv run run_experiments.py --device cpu --total-steps 80000 --only ppo

Best model per family is selected by greedy mean return.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from evaluate import load_and_evaluate
from training.dqn_training import dqn_hyperparameter_grid, train_dqn
from training.pg_training import pg_hyperparameter_grid, train_pg


def select_best(family_dir: Path, model_filename: str) -> None:
    """Copy the experiment with highest greedy_mean into family_dir/best_model/."""
    best_score = -1e9
    best_dir = None
    for exp_dir in sorted(family_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "best_model":
            continue
        greedy_path = exp_dir / "greedy_eval.json"
        results_path = exp_dir / "results.json"
        # Ignore legacy four-plot checkpoints (18 inputs); the current six-plot
        # environment has 26 inputs. Never select an incompatible checkpoint.
        candidate_model = exp_dir / model_filename
        if candidate_model.exists():
            try:
                import torch
                ckpt = torch.load(candidate_model, map_location="cpu", weights_only=False)
                if int(ckpt.get("obs_dim", -1)) != 26:
                    continue
            except Exception:
                continue
        else:
            continue
        score = None
        if greedy_path.exists():
            score = json.loads(greedy_path.read_text())["mean_return"]
        elif results_path.exists():
            score = json.loads(results_path.read_text()).get("mean_return_last_20", None)
        if score is not None and score > best_score:
            best_score = score
            best_dir = exp_dir
    if best_dir is None:
        print(f"No results under {family_dir}")
        return
    dest = family_dir / "best_model"
    dest.mkdir(parents=True, exist_ok=True)
    src_model = best_dir / model_filename
    if src_model.exists():
        shutil.copy2(src_model, dest / model_filename)
    for extra in ("results.json", "greedy_eval.json", "training_plot.png"):
        p = best_dir / extra
        if p.exists():
            shutil.copy2(p, dest / extra)
    meta = {"source_experiment": best_dir.name, "greedy_mean": best_score}
    (dest / "selection.json").write_text(json.dumps(meta, indent=2))
    print(f"Best for {family_dir.name}: {best_dir.name} (score={best_score:.3f}) → {dest}")


def run_dqn(device: str, total_steps: int):
    print("\n" + "=" * 60 + "\nDQN – 10 experiments\n" + "=" * 60)
    summary = []
    for cfg in dqn_hyperparameter_grid(device, total_steps):
        print(f"\n>>> {cfg.exp_name} lr={cfg.lr} gamma={cfg.gamma} batch={cfg.batch_size}")
        res = train_dqn(cfg)
        model_path = Path(cfg.save_dir) / cfg.exp_name / "dqn_model.pth"
        greedy = load_and_evaluate(str(model_path), n_episodes=10)
        res["greedy_mean"] = greedy["mean_return"]
        with open(model_path.parent / "results.json", "w") as f:
            json.dump(res, f, indent=2)
        with open(model_path.parent / "greedy_eval.json", "w") as f:
            json.dump(greedy, f, indent=2)
        summary.append(
            {
                "exp": cfg.exp_name,
                "lr": cfg.lr,
                "gamma": cfg.gamma,
                "batch_size": cfg.batch_size,
                "mean_last20": res["mean_return_last_20"],
                "greedy_mean": greedy["mean_return"],
                "num_episodes": res["num_episodes"],
            }
        )
        print(f"    episodes={res['num_episodes']} last20={res['mean_return_last_20']:.2f} greedy={greedy['mean_return']:.2f}")
    select_best(Path("models/dqn"), "dqn_model.pth")
    return summary


def run_pg(algorithm: str, device: str, total_steps: int):
    print("\n" + "=" * 60 + f"\n{algorithm.upper()} – 10 experiments\n" + "=" * 60)
    summary = []
    for cfg in pg_hyperparameter_grid(algorithm, device, total_steps):
        print(f"\n>>> {cfg.exp_name} lr={cfg.lr} gamma={cfg.gamma} entropy={cfg.entropy_coef}")
        res = train_pg(cfg)
        model_path = Path(cfg.save_dir) / cfg.exp_name / f"{algorithm}_model.pth"
        greedy = load_and_evaluate(str(model_path), n_episodes=10)
        res["greedy_mean"] = greedy["mean_return"]
        with open(model_path.parent / "results.json", "w") as f:
            json.dump(res, f, indent=2)
        with open(model_path.parent / "greedy_eval.json", "w") as f:
            json.dump(greedy, f, indent=2)
        summary.append(
            {
                "exp": cfg.exp_name,
                "lr": cfg.lr,
                "gamma": cfg.gamma,
                "entropy_coef": cfg.entropy_coef,
                "mean_last20": res["mean_return_last_20"],
                "greedy_mean": greedy["mean_return"],
                "num_episodes": res["num_episodes"],
            }
        )
        print(f"    episodes={res['num_episodes']} last20={res['mean_return_last_20']:.2f} greedy={greedy['mean_return']:.2f}")
    # Per-algorithm best (kept for analysis)
    select_best(Path("models/pg"), f"{algorithm}_model.pth")
    algo_best = Path("models/pg") / f"{algorithm}_best"
    src = Path("models/pg") / "best_model"
    if src.exists():
        if algo_best.exists():
            shutil.rmtree(algo_best)
        shutil.copytree(src, algo_best)
    return summary


def select_overall_pg_best() -> None:
    """Pick the single best checkpoint across REINFORCE / A2C / PPO into models/pg/best_model.

    Ranking metric: greedy mean return (same as formative).
    """
    family = Path("models/pg")
    best_score = -1e9
    best_dir = None
    best_algo = None
    best_model_name = None
    for exp_dir in sorted(family.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if name in ("best_model",) or name.endswith("_best"):
            continue
        # Identify algorithm from folder / files
        model_file = None
        algo = None
        for candidate, alg in (
            ("ppo_model.pth", "ppo"),
            ("a2c_model.pth", "a2c"),
            ("reinforce_model.pth", "reinforce"),
        ):
            if (exp_dir / candidate).exists():
                model_file = candidate
                algo = alg
                break
        if model_file is None:
            continue
        try:
            import torch
            ckpt = torch.load(exp_dir / model_file, map_location="cpu", weights_only=False)
            if int(ckpt.get("obs_dim", -1)) != 26:
                continue
        except Exception:
            continue
        greedy_path = exp_dir / "greedy_eval.json"
        results_path = exp_dir / "results.json"
        score = None
        if greedy_path.exists():
            score = json.loads(greedy_path.read_text()).get("mean_return")
        elif results_path.exists():
            score = json.loads(results_path.read_text()).get("mean_return_last_20")
        if score is not None and score > best_score:
            best_score = score
            best_dir = exp_dir
            best_algo = algo
            best_model_name = model_file
    if best_dir is None:
        print("No PG results found for overall best selection")
        return
    dest = family / "best_model"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # Copy the winning model under its original name AND a generic name for convenience
    shutil.copy2(best_dir / best_model_name, dest / best_model_name)
    # Also keep any sibling models that were already in per-algo best folders
    for extra in ("results.json", "greedy_eval.json", "training_plot.png"):
        p = best_dir / extra
        if p.exists():
            shutil.copy2(p, dest / extra)
    meta = {
        "source_experiment": best_dir.name,
        "selected_algorithm": best_algo.upper() if best_algo else None,
        "greedy_mean": best_score,
        "note": "Shared overall best across REINFORCE / A2C / PPO (ranked by greedy mean return).",
    }
    (dest / "selection.json").write_text(json.dumps(meta, indent=2))
    print(f"Overall PG best_model ← {best_dir.name} ({best_algo}) greedy={best_score:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--only", default="all", choices=["all", "dqn", "reinforce", "a2c", "ppo"])
    parser.add_argument("--total-steps", type=int, default=80_000,
                        help="Environment steps per experiment (default 80000 ≈ 880 episodes of the 90-day season)")
    args = parser.parse_args()

    Path("models/dqn").mkdir(parents=True, exist_ok=True)
    Path("models/pg").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    all_summaries = {}
    if args.only in ("all", "dqn"):
        all_summaries["dqn"] = run_dqn(args.device, args.total_steps)
    ran_pg = False
    for algo in ("reinforce", "a2c", "ppo"):
        if args.only in ("all", algo):
            all_summaries[algo] = run_pg(algo, args.device, args.total_steps)
            ran_pg = True

    # Shared overall best across the whole PG family (user requirement)
    if ran_pg or args.only == "all":
        select_overall_pg_best()

    with open("logs/full_experiment_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nALL REQUESTED EXPERIMENTS FINISHED")
    print("DQN best → models/dqn/best_model/")
    print("PG overall best (shared) → models/pg/best_model/")
    print("Summary → logs/full_experiment_summary.json")


if __name__ == "__main__":
    main()
