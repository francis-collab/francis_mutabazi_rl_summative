# RwandaCropGuard — Mission-Based Reinforcement Learning

**Author:** Francis Mutabazi  
**Course:** Machine Learning Techniques II — Summative  
**Stack:** Python 3.11+, [uv](https://github.com/astral-sh/uv), Gymnasium, PyTorch, TorchRL, Three.js  

**Mission:** An agricultural extension agent protects food security on a six-plot Rwandan mixed-crop farm through early disease detection, treatment, irrigation, community response, and timely harvest.

---

## Table of contents

1. [Assignment alignment](#1-assignment-alignment)
2. [Repository layout](#2-repository-layout)
3. [Quick start (uv only)](#3-quick-start-uv-only)
4. [Mission environment](#4-mission-environment)
5. [Reinforcement learning algorithms](#5-reinforcement-learning-algorithms)
6. [Training & hyperparameter study](#6-training--hyperparameter-study)
7. [Evaluation & live play](#7-evaluation--live-play)
8. [Visualization (2D & 3D)](#8-visualization-2d--3d)
9. [Video demonstration](#9-video-demonstration)
10. [Tests](#10-tests)
11. [Design notes](#11-design-notes)

---

## 1. Assignment alignment

| Requirement | How this project meets it |
|-------------|---------------------------|
| Custom non-generic environment | `RwandaCropGuardEnv` — six mixed plots, seasonal dynamics, limited inputs |
| Exhaustive, real-world actions | Inspect, organic/chemical treatment, irrigate, move, alert community, wait, harvest |
| Observation space | 26-dim continuous vector (plot one-hot, disease, moisture, stocks, weather, risk, …) |
| Rewards tied to states/actions | Treatment efficacy, irrigation under drought/sun, harvest yield, penalties for waste and neglect |
| Start state & terminals | Reset with stocks and mild baseline disease; harvest ends episode; season length truncates |
| Advanced visualization | Matplotlib 2D farm view + **Three.js** interactive 3D village farm |
| Four algorithms, same env | **DQN**, **REINFORCE**, **A2C**, **PPO** via TorchRL wrappers |
| ≥10 hyperparameter runs each | `run_experiments.py` — 40 full runs (10 × 4) |
| Runnable with **uv** only | `uv sync` then `uv run …` — no manual venv |

Horizon: **90 decision days** (one compressed season). Harvest opens at day **78**.

---

## 2. Repository layout

```
francis_mutabazi_rl_summative/
├── pyproject.toml          # Dependencies (mandatory for uv)
├── uv.lock                 # Locked versions
├── README.md
├── main.py                 # CLI: demo | train_dqn | train_pg | evaluate | play
├── evaluate.py             # Greedy eval + live play + trajectory export
├── run_experiments.py      # Full 40-run hyperparameter study
│
├── environment/
│   ├── custom_env.py       # RwandaCropGuardEnv (Gymnasium)
│   └── rendering.py        # 2D matplotlib farm frames
│
├── training/
│   ├── networks.py         # QNet, PolicyNet, ActorCriticNet
│   ├── dqn_training.py     # TorchRL DQN
│   └── pg_training.py      # REINFORCE / A2C / PPO
│
├── models/
│   ├── dqn/                # Per-experiment + best checkpoints
│   └── pg/                 # reinforce_*, a2c_*, ppo_*
│
├── visualization/
│   └── viewer_3d.html      # Pure Three.js interactive farm
│
├── assets/
│   ├── trajectories/       # latest_trajectory.json for the 3D viewer
│   
│
├── logs/                   # full_experiment_summary.json
├── tests/                  # Environment regression tests
```

---

## 3. Quick start (uv only)

Others must be able to clone and run with **only** uv:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

cd francis_mutabazi_rl_summative
uv sync

# Random-policy demo 
uv run main.py --mode demo

# Live agent + terminal verbose + trajectory for 3D viewer
uv run main.py --mode play \
  --model models/dqn/best_model/dqn_model.pth \
  --episodes 1 --viz 3d --delay 0.3

# Serve the repo root, then open the 3D viewer
uv run python -m http.server 8000
# Browser: http://localhost:8000/visualization/viewer_3d.html
```

Greedy evaluation (numbers only):

```bash
uv run evaluate.py --model models/dqn/best_model/dqn_model.pth --episodes 10
# or
uv run main.py --mode evaluate --model models/dqn/best_model/dqn_model.pth --episodes 10
```

---

## 4. Mission environment

### Plots (fixed layout)

| Plot | Crop   |
|------|--------|
| 0    | Beans  |
| 1    | Maize  |
| 2    | Cassava|
| 3    | Beans  |
| 4    | Maize  |
| 5    | Cassava|

### Actions (8)

| ID | Name | Role |
|----|------|------|
| 0 | `INSPECT` | Reveal / confirm state of the current plot (once usefully) |
| 1 | `APPLY_ORGANIC` | Organic disease treatment on current plot |
| 2 | `APPLY_CHEMICAL` | Synthetic treatment (scarcer stock) |
| 3 | `IRRIGATE` | Farm-wide irrigation from the shared water tank |
| 4 | `MOVE_NEXT` | Walk the path network to the next plot |
| 5 | `ALERT_COMMUNITY` | One-time call for neighbour help (near harvest / high risk) |
| 6 | `WAIT_MONITOR` | Advance the day without intervention |
| 7 | `HARVEST` | End-of-season harvest (terminal; opens day 78) |

### Observation (26 continuous values)

- Current-plot one-hot (6)
- Disease severity per plot (6)
- Soil moisture per plot (6)
- Weather index, chemical / organic / labour stocks (normalized)
- Season progress, cumulative yield risk
- Water reserve & related globals as implemented in `_get_obs()`

### Dynamics (design intent)

- **Disease** can appear and progress over the season; treatment reduces severity and marks plots as cared for.
- **Moisture** falls with evaporation (worse under hot/sunny weather) and rises with irrigation.
- **Weather** varies; mid-season stretches can be forced sunnier to stress water demand.
- **Limited stocks**: chemical, organic, labour, water — wasteful actions are penalized.
- **Terminals:** successful `HARVEST` after maturity, or season timeout without harvest (large penalty).

### Reward themes

- Positive: effective early treatment, useful inspection, irrigation when the farm is dry (especially under sun), justified community alert, high harvest yield.
- Negative: re-inspect spam, treating already-clean plots, leaving disease untreated, abandoning a sick plot without treating, premature harvest, missing the deadline.

Exact coefficients live in `environment/custom_env.py` and may be tuned for pedagogy; retrain after any reward change.

---

## 5. Reinforcement learning algorithms

All four algorithms interact with the **same** `RwandaCropGuardEnv` for a fair comparison.

| Family | Algorithm | Implementation |
|--------|-----------|----------------|
| Value-based | **DQN** | `training/dqn_training.py` (TorchRL) |
| Policy gradient | **REINFORCE** | `training/pg_training.py` |
| Actor–critic | **A2C** | `training/pg_training.py` |
| Actor–critic | **PPO** | `training/pg_training.py` |

Shared networks: `training/networks.py` (`QNet`, `PolicyNet`, `ActorCriticNet`).

Single-run examples:

```bash
uv run main.py --mode train_dqn --total-steps 80000 --device cpu
uv run main.py --mode train_pg --algorithm ppo --total-steps 80000 --device cpu
uv run main.py --mode train_pg --algorithm a2c --total-steps 80000 --device cpu
uv run main.py --mode train_pg --algorithm reinforce --total-steps 80000 --device cpu
```

---

## 6. Training & hyperparameter study

Default budget per experiment: **80 000** environment steps (full learning runs, not smoke tests).

```bash
# Full study: 10 hyperparameter combinations × 4 algorithms = 40 runs
uv run run_experiments.py --device cuda --total-steps 80000
#Or
uv run run_experiments.py --device cpu --total-steps 80000

# Subsets
uv run run_experiments.py --device cpu --only dqn --total-steps 80000
uv run run_experiments.py --device cpu --only ppo --total-steps 80000
uv run run_experiments.py --device cpu --only reinforce --total-steps 80000
uv run run_experiments.py --device cpu --only a2c --total-steps 80000
```

Artifacts per run (under `models/dqn/` or `models/pg/<algo>_expXX/`):

- `*_model.pth` — checkpoint  
- `results.json` — training curve stats  
- `greedy_eval.json` — deterministic eval  
- `training_plot.png`  

Best checkpoints are summarized under `models/*/best_model/` and `logs/full_experiment_summary.json`.

> **Note:** If the environment observation size or reward rules change, old checkpoints are not comparable. Rerun the study before reporting final numbers.

---

## 7. Evaluation & live play

| Mode | Command | What you see |
|------|---------|----------------|
| Stats only | `uv run evaluate.py --model <path> --episodes 10` | Mean/std return JSON |
| Live + verbose | `uv run main.py --mode play --model <path> --viz 2d` | Terminal ANSI + matplotlib GUI |
| Live + 3D trajectory | `uv run main.py --mode play --model <path> --viz 3d` | Terminal + `assets/trajectories/latest_trajectory.json` |

`--delay` controls pause between steps (e.g. `1.2` ≈ two minutes for a 90-day episode at default pace).

Play prints each step’s action, reward, and farm state so you can follow the agent’s decisions without guessing.

---

## 8. Visualization (2D & 3D)

### 2D (matplotlib)

Built into `play` with `--viz 2d` and `environment/rendering.py`. Good for quick local checks and report figures.

### 3D (Three.js — advanced simulation library)

`visualization/viewer_3d.html` is a **pure Three.js** scene:

- Six raised beds with **one crop type each** (beans / maize / cassava)
- Dirt path grid; agent follows paths
- Disease → yellow/brown tint & wilt; growth → larger plants  
- **Blue** spray = irrigation (all plots) · **Cyan** = chemical · **Lime** = organic  
- Sunny mid-season lighting when weather is hot  
- Community alert: neighbours walk from houses toward the farm  
- HUD: day, action, reward, disease/moisture/growth, stocks  

Workflow:

```bash
uv run main.py --mode play --model models/dqn/best_model/dqn_model.pth --episodes 1 --viz 3d
python -m http.server 8000
# Open http://localhost:8000/visualization/viewer_3d.html → Press Play
```

Do **not** open the HTML via `file://` under WSL; use the local HTTP server so the trajectory JSON can load.

---

## 9. Video demonstration

Link to the Demo: https://youtu.be/3sXW7OI9xLE
---

## 10. Tests

```bash
uv run pytest -q
```

Coverage includes six-plot layout, observation shape, action masking, irrigation affecting all plots, one-time alert, and harvest terminal behaviour. Extend tests when changing dynamics or observation dimension.

---

## 11. Design notes

- **Same environment for all algorithms** — comparison is on learning method and hyperparameters, not on mismatched envs.  
- **Action masking** reduces illegal or wasteful actions during training and play (e.g. harvest before maturity, treat with empty stock).  
- **TorchRL** is used for the RL loops; the environment itself stays plain Gymnasium so demo mode works without heavy imports.  
- **uv** is used: `pyproject.toml` + `uv.lock` so you don't need to hand-install dependencies.  

---

## License & academic integrity

Submitted as coursework for Machine Learning Techniques II. Reuse outside the course only with attribution and in line with institutional rules.

**Contact / author:** Francis Mutabazi  
