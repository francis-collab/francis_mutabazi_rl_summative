# Francis Mutabazi – RL Summative  
**RwandaCropGuard: mission-based RL for early crop-disease action & food security**

**Student:** Francis Mutabazi  
**Mission:** Improve health and food security in Rwanda and similar low-resource African settings by 2035 through earlier detection and management of crop disease, using ML tools co-developed with local communities.

---

## 1. Overview

`RwandaCropGuard-v0` is a custom Gymnasium environment: an agricultural extension / digital surveillance worker manages a smallholder farm (typical mixed plots) over one growing season (~60 decision days).

Four algorithms are trained on the **same** environment using **TorchRL** (`GymWrapper`, `TensorDictReplayBuffer`):

| Algorithm | Family |
|-----------|--------|
| DQN | Value-based |
| REINFORCE | Policy gradient |
| A2C | Actor–critic |
| PPO | Actor–critic (clipped) |

**40 experiments** (10 hyper-parameter combinations × 4 algorithms).  
Default budget: **40 000 environment steps per experiment** (~650–700 episodes of length 60) — full learning runs, not smoke tests.

---

## 2. Repository layout

```
francis_mutabazi_rl_summative/
├── pyproject.toml
├── README.md
├── main.py
├── evaluate.py
├── run_experiments.py
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── networks.py
│   ├── dqn_training.py      # TorchRL DQN
│   └── pg_training.py       # TorchRL-wrapped REINFORCE / A2C / PPO
├── models/dqn/  models/pg/
├── logs/
├── assets/
├── notebooks/rwanda_cropguard_experiments.ipynb
└── tests/
```

---

## 3. Run with `uv` only (marker path)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd francis_mutabazi_rl_summative
uv sync

# Environment demo + preview image
uv run main.py --mode demo

# Single experiment examples
uv run main.py --mode train_dqn --total-steps 40000 --device cpu
uv run main.py --mode train_pg --algorithm ppo --total-steps 40000

# Full study (40 runs) – use cuda when available
uv run run_experiments.py --device cuda --total-steps 40000
uv run run_experiments.py --device cpu --only dqn --total-steps 40000

# Greedy evaluation
uv run evaluate.py --model models/dqn/best_model/dqn_model.pth --episodes 10
```

---

## 4. Kaggle (GPU) notebook

If local CPU is too slow, open `notebooks/rwanda_cropguard_experiments.ipynb` on Kaggle:

1. GPU accelerator + Internet on  
2. Upload / clone this repo  
3. Run all cells → 40 experiments, plots, greedy eval, zip download  

---

## 5. Environment design (mission match)

| Element | Design |
|---------|--------|
| **Agent** | Community agricultural extension / surveillance worker |
| **Actions (8)** | Inspect, Apply organic, Apply chemical, Irrigate, Move next, Alert community, Wait/monitor, Harvest |
| **Observations** | Plot disease severity, soil moisture, weather index, input stocks, day index (phone camera, probe, radio bulletin, resource log) |
| **Reward** | Early detection +, effective treatment +, final yield +; disease spread −, wasted inputs − |
| **Horizon** | One season ≈ 60 steps |

---

## 6. Video checklist

1. Entire screen + camera on  
2. State problem, agent behaviour, reward, objective  
3. Show `uv run` navigation and best-agent evaluation / live episode  
4. ~¾ of the video = agent acting in the environment  

---

**Course:** Machine Learning Techniques II – Summative  
**Author:** Francis Mutabazi
