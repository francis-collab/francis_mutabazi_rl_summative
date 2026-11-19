"""Simple farm visualization for reports and demos (matplotlib / RGB arrays)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:  # pragma: no cover
    plt = None


def render_farm_rgb(env) -> np.ndarray:
    """Return an RGB image (H, W, 3) of the current farm state."""
    if plt is None:
        raise ImportError("matplotlib is required for rendering")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    n = env.num_plots
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"RwandaCropGuard  day={env._step_count}/{env.season_length}  "
        f"plot={env._current_plot}  chem={env._chemical} org={env._organic}",
        fontsize=10,
    )

    for i in range(n):
        sev = float(env._disease[i])
        moist = float(env._moisture[i])
        color = (0.45 + 0.4 * (1 - sev), 0.25 + 0.55 * (1 - sev), 0.15)
        rect = Rectangle((i + 0.08, 0.4), 0.84, 1.2, facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i + 0.5, 1.0, f"P{i}\n{sev:.2f}", ha="center", va="center", fontsize=9, color="white")
        ax.text(i + 0.5, 0.2, f"m={moist:.2f}", ha="center", fontsize=8)
        if i == env._current_plot:
            ax.plot(i + 0.5, 1.85, marker="v", markersize=12, color="gold")

    fig.canvas.draw()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
    except Exception:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
    plt.close(fig)
    return img


def save_episode_frames(env_seed: int = 0, out_path: str = "assets/env_preview.png", steps: int = 12) -> str:
    """Roll a short random episode and save a montage for the report."""
    from environment.custom_env import RwandaCropGuardEnv

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env = RwandaCropGuardEnv(render_mode="rgb_array", seed=env_seed)
    env.reset(seed=env_seed)
    frames = [render_farm_rgb(env)]
    for _ in range(steps):
        env.step(env.action_space.sample())
        frames.append(render_farm_rgb(env))
        if env._step_count >= env.season_length:
            break
    env.close()

    n = len(frames)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, frame in zip(axes, frames):
        ax.imshow(frame)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("RwandaCropGuard – sample trajectory (random policy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
