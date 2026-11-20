"""
RwandaCropGuard-v0
==================
A mission-aligned custom Gymnasium environment for reinforcement learning.

Context (Francis Mutabazi mission):
Improve health and food security of citizens in Rwanda and similar low-resource
settings in Africa by 2035 through earlier detection of diseases in crops,
working with local communities, farmers and health workers.

The agent acts as a community agricultural extension / digital surveillance
worker managing a smallholder farm (typical Rwandan mixed maize/beans plot).
The goal is early disease detection and resource-efficient management so that
yield (and therefore household food security and nutrition) is maximised by
the end of the growing season.

Observation sources map to realistic low-resource tools:
- Visual inspection / simple phone camera (disease severity)
- Soil moisture probe or farmer report
- Local weather station / radio bulletin
- Community resource log (labour, organic inputs, limited chemical)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any

# Minimal Gymnasium-compatible spaces so the project runs even if
# gymnasium is not yet installed (uv sync will pull the real package).
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class _Discrete:
        def __init__(self, n: int):
            self.n = n
            self.shape = ()
            self.dtype = np.int64
        def sample(self):
            return int(np.random.randint(0, self.n))
        def contains(self, x):
            return 0 <= int(x) < self.n

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
        def sample(self):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
        def contains(self, x):
            return True

    class spaces:
        Discrete = _Discrete
        Box = _Box

    class gym:
        class Env:
            pass


class RwandaCropGuardEnv(gym.Env):
    """
    Custom discrete-action environment representing a smallholder farm
    in rural Rwanda during one growing season.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    # Action meanings (exhaustive and real-world mappable)
    ACTIONS = {
        0: "INSPECT",          # Visual / phone-camera check of current plot
        1: "APPLY_ORGANIC",    # Compost, neem, local botanical treatment
        2: "APPLY_CHEMICAL",   # Limited synthetic (only if stock > 0)
        3: "IRRIGATE",         # Water / mulch to improve plant resilience
        4: "MOVE_NEXT",        # Walk / bicycle to next plot
        5: "ALERT_COMMUNITY",  # Report to village health / agronomy worker
        6: "WAIT_MONITOR",     # Observe without intervention
        7: "HARVEST",          # End-of-season harvest action (terminal)
    }

    def __init__(
        self,
        num_plots: int = 4,
        season_length: int = 60,          # decision steps ≈ days
        max_chemical: int = 3,
        max_organic: int = 8,
        max_labour: int = 12,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_plots = num_plots
        self.season_length = season_length
        self.max_chemical = max_chemical
        self.max_organic = max_organic
        self.max_labour = max_labour
        self.render_mode = render_mode

        # Discrete action space
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Observation: continuous vector (easy for both DQN and policy methods)
        # [current_plot_onehot (num_plots),
        #  disease_severity (num_plots),
        #  soil_moisture (num_plots),
        #  weather_index,
        #  chemical_stock_norm,
        #  organic_stock_norm,
        #  labour_norm,
        #  day_norm,
        #  cumulative_yield_risk]
        # onehot + disease + moisture + [weather, chem, org, labour, day, risk] = 4*3 + 6
        obs_dim = self.num_plots * 3 + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._state = None
        self._step_count = 0
        self._current_plot = 0
        self._chemical = 0
        self._organic = 0
        self._labour = 0
        self._disease = None
        self._moisture = None
        self._weather = 0.5
        self._yield_risk = 0.0
        self._inspected = None
        self._last_action = None
        self._episode_reward = 0.0

    def _get_obs(self) -> np.ndarray:
        plot_onehot = np.zeros(self.num_plots, dtype=np.float32)
        plot_onehot[self._current_plot] = 1.0
        obs = np.concatenate(
            [
                plot_onehot,
                self._disease.astype(np.float32),
                self._moisture.astype(np.float32),
                np.array(
                    [
                        self._weather,
                        self._chemical / max(self.max_chemical, 1),
                        self._organic / max(self.max_organic, 1),
                        self._labour / max(self.max_labour, 1),
                        self._step_count / self.season_length,
                        np.clip(self._yield_risk, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_plot": self._current_plot,
            "disease": self._disease.copy(),
            "moisture": self._moisture.copy(),
            "chemical": self._chemical,
            "organic": self._organic,
            "labour": self._labour,
            "day": self._step_count,
            "yield_risk": self._yield_risk,
            "last_action": self._last_action,
            "episode_reward": self._episode_reward,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._current_plot = 0
        self._chemical = self.max_chemical
        self._organic = self.max_organic
        self._labour = self.max_labour

        # Realistic starting conditions for Rwanda smallholder (variable)
        self._disease = self._rng.uniform(0.0, 0.25, size=self.num_plots)
        self._moisture = self._rng.uniform(0.3, 0.7, size=self.num_plots)
        self._weather = float(self._rng.uniform(0.3, 0.8))
        self._yield_risk = 0.0
        self._inspected = np.zeros(self.num_plots, dtype=bool)
        self._last_action = None
        self._episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._last_action = action
        reward = 0.0
        terminated = False
        truncated = False

        # ---- Action effects (inspect early → treat immediately → irrigate dry → harvest late) ----
        day_frac = self._step_count / max(1, self.season_length)
        early = max(0.0, 1.0 - day_frac)          # 1 at start → 0 at end
        mid_early = max(0.0, 1.0 - day_frac * 1.4)  # decays faster; strong only in first ~40%

        if action == 0:  # INSPECT
            if self._labour > 0:
                self._labour -= 1
                self._inspected[self._current_plot] = True
                sev = float(self._disease[self._current_plot])
                # Detection is useful early; weak later so it cannot be farmed
                if sev > 0.10:
                    reward += 0.9 + 2.0 * sev * mid_early
                else:
                    reward += 0.2 * mid_early
            else:
                reward -= 0.6

        elif action == 1:  # APPLY_ORGANIC
            if self._organic > 0 and self._labour > 0:
                self._organic -= 1
                self._labour -= 1
                sev_before = float(self._disease[self._current_plot])
                self._disease[self._current_plot] = max(0.0, sev_before - 0.30)
                self._moisture[self._current_plot] = min(
                    1.0, self._moisture[self._current_plot] + 0.06
                )
                # Big bonus for treating sick plots early; extra if already inspected
                inspected_bonus = 1.2 if self._inspected[self._current_plot] else 0.0
                reward += 2.5 + 5.0 * sev_before * (0.35 + 0.65 * mid_early) + inspected_bonus
            else:
                reward -= 1.0

        elif action == 2:  # APPLY_CHEMICAL
            if self._chemical > 0 and self._labour > 0:
                self._chemical -= 1
                self._labour -= 1
                sev_before = float(self._disease[self._current_plot])
                self._disease[self._current_plot] = max(0.0, sev_before - 0.50)
                inspected_bonus = 1.2 if self._inspected[self._current_plot] else 0.0
                reward += 3.0 + 5.5 * sev_before * (0.35 + 0.65 * mid_early) + inspected_bonus
            else:
                reward -= 1.2

        elif action == 3:  # IRRIGATE
            if self._labour > 0:
                self._labour -= 1
                moist_before = float(self._moisture[self._current_plot])
                self._moisture[self._current_plot] = min(1.0, moist_before + 0.30)
                dryness = 1.0 - moist_before
                reward += 1.2 + 3.0 * dryness
            else:
                reward -= 0.5

        elif action == 4:  # MOVE_NEXT
            self._current_plot = (self._current_plot + 1) % self.num_plots
            # Encourage coverage early only
            reward += 0.12 * mid_early

        elif action == 5:  # ALERT_COMMUNITY
            if self._labour > 0:
                self._labour -= 1
                mean_sev = float(np.mean(self._disease))
                self._disease = np.maximum(0.0, self._disease - 0.12)
                reward += 2.0 + 1.5 * mean_sev * mid_early
            else:
                reward -= 0.4

        elif action == 6:  # WAIT_MONITOR
            reward -= 0.20

        elif action == 7:  # HARVEST
            terminated = True
            avg_health = 1.0 - float(np.mean(self._disease))
            avg_moisture = float(np.mean(self._moisture))
            final_yield = avg_health * (0.55 + 0.45 * avg_moisture)
            reward += 24.0 * final_yield - 7.0 * self._yield_risk
            # Only allow good harvest near end of season
            if self._step_count < int(0.82 * self.season_length):
                reward -= 15.0
            elif self._step_count < int(0.90 * self.season_length):
                reward -= 5.0
            if self._chemical > 0:
                reward += 0.8

        # ---- Environment dynamics ----
        if not terminated:
            # Slightly slower spread so timely treatment can keep plots healthy
            weather_stress = 1.0 + 0.35 * (1.0 - self._weather)
            spread = self._rng.uniform(0.010, 0.035, size=self.num_plots) * weather_stress
            for i in range(self.num_plots):
                left = self._disease[(i - 1) % self.num_plots]
                right = self._disease[(i + 1) % self.num_plots]
                cross = 0.028 * (left + right) / 2.0
                self._disease[i] = np.clip(
                    self._disease[i] + spread[i] + cross, 0.0, 1.0
                )

            self._moisture = np.clip(
                self._moisture - self._rng.uniform(0.02, 0.048, size=self.num_plots),
                0.0,
                1.0,
            )

            self._weather = float(
                np.clip(self._weather + self._rng.normal(0, 0.05), 0.1, 0.95)
            )

            risk_inc = 0.14 * np.mean(self._disease) + 0.08 * (1.0 - np.mean(self._moisture))
            self._yield_risk = min(1.0, self._yield_risk + risk_inc)

            # Ongoing cost of unmanaged disease (stronger mid-season)
            reward -= 0.45 * float(np.mean(self._disease)) * (0.5 + 0.5 * day_frac)

            self._step_count += 1
            if self._step_count >= self.season_length:
                truncated = True
                avg_health = 1.0 - float(np.mean(self._disease))
                avg_moisture = float(np.mean(self._moisture))
                final_yield = avg_health * (0.55 + 0.45 * avg_moisture)
                reward += 18.0 * final_yield - 5.0 * self._yield_risk

        self._episode_reward += reward
        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ("human", "rgb_array"):
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        lines = [
            f"Day {self._step_count}/{self.season_length} | Plot {self._current_plot} | "
            f"Chem={self._chemical} Org={self._organic} Lab={self._labour}",
            f"Weather={self._weather:.2f}  YieldRisk={self._yield_risk:.2f}",
            "Plot | Disease | Moisture | Inspected",
        ]
        for i in range(self.num_plots):
            mark = "<<" if i == self._current_plot else "  "
            lines.append(
                f"{mark}{i:2d}  |  {self._disease[i]:.2f}   |   {self._moisture[i]:.2f}   | "
                f"{'Y' if self._inspected[i] else 'N'}"
            )
        if self._last_action is not None:
            lines.append(f"Last action: {self.ACTIONS[self._last_action]}")
        return "\n".join(lines)

    def _render_rgb(self) -> Optional[np.ndarray]:
        """Simple but informative RGB visualisation (can be extended to 3D)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
            import io
            from PIL import Image
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, self.num_plots + 1)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Rwanda CropGuard – Day {self._step_count} | Risk {self._yield_risk:.2f}",
            fontsize=12,
            color="#1a5f2a",
        )

        for i in range(self.num_plots):
            # Soil / plot base
            color = plt.cm.YlGn(0.3 + 0.5 * self._moisture[i])
            rect = Rectangle((i + 0.15, 0.8), 0.7, 1.8, facecolor=color, edgecolor="#553311", lw=2)
            ax.add_patch(rect)

            # Disease overlay (red intensity)
            if self._disease[i] > 0.05:
                drect = Rectangle(
                    (i + 0.15, 0.8),
                    0.7,
                    1.8 * self._disease[i],
                    facecolor=(0.8, 0.1, 0.1, 0.55),
                    edgecolor="none",
                )
                ax.add_patch(drect)

            # Current plot marker
            if i == self._current_plot:
                circ = Circle((i + 0.5, 3.0), 0.18, facecolor="#ffcc00", edgecolor="black")
                ax.add_patch(circ)
                ax.text(i + 0.5, 3.35, "AGENT", ha="center", fontsize=8, fontweight="bold")

            ax.text(i + 0.5, 0.5, f"P{i}", ha="center", fontsize=9)
            ax.text(
                i + 0.5,
                0.25,
                f"D{self._disease[i]:.2f}",
                ha="center",
                fontsize=7,
                color="darkred",
            )

        # Resource bar
        ax.text(0.2, 3.7, f"Chem:{self._chemical}  Org:{self._organic}  Lab:{self._labour}", fontsize=9)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        return img

    def close(self):
        pass


# Convenience registration helper
def make_env(render_mode: Optional[str] = None, **kwargs) -> RwandaCropGuardEnv:
    return RwandaCropGuardEnv(render_mode=render_mode, **kwargs)
