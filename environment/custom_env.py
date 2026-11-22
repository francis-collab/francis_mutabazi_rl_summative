"""
RwandaCropGuard-v0: six-plot Rwandan mixed-smallholder farm.

Mission: an agricultural extension worker inspects and treats six plots,
manages water, alerts the community once when needed, and harvests the
whole farm during the maturity window.

Observation (26 values):
  current-plot one-hot[6], disease[6], moisture[6],
  weather, chemical, organic, labour, season progress, yield risk,
  water reserve, community-alert-used.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Discrete:
        def __init__(self, n): self.n=n; self.shape=(); self.dtype=np.int64
        def sample(self): return int(np.random.randint(self.n))
        def contains(self, x): return isinstance(x,(int,np.integer)) and 0 <= int(x) < self.n
    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low=low; self.high=high; self.shape=shape; self.dtype=dtype
        def sample(self): return np.random.uniform(self.low,self.high,self.shape).astype(self.dtype)
        def contains(self,x): return np.asarray(x).shape==self.shape
    class spaces: Discrete=_Discrete; Box=_Box
    class gym:
        class Env: pass


class RwandaCropGuardEnv(gym.Env):
    metadata={"render_modes":["human","rgb_array","ansi"],"render_fps":4}
    ACTIONS={
        0:"INSPECT", 1:"APPLY_ORGANIC", 2:"APPLY_CHEMICAL",
        3:"IRRIGATE", 4:"MOVE_NEXT", 5:"ALERT_COMMUNITY",
        6:"WAIT_MONITOR", 7:"HARVEST"
    }
    CROP_TYPES=("beans","maize","cassava","beans","maize","cassava")
    HARVEST_OPEN_DAY=78

    def __init__(self, num_plots:int=6, season_length:int=90,
                 max_chemical:int=3, max_organic:int=24, max_labour:int=45,
                 max_water:int=10, render_mode:Optional[str]=None,
                 seed:Optional[int]=None):
        super().__init__()
        if num_plots != 6:
            raise ValueError("RwandaCropGuardEnv uses exactly six crop plots.")
        self.num_plots=6
        self.season_length=int(season_length)
        self.max_chemical=max_chemical
        self.max_organic=max_organic
        self.max_labour=max_labour
        self.max_water=max_water
        self.render_mode=render_mode
        self.action_space=spaces.Discrete(8)
        # 6 plot flags + 6 disease + 6 moisture + 8 global state values.
        self.observation_space=spaces.Box(0.,1.,shape=(26,),dtype=np.float32)
        self._rng=np.random.default_rng(seed)
        self._step_count=0
        self._current_plot=0
        self._chemical=0; self._organic=0; self._labour=0; self._water=0
        self._disease=np.zeros(6,dtype=np.float32)
        self._disease_onset=np.arange(10, 16, dtype=np.int32)  # placeholder; set in reset
        self._moisture=np.zeros(6,dtype=np.float32)
        self._weather=.5; self._yield_risk=0.
        self._inspected=np.zeros(6,dtype=bool)
        self._treated=np.zeros(6,dtype=bool)
        self._last_action=None; self._episode_reward=0.
        self._harvested=False; self._alerted=False
        self._steps_on_plot=0

    def _growth_stage(self, plot:int) -> float:
        # Species-specific maturity curves over the compressed 90-day demo season.
        progress=self._step_count/max(1,self.season_length)
        crop=self.CROP_TYPES[plot]
        maturity={"beans":0.76,"maize":0.88,"cassava":1.00}[crop]
        return float(np.clip(progress/maturity,0.,1.))

    def action_mask(self):
        """Return valid actions for the current mission state.

        MOVE_NEXT stays available all season so the agent can circulate when
        disease reappears. Harvest is locked until maturity; community alert
        is one-time; on the final decision day harvest is mandatory.
        """
        mask = np.ones(8, dtype=bool)
        p = self._current_plot
        sev = float(self._disease[p])
        mean_moisture = float(np.mean(self._moisture))
        mean_disease = float(np.mean(self._disease))
        # Inspect always allowed (re-inspect has a small reward penalty).
        mask[0] = self._labour > 0
        # Treat as soon as disease is present on the current plot (and not yet cured).
        untreated = not bool(self._treated[p])
        mask[1] = self._organic > 0 and self._labour > 0 and sev >= 0.05 and untreated
        mask[2] = self._chemical > 0 and self._labour > 0 and sev >= 0.10 and untreated
        mask[3] = self._water > 0 and self._labour > 0 and mean_moisture < 0.80
        mask[4] = True  # MOVE always available — agent should go to sick plots
        mask[5] = (not self._alerted and self._labour >= 2 and
                   (mean_disease >= 0.14 or
                    self._step_count / max(1, self.season_length) >= 0.72))
        mask[6] = True  # WAIT always available
        mask[7] = self._step_count >= self.HARVEST_OPEN_DAY
        if self._step_count >= self.season_length - 1:
            mask[:] = False
            mask[7] = True
        return mask

    def _get_obs(self):
        hot=np.zeros(6,dtype=np.float32); hot[self._current_plot]=1.
        globals_=np.array([
            self._weather,
            self._chemical/max(1,self.max_chemical),
            self._organic/max(1,self.max_organic),
            self._labour/max(1,self.max_labour),
            self._step_count/max(1,self.season_length),
            np.clip(self._yield_risk,0.,1.),
            self._water/max(1,self.max_water),
            float(self._alerted),
        ],dtype=np.float32)
        return np.concatenate([hot,self._disease,self._moisture,globals_]).astype(np.float32)

    def _get_info(self):
        growth=[self._growth_stage(i) for i in range(6)]
        health=np.clip(1.-self._disease,0.,1.)
        return {
            "current_plot":self._current_plot,
            "crop_type":self.CROP_TYPES[self._current_plot],
            "crop_types":list(self.CROP_TYPES),
            "disease":self._disease.copy(),
            "moisture":self._moisture.copy(),
            "crop_growth":growth,
            "crop_health":health.copy(),
            "chemical":self._chemical,
            "organic":self._organic,
            "labour":self._labour,
            "water":self._water,
            "day":self._step_count,
            "season_length":self.season_length,
            "weather":self._weather,
            "weather_state":"hot_sunny" if self._weather>=.68 else ("cloudy" if self._weather<=.32 else "mild"),
            "yield_risk":self._yield_risk,
            "last_action":self._last_action,
            "episode_reward":self._episode_reward,
            "harvest_maturity_day":self.HARVEST_OPEN_DAY,
            "alerted":self._alerted,
            "harvested":self._harvested,
            "inspected":self._inspected.copy(),
            "action_mask":self.action_mask().copy(),
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None: self._rng=np.random.default_rng(seed)
        self._step_count=0; self._current_plot=0
        self._chemical=self.max_chemical
        self._organic=self.max_organic
        self._labour=self.max_labour
        self._water=self.max_water
        # Disease starts near zero; each plot gets an onset day in [10, 60].
        self._disease = self._rng.uniform(0.0, 0.03, 6).astype(np.float32)
        self._disease_onset = np.sort(
            self._rng.integers(10, 61, size=6)
        ).astype(np.int32)  # staggered: one plot after another through day 60
        self._moisture = self._rng.uniform(0.58, 0.86, 6).astype(np.float32)
        self._weather = float(self._rng.uniform(0.58, 0.82))
        self._yield_risk = 0.
        self._inspected = np.zeros(6, dtype=bool)
        self._treated = np.zeros(6, dtype=bool)  # once True, disease never returns on that plot
        self._last_action = None
        self._episode_reward = 0.
        self._harvested = False
        self._alerted = False
        self._steps_on_plot = 0
        return self._get_obs(),self._get_info()

    def _final_yield(self):
        health=1.-float(np.mean(self._disease))
        moisture=float(np.mean(self._moisture))
        water_factor=.45+.55*np.clip(moisture,0.,1.)
        growth=np.mean([self._growth_stage(i) for i in range(6)])
        return float(np.clip(health*water_factor*growth,0.,1.))

    def step(self, action:int)->Tuple[np.ndarray,float,bool,bool,Dict[str,Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action=int(action); self._last_action=action
        reward=0.; terminated=False; truncated=False
        day_frac=self._step_count/max(1,self.season_length)
        early=max(0.,1.-day_frac)
        mid=max(0.,1.-day_frac*1.5)
        p=self._current_plot
        sev=float(self._disease[p]); moist=float(self._moisture[p])
        sunny=max(0.,(self._weather-.55)/.45)

        if action==0:  # INSPECT selected plot
            if self._labour>0:
                self._labour-=1
                if not self._inspected[p]:
                    self._inspected[p]=True
                    reward += 1.2 + 1.0 * sev * (0.5 + 0.5 * early)
                else:
                    # Re-inspecting the same plot is wasteful — agent should move or treat.
                    reward -= 1.25
            else:
                reward -= 0.5

        elif action==1:  # ORGANIC treatment on selected plot
            if self._organic > 0 and self._labour > 0:
                self._organic -= 1
                self._labour -= 1
                if self._treated[p]:
                    reward -= 0.8  # already cured — waste
                elif sev < 0.05:
                    reward -= 0.6
                else:
                    # Permanent cure: disease cleared and never returns on this plot.
                    self._disease[p] = 0.0
                    self._moisture[p] = min(1.0, moist + 0.045)
                    self._treated[p] = True
                    # Treating immediately is the highest-value action in the mission.
                    reward += 5.0 + 8.0 * sev * (0.7 + 0.3 * early)
                    if self._inspected[p]:
                        reward += 1.0
            else:
                reward -= 0.9

        elif action==2:  # CHEMICAL treatment on selected plot
            if self._chemical > 0 and self._labour > 0:
                self._chemical -= 1
                self._labour -= 1
                if self._treated[p]:
                    reward -= 1.0
                elif sev < 0.08:
                    reward -= 0.9
                else:
                    self._disease[p] = 0.0
                    self._treated[p] = True  # permanent cure
                    reward += 5.5 + 8.5 * sev * (0.7 + 0.3 * early)
                    if self._inspected[p]:
                        reward += 1.1
            else:
                reward -= 1.0

        elif action==3:  # IRRIGATE all six plots from the shared water point
            if self._water > 0 and self._labour > 0:
                self._water -= 1
                self._labour -= 1
                dryness = np.maximum(0., 0.64 - self._moisture)
                gains = np.minimum(0.20, 1. - self._moisture)
                self._moisture = np.clip(self._moisture + gains, 0., 1.)
                mean_need = float(np.mean(dryness))
                mean_gain = float(np.mean(gains))
                reward += 0.45 + 5.0 * mean_need + 1.8 * sunny * mean_need + 0.8 * mean_gain
                if 45 <= self._step_count <= 65:
                    reward += 2.2 + 2.5 * sunny
                elif self._step_count < 30:
                    reward -= 0.4
                elif self._step_count > 75:
                    reward -= 0.5
                if float(np.mean(self._moisture)) > 0.85:
                    reward -= 0.65
            else:
                reward -= 0.9

        elif action==4:  # MOVE_NEXT through the farm's passage network
            # Leaving a sick untreated plot is a serious mistake — treat first.
            cur_sev = float(self._disease[p])
            cur_needs = cur_sev >= 0.05 and not bool(self._treated[p])
            if cur_needs:
                # Large penalty for walking away without treating.
                reward -= 2.5 + 4.0 * cur_sev
            self._current_plot = (p + 1) % 6
            self._steps_on_plot = 0
            dest = self._current_plot
            dest_sev = float(self._disease[dest])
            dest_needs = dest_sev >= 0.05 and not bool(self._treated[dest])
            others = [
                float(self._disease[i])
                for i in range(6)
                if i != p and not bool(self._treated[i]) and float(self._disease[i]) >= 0.05
            ]
            max_other = max(others) if others else 0.0
            if dest_needs:
                reward += 1.5 + 2.5 * dest_sev  # arrived where treatment is needed
            elif max_other >= 0.05 and not cur_needs:
                reward += 0.45 + 1.0 * max_other  # only if we did not abandon a sick plot
            elif not self._inspected[dest]:
                reward += 0.35
            else:
                reward += 0.08

        elif action==5:  # One community alert per season, only when justified
            if self._alerted:
                reward -= 2.0
            elif self._labour < 2:
                reward -= 0.7
            else:
                mean_disease = float(np.mean(self._disease))
                near_harvest = self._step_count >= 70
                late = day_frac >= 0.72
                justified = near_harvest or mean_disease >= 0.14
                if justified:
                    self._labour -= 2
                    self._alerted = True
                    # Alert does not re-infect cured plots; only slows untreated disease slightly.
                    for i in range(6):
                        if not self._treated[i]:
                            self._disease[i] = max(0.0, float(self._disease[i]) - 0.02)
                    if near_harvest:
                        reward += 5.0 + 1.0 * mean_disease
                    elif late:
                        reward += 3.2 + 0.9 * mean_disease
                    else:
                        reward += 1.5 + 0.7 * mean_disease
                else:
                    reward -= 1.25

        elif action==6:  # WAIT / MONITOR
            reward -= 0.38

        elif action==7:  # HARVEST whole farm
            if self._step_count < self.HARVEST_OPEN_DAY:
                reward -= 10.0
            else:
                self._harvested=True
                terminated=True
                y=self._final_yield()
                maturity=np.clip(.90+.10*(self._step_count-self.HARVEST_OPEN_DAY)/
                                 max(1,self.season_length-self.HARVEST_OPEN_DAY),.90,1.)
                reward += 90. + 65.*y*maturity - 6.*self._yield_risk
                if self._alerted: reward += 6.
                reward += 2.0 * float(np.mean(self._inspected.astype(np.float32)))
                # Coverage: treating multiple plots during the season is rewarded.
                reward += 10.0 * float(np.mean(self._treated.astype(np.float32)))

        # Anti-camping only if the current plot does NOT need treatment.
        # If this plot is sick and untreated, the agent should stay and apply organic/chemical.
        if not terminated and action != 4:
            self._steps_on_plot = getattr(self, "_steps_on_plot", 0) + 1
            cur_needs = (float(self._disease[self._current_plot]) >= 0.05
                         and not bool(self._treated[self._current_plot]))
            if not cur_needs:
                other_disease = float(np.max(np.delete(self._disease, self._current_plot))) if self.num_plots > 1 else 0.0
                if other_disease >= 0.12 and self._steps_on_plot >= 2:
                    reward -= (0.45 + 0.90 * other_disease) * min(4, self._steps_on_plot - 1)

        # Small end-of-season urgency; avoids spending the final days farming alert rewards.
        if not terminated and self._step_count >= self.HARVEST_OPEN_DAY:
            reward -= 1.0
        if not terminated:
            # --- Staggered disease onset (days 10–60), permanent cure after treatment ---
            for i in range(6):
                if self._treated[i]:
                    # Cured plots stay clean forever.
                    self._disease[i] = 0.0
                    continue
                onset = int(self._disease_onset[i])
                if self._step_count + 1 >= onset:
                    # First infection day: seed a visible infection.
                    if self._disease[i] < 0.05 and self._step_count + 1 == onset:
                        self._disease[i] = float(self._rng.uniform(0.12, 0.22))
                    else:
                        # Natural progression only on untreated infected plots.
                        weather_stress = 1.0 + 0.18 * (1.0 - self._weather)
                        growth = float(self._rng.uniform(0.008, 0.022)) * weather_stress
                        self._disease[i] = min(1.0, float(self._disease[i]) + growth)

            evap = self._rng.uniform(0.007, 0.018, 6) * (1.10 + 0.35 * sunny)
            self._moisture = np.clip(self._moisture - evap, 0., 1.)

            # Days 45–65: hot sunny peak (encourages mid-season irrigation).
            if 45 <= self._step_count <= 65:
                self._weather = float(np.clip(
                    self._weather + self._rng.uniform(0.02, 0.06), 0.68, 0.96))
            else:
                self._weather = float(np.clip(
                    self._weather + self._rng.normal(0, 0.035), 0.12, 0.96))

            # Only untreated disease counts toward risk / penalties.
            active = np.array([
                0.0 if self._treated[i] else float(self._disease[i]) for i in range(6)
            ], dtype=np.float32)
            mean_d = float(np.mean(active))
            max_d = float(np.max(active))
            risk_inc = 0.07 * mean_d + 0.04 * float(np.mean(1.0 - self._moisture)) + 0.03 * max_d
            self._yield_risk = min(1.0, self._yield_risk + risk_inc)

            # Heavy cost for leaving active disease untreated — pushes early organic care.
            early_weight = 0.85 + 0.55 * early
            reward -= (0.85 * mean_d + 0.55 * max_d) * early_weight

            # Bonus for standing on a sick untreated plot (ready to treat).
            if not self._treated[self._current_plot] and self._disease[self._current_plot] >= 0.05:
                reward += 0.25

            self._step_count += 1
            if self._step_count >= self.season_length:
                truncated = True
                reward -= 35.0

        self._episode_reward+=reward
        return self._get_obs(),float(reward),terminated,truncated,self._get_info()

    def _render_ansi(self):
        lines=[
            f"Day {self._step_count}/{self.season_length} | Plot {self._current_plot} "
            f"({self.CROP_TYPES[self._current_plot]}) | Chem={self._chemical} "
            f"Org={self._organic} Water={self._water} Labour={self._labour}",
            f"Weather={self._weather:.2f} YieldRisk={self._yield_risk:.2f} | "
            f"harvest opens day {self.HARVEST_OPEN_DAY}",
            "Plot | Crop | Growth | Disease | Moisture"
        ]
        for i,c in enumerate(self.CROP_TYPES):
            mark="<<" if i==self._current_plot else "  "
            lines.append(f"{mark}{i} | {c:7s} | {self._growth_stage(i):.2f} | "
                         f"{self._disease[i]:.2f} | {self._moisture[i]:.2f}")
        if self._last_action is not None:
            lines.append("Last action: "+self.ACTIONS[self._last_action])
        return "\n".join(lines)

    def render(self):
        if self.render_mode=="ansi": return self._render_ansi()
        return None

    def close(self): pass


def make_env(render_mode:Optional[str]=None,**kwargs):
    return RwandaCropGuardEnv(render_mode=render_mode,**kwargs)
