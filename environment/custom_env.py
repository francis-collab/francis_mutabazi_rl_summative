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
        # Treatments only when resources exist and disease is meaningful.
        mask[1] = self._organic > 0 and self._labour > 0 and sev >= 0.10
        mask[2] = self._chemical > 0 and self._labour > 0 and sev >= 0.16
        mask[3] = self._water > 0 and self._labour > 0 and mean_moisture < 0.78
        # Circulation is never locked after the first tour.
        mask[4] = True
        mask[5] = (not self._alerted and self._labour >= 2 and
                   (mean_disease >= 0.16 or
                    self._step_count / max(1, self.season_length) >= 0.80))
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
        self._disease=self._rng.uniform(.035,.16,6).astype(np.float32)
        self._moisture=self._rng.uniform(.58,.86,6).astype(np.float32)
        self._weather=float(self._rng.uniform(.58,.82))
        self._yield_risk=0.
        self._inspected=np.zeros(6,dtype=bool)
        self._treated=np.zeros(6,dtype=bool)
        self._last_action=None; self._episode_reward=0.
        self._harvested=False; self._alerted=False
        self._steps_on_plot=0
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
                    reward += 1.0 + .8*sev*mid
                else:
                    reward -= .30
            else:
                reward -= .5

        elif action==1:  # ORGANIC treatment on selected plot
            if self._organic>0 and self._labour>0:
                self._organic-=1; self._labour-=1
                if sev < .10:
                    reward -= 1.0
                else:
                    reduction=.24+.05*early
                    self._disease[p]=max(0.,sev-reduction)
                    self._moisture[p]=min(1.,moist+.045)
                    self._treated[p]=True
                    reward += 1.3+4.5*sev*(.4+.6*mid)
                    if self._inspected[p]: reward += .5
            else:
                reward -= .9

        elif action==2:  # CHEMICAL treatment on selected plot
            if self._chemical>0 and self._labour>0:
                self._chemical-=1; self._labour-=1
                if sev < .16:
                    reward -= 1.2
                else:
                    self._disease[p]=max(0.,sev-.40)
                    self._treated[p]=True
                    reward += 1.5+5.0*sev*(.4+.6*mid)
                    if self._inspected[p]: reward += .6
            else:
                reward -= 1.0

        elif action==3:  # IRRIGATE all six plots from the shared water point
            if self._water>0 and self._labour>0:
                self._water-=1
                self._labour-=1
                dryness=np.maximum(0.,.64-self._moisture)
                gains=np.minimum(.20,1.-self._moisture)
                self._moisture=np.clip(self._moisture+gains,0.,1.)
                mean_need=float(np.mean(dryness))
                mean_gain=float(np.mean(gains))
                reward += .45+5.0*mean_need+1.8*sunny*mean_need+.8*mean_gain
                if float(np.mean(self._moisture))>.82:
                    reward -= .65
            else:
                reward -= .9

        elif action==4:  # MOVE_NEXT through the farm's passage network
            # Always allow circulation; large bonus when other plots need care.
            other_disease = float(np.max(np.delete(self._disease, p))) if self.num_plots > 1 else 0.0
            self._current_plot = (p + 1) % 6
            self._steps_on_plot = 0
            if other_disease >= 0.10:
                reward += 0.70 + 2.2 * other_disease
            elif not self._inspected[self._current_plot] and mid > 0.25:
                reward += 0.35
            else:
                reward += 0.12

        elif action==5:  # One community alert per season, only when justified
            if self._alerted:
                reward -= 2.0
            elif self._labour<2:
                reward -= .7
            else:
                mean_disease=float(np.mean(self._disease))
                late=day_frac>=.80
                justified=(mean_disease>=.16) or late
                if justified:
                    self._labour-=2
                    self._alerted=True
                    # Community response slows disease spread, but does not replace treatment.
                    self._disease=np.maximum(0.,self._disease-.025)
                    reward += (3.0 if late else 1.8) + .8*mean_disease
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

        # Anti-camping: if other plots are sick, staying put is costly.
        if not terminated and action != 4:
            self._steps_on_plot = getattr(self, "_steps_on_plot", 0) + 1
            other_disease = float(np.max(np.delete(self._disease, self._current_plot))) if self.num_plots > 1 else 0.0
            if other_disease >= 0.12 and self._steps_on_plot >= 2:
                reward -= (0.45 + 0.90 * other_disease) * min(4, self._steps_on_plot - 1)

        # Small end-of-season urgency; avoids spending the final days farming alert rewards.
        if not terminated and self._step_count >= self.HARVEST_OPEN_DAY:
            reward -= 1.0
        if not terminated:
            # Crop stress and disease progress by one simulated day.
            weather_stress=1.+.18*(1.-self._weather)
            spread=self._rng.uniform(.0012,.006,6)*weather_stress
            neighbour=np.roll(self._disease,1)+np.roll(self._disease,-1)
            self._disease=np.clip(self._disease+spread+.006*neighbour/2.,0.,1.)
            evap=self._rng.uniform(.007,.018,6)*(1.10+.35*sunny)
            self._moisture=np.clip(self._moisture-evap,0.,1.)
            # Weather index: high values mean clear/hot, low values mean cloudy/wet.
            self._weather=float(np.clip(self._weather+self._rng.normal(0,.035),.12,.96))
            mean_d = float(np.mean(self._disease))
            max_d = float(np.max(self._disease))
            # Untreated disease (especially the worst plot) raises yield risk faster.
            risk_inc = 0.07 * mean_d + 0.04 * float(np.mean(1.0 - self._moisture)) + 0.03 * max_d
            self._yield_risk = min(1.0, self._yield_risk + risk_inc)
            # Stronger per-step cost for leaving disease untreated on any plot.
            reward -= (0.55 * mean_d + 0.35 * max_d) * (0.40 + 0.60 * day_frac)
            self._step_count+=1
            if self._step_count>=self.season_length:
                # The agent had its final decision on day 89; no harvest means a failed season.
                truncated=True
                reward-=35.0

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
