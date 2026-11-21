"""
RwandaCropGuard-v0: six-plot Rwandan mixed-smallholder farm.
The six managed plots are ordered Beans 1, Maize 1, Cassava 1,
Beans 2, Maize 2, Cassava 2. Observation layout remains a compact
continuous vector: plot one-hot, disease[6], moisture[6], and six globals.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Discrete:
        def __init__(self,n): self.n=n; self.shape=(); self.dtype=np.int64
        def sample(self): return int(np.random.randint(self.n))
        def contains(self,x): return isinstance(x,(int,np.integer)) and 0<=int(x)<self.n
    class _Box:
        def __init__(self,low,high,shape,dtype=np.float32): self.low=low; self.high=high; self.shape=shape; self.dtype=dtype
        def sample(self): return np.random.uniform(self.low,self.high,self.shape).astype(self.dtype)
        def contains(self,x): return np.asarray(x).shape==self.shape
    class spaces: Discrete=_Discrete; Box=_Box
    class gym:
        class Env: pass

class RwandaCropGuardEnv(gym.Env):
    metadata={"render_modes":["human","rgb_array","ansi"],"render_fps":4}
    ACTIONS={0:"INSPECT",1:"APPLY_ORGANIC",2:"APPLY_CHEMICAL",3:"IRRIGATE",4:"MOVE_NEXT",5:"ALERT_COMMUNITY",6:"WAIT_MONITOR",7:"HARVEST"}
    CROP_TYPES=("beans","maize","cassava","beans","maize","cassava")
    def __init__(self,num_plots:int=6,season_length:int=90,max_chemical:int=3,max_organic:int=24,max_labour:int=45,render_mode:Optional[str]=None,seed:Optional[int]=None):
        super().__init__()
        if num_plots != 6:
            # The viewer and crop rotation use a fixed six-plot village layout.
            raise ValueError("RwandaCropGuardEnv uses exactly six crop plots.")
        self.num_plots=6; self.season_length=season_length
        self.max_chemical=max_chemical; self.max_organic=max_organic; self.max_labour=max_labour
        self.render_mode=render_mode
        self.action_space=spaces.Discrete(8)
        self.observation_space=spaces.Box(0.,1.,shape=(self.num_plots*3+6,),dtype=np.float32)
        self._rng=np.random.default_rng(seed)
        self._step_count=0; self._current_plot=0; self._chemical=0; self._organic=0; self._labour=0
        self._disease=np.zeros(6); self._moisture=np.zeros(6); self._weather=.5; self._yield_risk=0.
        self._inspected=np.zeros(6,dtype=bool); self._last_action=None; self._episode_reward=0.
        self._harvested=False; self._alerted=False
    def _get_obs(self):
        hot=np.zeros(6,dtype=np.float32); hot[self._current_plot]=1
        return np.concatenate([hot,self._disease.astype(np.float32),self._moisture.astype(np.float32),
            np.array([self._weather,self._chemical/self.max_chemical,self._organic/self.max_organic,
            self._labour/self.max_labour,self._step_count/self.season_length,np.clip(self._yield_risk,0,1)],dtype=np.float32)])
    def _get_info(self):
        return {"current_plot":self._current_plot,"crop_type":self.CROP_TYPES[self._current_plot],
            "crop_types":list(self.CROP_TYPES),"disease":self._disease.copy(),"moisture":self._moisture.copy(),
            "chemical":self._chemical,"organic":self._organic,"labour":self._labour,"day":self._step_count,
            "yield_risk":self._yield_risk,"last_action":self._last_action,"episode_reward":self._episode_reward,
            "harvest_maturity_day":78}
    def reset(self,*,seed=None,options=None):
        if seed is not None: self._rng=np.random.default_rng(seed)
        self._step_count=0; self._current_plot=0
        self._chemical=self.max_chemical; self._organic=self.max_organic; self._labour=self.max_labour
        # Slight crop-to-crop variation, healthy but non-trivial initial conditions.
        self._disease=self._rng.uniform(.035,.16,6)
        self._moisture=self._rng.uniform(.58,.86,6)
        self._weather=float(self._rng.uniform(.38,.78)); self._yield_risk=0.
        self._inspected=np.zeros(6,dtype=bool); self._last_action=None; self._episode_reward=0.; self._harvested=False; self._alerted=False
        return self._get_obs(),self._get_info()
    def _final_yield(self):
        health=1.-float(np.mean(self._disease))
        moisture=float(np.mean(self._moisture))
        # Moderate moisture shortfall is recoverable; severe drought is costly.
        water_factor=.55+.45*np.clip(moisture,0,1)
        return float(np.clip(health*water_factor,0,1))
    def step(self,action:int)->Tuple[np.ndarray,float,bool,bool,Dict[str,Any]]:
        if not self.action_space.contains(action): raise ValueError(f"Invalid action {action}")
        action=int(action); self._last_action=action
        reward=0.; terminated=False; truncated=False
        day_frac=self._step_count/max(1,self.season_length)
        early=max(0.,1.-day_frac); mid=max(0.,1.-day_frac*1.5)
        p=self._current_plot; sev=float(self._disease[p]); moist=float(self._moisture[p])
        # Desired policy: inspect each plot → treat only when diseased →
        # irrigate when dry/sunny → alert near harvest → harvest after day 78.
        sunny = max(0., self._weather - 0.55)  # prolonged clear/dry pressure

        if action==0:  # INSPECT — reward first inspection of each plot
            if self._labour>0:
                self._labour-=1
                if not self._inspected[p]:
                    self._inspected[p]=True
                    # Cover all plots early
                    reward += 1.1 + 1.0*sev*mid
                else:
                    reward -= 0.35  # re-inspecting is wasteful
            else:
                reward -= 0.5

        elif action==1:  # APPLY_ORGANIC — only pay off if disease is present
            if self._organic>0 and self._labour>0:
                self._organic-=1; self._labour-=1
                if sev < 0.10:
                    # Spamming organic on healthy plots is punished
                    reward -= 1.2
                else:
                    reduction = 0.26 + 0.06*early
                    self._disease[p] = max(0., sev - reduction)
                    self._moisture[p] = min(1., moist + 0.08)
                    reward += 1.5 + 5.5*sev*(0.4 + 0.6*mid)
                    if self._inspected[p]:
                        reward += 0.8
            else:
                reward -= 0.9

        elif action==2:  # APPLY_CHEMICAL — only when disease is meaningful
            if self._chemical>0 and self._labour>0:
                self._chemical-=1; self._labour-=1
                if sev < 0.18:
                    reward -= 1.5
                else:
                    self._disease[p] = max(0., sev - 0.45)
                    reward += 1.8 + 6.0*sev*(0.4 + 0.6*mid)
                    if self._inspected[p]:
                        reward += 0.9
            else:
                reward -= 1.1

        elif action==3:  # IRRIGATE — strong when dry, especially under sun/drought
            if self._labour>0:
                self._labour-=1
                dryness = max(0., 0.55 - moist)
                gain = min(0.36, 1.0 - moist)
                self._moisture[p] = min(1., moist + gain)
                # Sunny + dry = clear irrigation need
                reward += 0.4 + 4.0*dryness + 2.0*sunny*dryness + 0.5*gain
                if moist > 0.7:
                    reward -= 0.8  # watering wet soil is wasteful
            else:
                reward -= 0.45

        elif action==4:  # MOVE_NEXT — prefer visiting uninspected plots early
            nxt = (p + 1) % 6
            self._current_plot = nxt
            if not self._inspected[nxt] and mid > 0.3:
                reward += 0.35
            else:
                reward += 0.05 * mid

        elif action==5:  # ALERT_COMMUNITY — disease help early; harvest call late
            if self._labour >= 2:
                self._labour -= 2
                before = float(np.mean(self._disease))
                self._disease = np.maximum(0., self._disease - 0.08)
                if day_frac >= 0.82:
                    # Call community for harvest time
                    self._alerted = True
                    reward += 3.5 + 1.0 * before
                else:
                    reward += 0.5 + 2.0 * before * mid
            else:
                reward -= 0.55

        elif action==6:  # WAIT
            reward -= 0.15

        elif action==7:  # HARVEST
            if self._step_count < 78:
                reward -= 8.0  # failed attempt; episode continues
            else:
                terminated = True
                self._harvested = True
                y = self._final_yield()
                maturity = min(1., 0.90 + 0.10 * (self._step_count - 78) / 12)
                reward += 40.0 * y * maturity - 6.0 * self._yield_risk
                if self._alerted:
                    reward += 4.0  # alerted community before harvest
                # Coverage bonus: inspected most plots
                reward += 1.5 * float(np.mean(self._inspected.astype(np.float32)))
        # Environment advances one simulated day after every non-terminal action.
        if not terminated:
            weather_stress=1.+.22*(1.-self._weather)
            # Gradual disease pressure; prompt treatment can keep it under control.
            spread=self._rng.uniform(.0015,.008,6)*weather_stress
            neighbour=np.roll(self._disease,1)+np.roll(self._disease,-1)
            self._disease=np.clip(self._disease+spread+.009*neighbour/2.,0.,1.)
            # Water loss is slower and variable; drought still requires active irrigation.
            evap=self._rng.uniform(.008,.022,6)*(1.15+.35*max(0.,self._weather-0.5))
            self._moisture=np.clip(self._moisture-evap,0.,1.)
            self._weather=float(np.clip(self._weather+self._rng.normal(0,.045),.12,.96))
            risk_inc=.075*float(np.mean(self._disease))+.055*float(np.mean(1.-self._moisture))
            self._yield_risk=min(1.,self._yield_risk+risk_inc)
            reward-=.55*float(np.mean(self._disease))*(.45+.55*day_frac)
            self._step_count+=1
            if self._step_count>=self.season_length:
                truncated=True
                reward += 34.*self._final_yield()-5.5*self._yield_risk
        self._episode_reward+=reward
        return self._get_obs(),float(reward),terminated,truncated,self._get_info()
    def _render_ansi(self):
        lines=[f"Day {self._step_count}/{self.season_length} | Plot {self._current_plot} ({self.CROP_TYPES[self._current_plot]}) | Chem={self._chemical} Org={self._organic} Labour={self._labour}",
            f"Weather={self._weather:.2f} YieldRisk={self._yield_risk:.2f} | harvest opens day 78",
            "Plot | Crop | Disease | Moisture"]
        for i,c in enumerate(self.CROP_TYPES):
            mark="<<" if i==self._current_plot else "  "
            lines.append(f"{mark}{i} | {c:7s} | {self._disease[i]:.2f} | {self._moisture[i]:.2f}")
        if self._last_action is not None: lines.append("Last action: "+self.ACTIONS[self._last_action])
        return "\n".join(lines)
    def render(self):
        if self.render_mode=="ansi": return self._render_ansi()
        return None
    def close(self): pass

def make_env(render_mode:Optional[str]=None,**kwargs):
    return RwandaCropGuardEnv(render_mode=render_mode,**kwargs)
