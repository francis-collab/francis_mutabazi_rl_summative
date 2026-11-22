"""Regression tests for the RwandaCropGuard mission environment."""
import numpy as np
from environment.custom_env import RwandaCropGuardEnv


def test_reset_and_step():
    env = RwandaCropGuardEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (26,)
    assert info["day"] == 0
    assert len(info["crop_growth"]) == 6
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(4)
        assert obs.shape == env.observation_space.shape
        assert 0 <= info["water"] <= env.max_water
        if terminated or truncated:
            break
    env.close()


def test_action_count_and_crop_layout():
    env = RwandaCropGuardEnv()
    assert env.action_space.n == 8
    assert env.CROP_TYPES == ("beans", "maize", "cassava", "beans", "maize", "cassava")


def test_move_always_available_after_tour():
    """MOVE must stay legal after every plot has been inspected (no camping lock)."""
    env = RwandaCropGuardEnv(seed=1)
    env.reset(seed=1)
    mask = env.action_mask()
    assert bool(mask[4])  # MOVE available from the start
    assert bool(mask[6])  # WAIT available
    # Simulate a full inspection tour
    for i in range(6):
        env._inspected[i] = True
    env._current_plot = 5
    mask2 = env.action_mask()
    assert bool(mask2[4]), "MOVE must remain available after all plots inspected"
    assert bool(mask2[6]), "WAIT must remain available after all plots inspected"


def test_irrigation_affects_all_six_plots():
    env = RwandaCropGuardEnv(seed=1)
    env.reset(seed=1)
    env._moisture[:] = .25
    before = env._moisture.copy()
    env.step(3)
    assert np.all(env._moisture > before + .10)
    assert env._water == env.max_water - 1


def test_community_alert_is_one_time_only():
    env = RwandaCropGuardEnv(seed=2)
    env.reset(seed=2)
    for _ in range(73):
        env.step(4)
    _, first_reward, _, _, first_info = env.step(5)
    assert first_info["alerted"] is True
    assert not env.action_mask()[5]
    _, second_reward, _, _, second_info = env.step(5)
    assert second_info["alerted"] is True
    assert second_reward < 0


def test_harvest_terminates_after_maturity_and_is_mandatory_on_last_day():
    env = RwandaCropGuardEnv(seed=3)
    env.reset(seed=3)
    for _ in range(env.HARVEST_OPEN_DAY):
        _, _, terminated, truncated, _ = env.step(4)
        assert not terminated and not truncated
    assert env.action_mask()[7]
    _, reward, terminated, truncated, info = env.step(7)
    assert terminated is True
    assert truncated is False
    assert info["harvested"] is True
    assert reward > 0

    env.reset(seed=4)
    for _ in range(env.season_length - 1):
        env.step(6)
    mask = env.action_mask()
    assert mask.tolist() == [False, False, False, False, False, False, False, True]
