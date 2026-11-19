"""Basic environment tests."""

from environment.custom_env import RwandaCropGuardEnv


def test_reset_and_step():
    env = RwandaCropGuardEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(10):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        if term or trunc:
            break
    env.close()


def test_action_count():
    env = RwandaCropGuardEnv()
    assert env.action_space.n == 8
