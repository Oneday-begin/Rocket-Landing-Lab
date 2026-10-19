import gymnasium as gym
import numpy as np
import pytest

from rocket_landing.evaluation import rollout, summarize
from rocket_landing.rewards import (
    FlightMetrics, RewardSettings, RewardShaping, hover_eligible, make_env, potential,
)


class ScriptedEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, terminated=False, truncated=False, reward=2.0):
        self.terminated, self.truncated, self.reward = terminated, truncated, reward
        self.obs = np.array([0, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.obs.copy(), {}

    def step(self, action):
        return self.obs.copy(), self.reward, self.terminated, self.truncated, {}


def test_hover_rule_pays_airborne_but_not_ground_contact():
    base = ScriptedEnv()
    env = RewardShaping(base, RewardSettings(hover_bonus=1.0))
    env.reset()
    assert env.step(0)[1] == 3.0
    base.obs[6:] = 1
    assert env.step(0)[1] == 2.0


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_fixed_reward_distinguishes_terminal_from_time_limit(terminated, truncated):
    env = RewardShaping(ScriptedEnv(terminated, truncated),
                        RewardSettings(time_cost=0.03, progress_scale=10))
    obs, _ = env.reset()
    before = potential(obs)
    _, reward, term, trunc, info = env.step(0)
    after = 0.0 if terminated else before
    expected = 10 * (0.99 * after - before) - 0.03
    assert reward == pytest.approx(2.0 + expected)
    assert info["base_reward"] == 2.0
    assert (term, trunc) == (terminated, truncated)


def test_reset_clears_previous_episode_potential():
    base = ScriptedEnv()
    env = RewardShaping(base, RewardSettings(progress_scale=10))
    env.reset()
    base.obs[1] = 2
    env.step(0)
    base.obs[1] = 0.5
    obs, _ = env.reset()
    assert env.previous_potential == potential(obs)


@pytest.mark.parametrize("reward,terminated,truncated,outcome", [
    (100, True, False, "landed"), (-100, True, False, "failed"),
    (2, False, True, "timeout"), (-100, True, True, "failed"),
])
def test_outcomes_and_original_reward_survive_shaping(reward, terminated, truncated, outcome):
    env = FlightMetrics(RewardShaping(ScriptedEnv(terminated, truncated, reward),
                                      RewardSettings(time_cost=0.03)))
    env.reset()
    flight = env.step(2)[-1]["flight"]
    assert flight["outcome"] == outcome
    assert flight["base_return"] == reward
    assert flight["shaped_return"] == pytest.approx(reward - 0.03)
    assert flight["fuel_proxy"] == pytest.approx(0.3)


def test_shaping_does_not_change_physics():
    baseline, shaped = make_env(), make_env("fixed")
    try:
        a, _ = baseline.reset(seed=123)
        b, _ = shaped.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        for action in [0, 2, 2, 1, 3] * 8:
            a, _, term_a, trunc_a, _ = baseline.step(action)
            b, _, term_b, trunc_b, _ = shaped.step(action)
            np.testing.assert_array_equal(a, b)
            assert (term_a, trunc_a) == (term_b, trunc_b)
            if term_a or trunc_a:
                break
    finally:
        baseline.close()
        shaped.close()


def test_original_evaluation_is_repeatable():
    first = rollout(episodes=2, seed=765)
    second = rollout(episodes=2, seed=765)
    assert first == second
    assert all(r["base_return"] == r["shaped_return"] for r in first)
    stats = summarize(first)
    assert stats["episodes"] == 2
    assert sum(r["outcome"] in ("landed", "failed", "timeout") for r in first) == 2


def test_positive_bonus_rule_excludes_fast_falling_rocket():
    assert not hover_eligible([0, 0.5, 0, -1, 0, 0, 0, 0])
