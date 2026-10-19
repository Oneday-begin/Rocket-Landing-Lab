"""Reward experiments; the environment's physics and observations never change."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class RewardSettings:
    time_cost: float = 0.0
    hover_bonus: float = 0.0
    progress_scale: float = 0.0
    gamma: float = 0.99


PRESETS = {
    "baseline": RewardSettings(),
    "time": RewardSettings(time_cost=0.03),
    "hover": RewardSettings(hover_bonus=1.0),
    "fixed": RewardSettings(time_cost=0.03, progress_scale=10.0),
}


def potential(obs):
    """Higher is better: closer to pad, slower, and more upright."""
    x, y, vx, vy, angle, spin, *_ = obs
    return -float(np.hypot(x, y) + 0.5 * np.hypot(vx, vy)
                  + 0.5 * abs(angle) + 0.1 * abs(spin))


def hover_eligible(obs):
    """The intentionally exploitable rule: upright, slow, and airborne."""
    _, height, vx, vy, angle, _, left, right = obs
    return bool(height > 0.1 and abs(angle) < 0.2
                and np.hypot(vx, vy) < 0.3 and not left and not right)


class RewardShaping(gym.Wrapper):
    def __init__(self, env, settings):
        super().__init__(env)
        self.settings = settings
        self.previous_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = potential(obs)
        return obs, info

    def step(self, action):
        if self.previous_potential is None:
            raise RuntimeError("Call reset() before step().")
        obs, base, terminated, truncated, info = self.env.step(action)
        cfg = self.settings
        # True terminal states have zero potential. Time limits retain it because
        # SB3 bootstraps their value; truncation is not a terminal MDP state.
        next_potential = 0.0 if terminated else potential(obs)
        progress = cfg.progress_scale * (
            cfg.gamma * next_potential - self.previous_potential
        )
        hover = cfg.hover_bonus * float(hover_eligible(obs) and not terminated)
        bonus = progress + hover - cfg.time_cost
        self.previous_potential = potential(obs)
        info = dict(info, base_reward=float(base), shaping_reward=float(bonus),
                    reward_terms={"progress": progress, "hover": hover,
                                  "time": -cfg.time_cost})
        return obs, float(base + bonus), terminated, truncated, info


class FlightMetrics(gym.Wrapper):
    """Expose episode metrics without modifying rewards or termination."""
    def reset(self, **kwargs):
        self.base_return = self.shaped_return = 0.0
        self.steps = self.main_steps = self.side_steps = self.hover_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = float(info.get("base_reward", reward))
        self.base_return += base
        self.shaped_return += float(reward)
        self.steps += 1
        self.main_steps += int(action == 2)
        self.side_steps += int(action in (1, 3))
        self.hover_steps += int(hover_eligible(obs))
        info = dict(info)
        if terminated or truncated:
            # LunarLander overwrites its last reward with +100 for resting,
            # -100 for a crash/out-of-bounds. A timeout is a separate outcome.
            outcome = ("landed" if terminated and base == 100.0 else
                       "failed" if terminated else "timeout")
            info["flight"] = {
                "base_return": self.base_return,
                "shaped_return": self.shaped_return,
                "steps": self.steps,
                "main_steps": self.main_steps,
                "side_steps": self.side_steps,
                "fuel_proxy": 0.3 * self.main_steps + 0.03 * self.side_steps,
                "hover_fraction": self.hover_steps / self.steps,
                "outcome": outcome,
            }
        return obs, reward, terminated, truncated, info


def make_env(preset="baseline", render_mode=None, gamma=0.99, max_steps=1000,
             scenario="calm"):
    from .scenarios import weather_options
    env = gym.make("LunarLander-v3", render_mode=render_mode,
                   max_episode_steps=max_steps, **weather_options(scenario))
    if preset != "baseline":
        cfg = asdict(PRESETS[preset])
        cfg["gamma"] = gamma
        env = RewardShaping(env, RewardSettings(**cfg))
    return FlightMetrics(env)
