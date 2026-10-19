"""CPU-friendly PPO training with raw-reward validation and resumable saves."""

import importlib.metadata
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .evaluation import rollout, summarize, write_json
from .models import load_ppo
from .rewards import PRESETS, make_env


class Progress(BaseCallback):
    def __init__(self, directory, interval=50_000, eval_episodes=10):
        super().__init__()
        self.directory = Path(directory)
        self.interval = interval
        self.eval_episodes = eval_episodes
        self.next_eval = interval
        self.next_message = 0
        self.best = -float("inf")
        self.recent = deque(maxlen=100)
        self.started = time.monotonic()

    def _on_training_start(self):
        self.start_steps = self.model.num_timesteps
        self.next_eval = (self.start_steps // self.interval + 1) * self.interval
        history = self.directory / "validation.jsonl"
        if history.exists():
            previous = [json.loads(line) for line in history.read_text().splitlines()]
            if previous:
                self.best = max(row["mean_base_return"] for row in previous)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "flight" in info:
                self.recent.append(info["flight"])
        now = time.monotonic()
        if now >= self.next_message:
            elapsed = max(0.001, now - self.started)
            speed = (self.num_timesteps - self.start_steps) / elapsed
            message = f"steps={self.num_timesteps:,} speed={speed:.0f}/s"
            if self.recent:
                stats = summarize(list(self.recent))
                message += (f" raw_mean={stats['mean_base_return']:.1f}"
                            f" shaped_mean={stats['mean_shaped_return']:.1f}"
                            f" landed={stats['landed_rate']:.0%}")
            print(message, flush=True)
            self.next_message = now + 30
        if self.num_timesteps >= self.next_eval:
            # Fixed validation seeds are distinct from final held-out test seeds.
            stats = summarize(rollout(self.model, self.eval_episodes, seed=50_000))
            stats["steps"] = self.num_timesteps
            with (self.directory / "validation.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            self.model.save(self.directory / "checkpoint")
            if stats["mean_base_return"] > self.best:
                self.best = stats["mean_base_return"]
                self.model.save(self.directory / "best_model")
            print(f"Validation: raw mean {stats['mean_base_return']:.1f}, "
                  f"landed {stats['landed_rate']:.0%}", flush=True)
            self.next_eval += self.interval
        return True


def train(preset="baseline", steps=1_000_000, seed=0, output="runs",
          n_envs=4, threads=1, resume=None, eval_every=50_000,
          eval_episodes=10, learning_rate=3e-4):
    if steps < 1 or n_envs < 1 or threads < 1 or eval_every < 1 or eval_episodes < 1:
        raise ValueError("steps, environment count, threads, and evaluation sizes must be positive")
    torch.set_num_threads(threads)
    directory = Path(output) / preset / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists() and resume is None:
        raise FileExistsError(f"Run exists: {directory}. Use --resume or a different --output/--seed.")
    if resume:
        if not config_path.exists():
            raise ValueError("Resume in the original run directory so metadata stays consistent.")
        previous = json.loads(config_path.read_text())
        if previous["preset"] != preset or previous["seed"] != seed:
            raise ValueError("Resume preset and seed must match the saved run.")
        resume_path = Path(resume).resolve()
        if resume_path.parent != directory.resolve():
            raise ValueError("Resume checkpoint must belong to the destination run directory.")
    factories = [lambda i=i: Monitor(make_env(preset),
                 str(directory / f"env_{i}{'_resume_' + str(time.time_ns()) if resume else ''}.monitor.csv"),
                 info_keywords=("flight",)) for i in range(n_envs)]
    env = DummyVecEnv(factories)
    env.seed(seed)
    started = time.monotonic()
    config = {
        "preset": preset, "seed": seed, "requested_steps": steps,
        "n_envs": n_envs, "threads": threads, "environment": "LunarLander-v3",
        "reward": asdict(PRESETS[preset]), "learning_rate": learning_rate,
        "ppo": {"n_steps": 1024, "batch_size": 64, "n_epochs": 10,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                "policy_layers": [64, 64]},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("gymnasium", "stable-baselines3", "torch", "numpy")},
    }
    if resume:
        # Preserve the initial configuration and append continuation metadata.
        config = previous
        config.setdefault("continuations", []).append({
            "additional_steps": steps, "resume": str(resume), "n_envs": n_envs})
    write_json(config_path, config)
    model = None
    interrupted = False
    try:
        if resume:
            model = load_ppo(resume, env=env, device="cpu")
        else:
            model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                        learning_rate=learning_rate, n_steps=1024, batch_size=64,
                        n_epochs=10, gamma=0.99,
                        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
        print(f"Training {preset}, seed {seed}, {steps:,} additional steps on CPU", flush=True)
        try:
            model.learn(total_timesteps=steps, callback=Progress(
                directory, eval_every, eval_episodes), reset_num_timesteps=not bool(resume))
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saving current model.", flush=True)
        model.save(directory / "final_model")
        write_json(directory / "training.json", {
            "total_steps": model.num_timesteps,
            "wall_seconds_this_session": time.monotonic() - started,
            "interrupted": interrupted,
        })
    finally:
        env.close()
    return directory
