"""Fine-tune a saved policy on seeded, episode-randomized wind and turbulence."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .telemetry import ROOT, DATA


class RandomWeather(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.weather_rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.weather_rng = np.random.default_rng(seed + 17000)
        # The underlying environment was created with wind enabled, so all wind state exists.
        self.unwrapped.wind_power = float(self.weather_rng.uniform(0, 18))
        self.unwrapped.turbulence_power = float(self.weather_rng.uniform(0, 1.8))
        return self.env.reset(seed=seed, options=options)


def train_broader(steps=200000, seed=1):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from .models import load_ppo
    from .rewards import make_env
    from .evaluation import write_json
    from .robustness import run_suite
    torch.set_num_threads(1)
    source = ROOT / 'runs/fixed/seed_0/final_model.zip'
    directory = ROOT / f'runs/robust/seed_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', {
        'preset': 'robust', 'seed': seed, 'source_model': source.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'weather': {'wind_uniform': [0,18], 'turbulence_uniform': [0,1.8]},
        'reward': 'original only', 'requested_additional_steps': steps,
        'description': 'Fine-tune fixed seed 0 with new episode weather; one training seed, experimental.'})
    env = DummyVecEnv([lambda: Monitor(RandomWeather(make_env(scenario='gusty'))) for _ in range(4)])
    env.seed(seed)
    policy = load_ppo(source, env=env, device='cpu')
    policy.set_random_seed(seed)
    source_steps = policy.num_timesteps

    class Progress(BaseCallback):
        def __init__(self): super().__init__(); self.last=0
        def _on_step(self):
            if time.monotonic()-self.last>25:
                self.last=time.monotonic()
                print(f'Broader training: {self.num_timesteps:,}/{steps:,} new steps',flush=True)
            return True
    start=time.monotonic()
    try:
        policy.learn(total_timesteps=steps, callback=Progress(), reset_num_timesteps=True)
        policy.save(directory/'final_model')
        write_json(directory/'training.json',{'source_training_steps':source_steps,
                   'additional_steps':policy.num_timesteps,'total_steps':source_steps+policy.num_timesteps,
                   'wall_seconds':time.monotonic()-start,'interrupted':False})
    finally: env.close()
    batch=run_suite([source,directory/'final_model.zip'], DATA/'broader-evaluation', episodes=50, seed=700000)
    result={'evaluation':batch.relative_to(ROOT).as_posix(), 'cases':json.loads((batch/'results.json').read_text()),
            'protocol':'50 unseen seed numbers per condition (700000–700049); fixed source vs wind-fine-tuned policy; one training seed.'}
    for case in result['cases']:
        case['model_path']=Path(case['model_path']).relative_to(ROOT).as_posix()
    write_json(DATA/'broader-results.json',result)
    print('Broader training and held-out weather comparison complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200000);p.add_argument('--seed',type=int,default=1)
    a=p.parse_args();train_broader(a.steps,a.seed)
