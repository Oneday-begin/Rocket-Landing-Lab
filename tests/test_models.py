import numpy as np
import torch
from stable_baselines3 import PPO

from rocket_landing.models import load_ppo
from rocket_landing.rewards import make_env


def test_saved_policy_loads_with_identical_predictions_and_weights(tmp_path):
    torch.set_num_threads(1)
    env = make_env()
    try:
        model = PPO("MlpPolicy", env, seed=21, device="cpu", n_steps=32, batch_size=32)
        obs, _ = env.reset(seed=200)
        action, _ = model.predict(obs, deterministic=True)
        path = tmp_path / "model.zip"
        model.save(path)
        original_torch_load = torch.load
        loaded = load_ppo(path, device="cpu")
        assert torch.load is original_torch_load
        restored, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action, restored)
        for key, tensor in model.policy.state_dict().items():
            torch.testing.assert_close(tensor, loaded.policy.state_dict()[key], rtol=0, atol=0)
    finally:
        env.close()
