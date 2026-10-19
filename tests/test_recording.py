import gymnasium as gym
import numpy as np
from PIL import Image

from rocket_landing.recording import GifRecorder


class RenderedEpisode(gym.Env):
    render_mode = "rgb_array"
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(8), {}

    def step(self, action):
        self.t += 1
        return np.zeros(8), 0.0, self.t == 4, False, {}

    def render(self):
        return np.full((40, 60, 3), self.t * 40, dtype=np.uint8)


def test_recorder_saves_separate_complete_episodes(tmp_path):
    env = GifRecorder(RenderedEpisode(), tmp_path)
    try:
        for _ in range(2):
            env.reset()
            for _ in range(4):
                env.step(0)
    finally:
        env.close()
    clips = sorted(tmp_path.glob("*.gif"))
    assert len(clips) == 2
    for path in clips:
        with Image.open(path) as clip:
            assert clip.n_frames == 3
            assert clip.size == (400, 267)
            assert clip.info["duration"] == 40
            clip.seek(2)
            assert np.asarray(clip.convert("RGB")).mean() > 100
