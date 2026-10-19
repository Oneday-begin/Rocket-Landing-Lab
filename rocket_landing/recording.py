"""GIF recordings need only Pillow; MP4 is available as an optional extra."""

from pathlib import Path

import gymnasium as gym
from PIL import Image


class GifRecorder(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.episode = -1
        self.step_number = 0

    def _capture(self):
        frame = self.env.render()
        # Half the 50 Hz frames, smaller images, and a palette keep long flights
        # manageable while preserving the actual trajectory and playback speed.
        image = Image.fromarray(frame).resize((400, 267))
        self.frames.append(image.quantize(colors=128, method=Image.Quantize.FASTOCTREE))

    def _save(self):
        if self.frames:
            self.frames[0].save(self.directory / f"flight-episode-{self.episode}.gif",
                                save_all=True, append_images=self.frames[1:],
                                duration=40, loop=0, optimize=False)
            self.frames.clear()

    def reset(self, **kwargs):
        self._save()
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.step_number = 0
        self._capture()
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.step_number += 1
        if self.step_number % 2 == 0 or result[2] or result[3]:
            self._capture()
        if result[2] or result[3]:
            self._save()
        return result

    def close(self):
        try:
            self._save()
        finally:
            self.env.close()
