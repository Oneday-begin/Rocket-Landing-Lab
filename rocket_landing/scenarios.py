"""Repeatable weather settings for evaluating already-trained policies."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Weather:
    title: str
    enable_wind: bool
    wind_power: float
    turbulence_power: float

    def env_options(self):
        return {key: value for key, value in asdict(self).items() if key != "title"}


SCENARIOS = {
    "calm": Weather("Calm", False, 0.0, 0.0),
    "breezy": Weather("Breezy", True, 8.0, 0.5),
    "gusty": Weather("Gusty", True, 15.0, 1.5),
}


def weather_options(scenario):
    try:
        return SCENARIOS[scenario].env_options()
    except KeyError:
        raise ValueError(f"Unknown weather scenario: {scenario}") from None
