import hashlib
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from rocket_landing.catalog import discover_models
from rocket_landing.evaluation import landing_interval, rollout, summarize
from rocket_landing.jobs import JobRunner
from rocket_landing.rewards import make_env


@pytest.mark.parametrize("successes,total,low,high", [
    (0, 20, 0, 0.1611), (20, 20, 0.8389, 1), (94, 100, 0.8752, 0.9722),
])
def test_landing_uncertainty_including_zero_and_perfect_runs(successes, total, low, high):
    interval = landing_interval(successes, total)
    assert interval == pytest.approx([low, high], abs=0.0002)


def test_empty_results_are_rejected():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        landing_interval(0, 0)


@pytest.mark.parametrize("scenario,wind,turbulence", [("calm", 0, 0), ("breezy", 8, 0.5), ("gusty", 15, 1.5)])
def test_scenario_reaches_simulator_and_original_rewards(scenario, wind, turbulence):
    env = make_env(scenario=scenario)
    try:
        assert env.unwrapped.wind_power == wind
        assert env.unwrapped.turbulence_power == turbulence
        assert env.unwrapped.enable_wind == (scenario != "calm")
        env.reset(seed=900)
        assert "base_reward" not in env.step(0)[-1]  # no custom reward wrapper
    finally:
        env.close()


def test_gusty_physics_are_repeatable_and_differ_from_calm():
    a, b, calm = make_env(scenario="gusty"), make_env(scenario="gusty"), make_env()
    try:
        for env in (a, b, calm):
            env.reset(seed=310)
        for _ in range(20):
            first, second, still = a.step(2), b.step(2), calm.step(2)
            np.testing.assert_array_equal(first[0], second[0])
        assert not np.allclose(first[0], still[0])
    finally:
        for env in (a, b, calm):
            env.close()


def test_catalog_never_attaches_final_score_to_best_or_changed_model(tmp_path):
    directory = tmp_path / "runs/baseline/seed_0"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"preset": "baseline", "seed": 0}))
    final, best = directory / "final_model.zip", directory / "best_model.zip"
    final.write_bytes(b"final")
    best.write_bytes(b"best")
    summary = {"model": str(final), "model_sha256": hashlib.sha256(b"final").hexdigest(),
               "mean_base_return": 200, "landed_rate": 0.9, "episodes": 100}
    (directory / "evaluation.json").write_text(json.dumps(summary))
    models = discover_models(tmp_path)
    assert models[0].summary == summary
    assert models[1].summary is None
    final.write_bytes(b"updated model")
    assert discover_models(tmp_path)[0].summary is None


def test_progress_callback_observes_every_completed_episode():
    progress = []
    rows = rollout(episodes=2, seed=1234, scenario="breezy",
                   on_episode=lambda done, total, flight: progress.append((done, total, flight["outcome"])))
    assert len(rows) == 2
    assert [r[:2] for r in progress] == [(1, 2), (2, 2)]
    assert all(r["base_return"] == r["shaped_return"] for r in rows)


def test_owned_job_streams_progress_and_prevents_double_launch():
    root = Path(__file__).resolve().parent.parent
    runner = JobRunner(root, sys.executable)
    try:
        runner.start(["--help"])
        with pytest.raises(RuntimeError):
            runner.start(["--help"])
        deadline = time.monotonic() + 30
        lines = []
        while time.monotonic() < deadline:
            kind, pid, value = runner.events.get(timeout=30)
            if kind == "line":
                lines.append(value)
            else:
                assert value == 0
                runner.acknowledge(pid)
                break
        else:
            pytest.fail("Worker did not finish")
        assert runner.process is None
        assert "robustness" in "".join(lines)
    finally:
        runner.close()
