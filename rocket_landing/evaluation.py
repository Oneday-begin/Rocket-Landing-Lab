"""Unshaped evaluation, paired episode seeds, recordings, and raw results."""

import csv
import hashlib
import json
import math
import html
from pathlib import Path

import numpy as np

from .rewards import make_env


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rollout(model=None, episodes=10, seed=100_000, preset="baseline",
            video_dir=None, human=False, print_observation=False, video_format="gif",
            scenario="calm", on_episode=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = make_env(preset, "human" if human else "rgb_array" if video_dir else None,
                   scenario=scenario)
    if video_dir:
        if video_format == "mp4":
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(env, str(Path(video_dir).resolve()),
                              episode_trigger=lambda episode: True,
                              name_prefix="flight", disable_logger=True)
        else:
            from .recording import GifRecorder
            env = GifRecorder(env, video_dir)
    rows = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            if print_observation:
                print("[x, y, vx, vy, angle, angular velocity, left leg, right leg]")
                print(obs.tolist(), flush=True)
            while True:
                action = (int(env.action_space.sample()) if model is None else
                          int(model.predict(obs, deterministic=True)[0]))
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    rows.append({"episode": episode, "seed": episode_seed,
                                 **info["flight"]})
                    if on_episode:
                        on_episode(episode + 1, episodes, rows[-1])
                    break
    finally:
        env.close()
    return rows


def landing_interval(successes, total):
    """95% Wilson interval for episode success probability, including 0/n and n/n."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Need a positive episode count and a valid success count")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize zero episodes")
    returns = np.array([r["base_return"] for r in rows], dtype=float)
    n = len(rows)
    return {
        "episodes": n,
        "mean_base_return": float(returns.mean()),
        "std_base_return": float(returns.std(ddof=1)) if n > 1 else 0.0,
        "min_base_return": float(returns.min()),
        "max_base_return": float(returns.max()),
        "landed_rate": sum(r["outcome"] == "landed" for r in rows) / n,
        "landed_ci95": landing_interval(sum(r["outcome"] == "landed" for r in rows), n),
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_fuel_proxy": float(np.mean([r["fuel_proxy"] for r in rows])),
        "mean_hover_fraction": float(np.mean([r["hover_fraction"] for r in rows])),
        "mean_shaped_return": float(np.mean([r["shaped_return"] for r in rows])),
        "benchmark_met": bool(returns.mean() >= 200.0),
    }


def evaluate_model(model_path, output, episodes=100, seed=100_000, scenario="calm"):
    from .models import load_ppo
    model_path = Path(model_path)
    if not model_path.exists() and model_path.with_suffix(".zip").exists():
        model_path = model_path.with_suffix(".zip")
    model = load_ppo(model_path, device="cpu")
    # No custom reward wrapper is used for this benchmark.
    def progress(done, total, flight):
        print(f"Episode {done}/{total}: {flight['outcome']}, "
              f"original score {flight['base_return']:.1f}", flush=True)

    rows = rollout(model, episodes, seed, preset="baseline", scenario=scenario,
                   on_episode=progress)
    summary = summarize(rows)
    summary.update(model=str(Path(model_path).resolve()), evaluation_seed=seed,
                   evaluated_training_steps=model.num_timesteps,
                   model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                   scenario=scenario,
                   benchmark_applicable=scenario == "calm",
                   reward="original Gymnasium reward; custom shaping disabled")
    if scenario != "calm":
        summary["benchmark_met"] = None
    write_csv(Path(output) / "evaluation.csv", rows)
    write_json(Path(output) / "evaluation.json", summary)
    render_evaluation(summary, output)
    return summary


def render_evaluation(summary, output):
    title = html.escape(Path(summary["model"]).parent.parent.name.title())
    interval = summary["landed_ci95"]
    weather = html.escape(summary["scenario"].title())
    benchmark = ("200-point mean benchmark: " + ("met" if summary["benchmark_met"] else "not met")
                 if summary["benchmark_applicable"] else
                 "Weather stress test: the original calm-weather benchmark is not applied here.")
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rocket evaluation</title>
<style>body{{font:18px/1.7 system-ui;background:#0b1120;color:#e2e8f0;max-width:850px;
margin:60px auto;padding:24px}}h1,a{{color:#67e8f9}}strong{{font-size:28px}}</style>
<h1>{title}: {weather} evaluation</h1>
<p><strong>{summary['mean_base_return']:.1f}</strong> average original score<br>
<strong>{summary['landed_rate']:.0%}</strong> landed in {summary['episodes']} episodes</p>
<p>95% landing-rate interval: {interval[0]:.0%}–{interval[1]:.0%}.<br>
Timeouts: {summary['timeout_rate']:.0%}. Score standard deviation: {summary['std_base_return']:.1f}.</p>
<p>{benchmark}</p><p>Custom reward shaping was disabled. Episode seeds start at
{summary['evaluation_seed']}. The interval describes uncertainty across episode outcomes
for this one policy, not variation across training seeds.</p>
<p><a href="evaluation.csv">Download every episode as CSV</a></p></html>'''
    (Path(output) / "report.html").write_text(content, encoding="utf-8")
