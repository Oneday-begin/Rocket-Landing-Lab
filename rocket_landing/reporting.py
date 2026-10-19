"""Build a portable, local experiment report from measured files only."""

import ast
import csv
import html
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import write_csv


def build_report(root="runs", destination="artifacts"):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def video_tag(video):
        relative = Path(os.path.relpath(video, destination)).as_posix()
        if video.suffix == ".gif":
            return f'<img class="clip" src="{quote(relative)}" alt="Recorded rocket flight">'
        return (f'<video controls preload="metadata" src="{quote(relative)}">'
                'Your browser does not support MP4 playback.</video>')
    runs = sorted(root.glob("*/seed_*/evaluation.json"))
    if not runs:
        raise FileNotFoundError(f"No evaluation.json files in {root}; run evaluate or experiment first.")
    results, cards = [], []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for evaluation in runs:
        directory = evaluation.parent
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads(evaluation.read_text())
        if summary.get("scenario", "calm") != "calm":
            raise ValueError(f"Weather-test results cannot replace the original benchmark: {evaluation}")
        evaluated_model = Path(summary["model"])
        if evaluated_model.exists() and summary.get("model_sha256") != hashlib.sha256(
                evaluated_model.read_bytes()).hexdigest():
            raise ValueError(f"Model changed after evaluation: {evaluated_model}. Evaluate it again.")
        label = f"{config['preset']} / seed {config['seed']}"
        results.append({"preset": config["preset"], "seed": config["seed"],
                        **{k: v for k, v in summary.items() if k not in ("model", "reward")}})
        history = directory / "validation.jsonl"
        if history.exists():
            entries = [json.loads(line) for line in history.read_text().splitlines()]
            axes[0].plot([r["steps"] for r in entries],
                         [r["mean_base_return"] for r in entries], label=label)
        flights = []
        for monitor in directory.glob("*.monitor.csv"):
            with monitor.open(encoding="utf-8") as f:
                next(f)
                for row in csv.DictReader(f):
                    if row.get("flight"):
                        flights.append((float(row["t"]), ast.literal_eval(row["flight"])))
        flights.sort(key=lambda row: row[0])
        if len(flights) >= 20:
            shaped = [r[1]["shaped_return"] for r in flights]
            smoothed = np.convolve(shaped, np.ones(20) / 20, mode="valid")
            axes[1].plot(np.arange(20, len(shaped) + 1), smoothed, label=label)
        video_html = ""
        for batch in sorted((directory / "videos").glob("*"), reverse=True):
            metadata = batch / "summary.json"
            if not metadata.exists():
                continue
            recorded = json.loads(metadata.read_text())
            if (recorded.get("training_steps") == summary.get("evaluated_training_steps") and
                    recorded.get("scenario", "calm") == "calm"):
                video_html = "".join(video_tag(v) for v in sorted(batch.glob("*"))
                                     if v.suffix in (".gif", ".mp4"))
                break
        train = directory / "training.json"
        train_info = json.loads(train.read_text()) if train.exists() else {}
        score = summary["mean_base_return"]
        met = "Benchmark met" if summary["benchmark_met"] else "Below 200 benchmark"
        cards.append(f'''<section><h2>{html.escape(label)}</h2>
<p class="score">{score:.1f} <small>original mean score</small></p>
<p>{met} &middot; {summary['episodes']} held-out episodes &middot;
{summary['landed_rate']:.0%} landed &middot; {summary['timeout_rate']:.0%} timed out</p>
<p>Standard deviation: {summary['std_base_return']:.1f}. Average episode:
{summary['mean_steps']:.0f} frames. Engine-use cost proxy: {summary['mean_fuel_proxy']:.1f}.
Airborne upright/slow frames: {summary['mean_hover_fraction']:.0%}.
Evaluated model's training steps: {summary.get('evaluated_training_steps', 0):,}.</p>
<p>Videos below use the original reward and fixed episode seeds. They are examples,
not the complete evaluation.</p>{video_html}</section>''')
    axes[0].axhline(200, linestyle="--", color="#555", label="200 benchmark")
    axes[0].set(title="Validation on original rewards", xlabel="Training frames",
                ylabel="Mean original score")
    axes[1].set(title="Training rewards (includes custom bonuses)", xlabel="Episode",
                ylabel="20-episode rolling average")
    for ax in axes:
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.savefig(destination / "learning-curves.png", dpi=160)
    plt.close(fig)
    write_csv(destination / "comparison.csv", results)
    rows = "".join(f"<tr><td>{html.escape(r['preset'])}</td><td>{r['seed']}</td>"
                   f"<td>{r['mean_base_return']:.1f}</td><td>{r['landed_rate']:.0%}</td>"
                   f"<td>{r['timeout_rate']:.0%}</td><td>{r['mean_steps']:.0f}</td></tr>"
                   for r in results)
    random_videos = "".join(video_tag(v) for v in sorted(
        (destination / "random").rglob("*")) if v.suffix in (".gif", ".mp4"))
    content = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rocket Landing Lab - measured results</title>
<style>body{font:17px/1.6 system-ui;background:#0b1120;color:#e2e8f0;max-width:1100px;
margin:40px auto;padding:0 24px}h1{font-size:42px;line-height:1.2}h2{color:#67e8f9}
section{background:#152034;padding:24px;border-radius:16px;margin:24px 0}
.score{font-size:38px;font-weight:700;margin:8px 0}small{font-size:16px;font-weight:400}
img{width:100%;border-radius:12px}video,.clip{width:min(100%,400px);margin:8px;border-radius:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #334155}
.scroll{overflow-x:auto}a{color:#67e8f9}</style>
<h1>Rocket Landing Lab</h1><p>PPO learns from eight numbers to choose between four engine actions.
These results compare the original reward with three reward-shaping experiments.</p>
<p><b>Read the original score first.</b> A custom bonus can increase training reward while making
landing worse. Each model is tested without custom reward shaping. The same episode seeds
are used across models. One training seed is a demonstration, not evidence of a robust improvement.</p>
<img src="learning-curves.png" alt="Original validation rewards and custom training rewards">
<div class="scroll"><table><thead><tr><th>Reward</th><th>Seed</th><th>Original score</th>
<th>Landed</th><th>Timeout</th><th>Frames</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p>The environment's +100 terminal reward identifies a resting lander. Failures combine crashes
and leaving the bounds. Timeouts are recorded separately. Engine-use cost is a proxy based on
the game's penalties, not kilograms of fuel. The 200-point benchmark is an average-return target,
not an individual-episode landing test.</p>''' + (
        '<section><h2>Before training: random rocket</h2>' + random_videos + '</section>'
        if random_videos else '') + "".join(cards) + '''
<section><h2>What to investigate</h2><p>Does the hover bonus produce more airborne frames or
timeouts? Does the fixed reward still land? Compare the full evaluation CSV and videos;
hovering is a possible learned behavior, not a guaranteed result.</p><p>The fixed reward removes
the hover bonus, adds a small time cost, and uses a discounted potential difference. It is an
experiment, not a promise that it beats the original reward.</p></section></html>'''
    report = destination / "report.html"
    report.write_text(content, encoding="utf-8")
    print(f"Report: {report}", flush=True)
    return report
