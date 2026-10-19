"""Evaluate a policy across weather conditions without changing its weights."""

import hashlib
import html
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import rollout, summarize, write_csv, write_json
from .models import load_ppo
from .scenarios import SCENARIOS


def run_suite(models, output="artifacts/robustness", episodes=20, seed=200_000,
              scenarios=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    scenarios = list(SCENARIOS) if scenarios is None else list(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Choose at least one distinct weather scenario")
    for name in scenarios:
        if name not in SCENARIOS:
            raise ValueError(f"Unknown weather scenario: {name}")
    model_paths = [Path(path).resolve(strict=True) for path in models]
    if not model_paths or len(set(model_paths)) != len(model_paths):
        raise ValueError("Choose at least one distinct model")
    batch = Path(output).resolve() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    batch.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "episodes_per_case": episodes,
                "first_episode_seed": seed, "scenarios": scenarios,
                "models": [str(path) for path in model_paths], "completed_cases": 0,
                "reward": "original environment reward; shaping disabled"}
    write_json(batch / "manifest.json", manifest)
    comparisons = []
    all_rows = []
    try:
        for index, model_path in enumerate(model_paths):
            model = load_ppo(model_path, device="cpu")
            label = f"{index + 1}. {model_path.parent.parent.name} / {model_path.parent.name} / {model_path.stem}"
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            for scenario in scenarios:
                print(f"Testing {label}: {SCENARIOS[scenario].title}", flush=True)

                def progress(done, total, flight):
                    print(f"  Episode {done}/{total}: {flight['outcome']}, "
                          f"score {flight['base_return']:.1f}", flush=True)

                rows = rollout(model, episodes, seed, scenario=scenario, on_episode=progress)
                summary = summarize(rows)
                case = {"model": label, "model_path": str(model_path),
                        "model_sha256": digest, "scenario": scenario,
                        "weather": asdict(SCENARIOS[scenario]), **summary}
                case["benchmark_applicable"] = scenario == "calm"
                if scenario != "calm":
                    case["benchmark_met"] = None
                comparisons.append(case)
                case_dir = batch / f"model_{index}" / scenario
                write_json(case_dir / "summary.json", case)
                write_csv(case_dir / "episodes.csv", rows)
                all_rows.extend({"model": label, "scenario": scenario, **row} for row in rows)
                manifest["completed_cases"] += 1
                write_json(batch / "manifest.json", manifest)
                print(f"  Mean score {summary['mean_base_return']:.1f}; "
                      f"landed {summary['landed_rate']:.0%}", flush=True)
        write_json(batch / "results.json", comparisons)
        write_csv(batch / "episodes.csv", all_rows)
        flat = [{key: value for key, value in row.items()
                 if key not in ("weather", "landed_ci95")} | {
                     "landing_ci95_low": row["landed_ci95"][0],
                     "landing_ci95_high": row["landed_ci95"][1]}
                for row in comparisons]
        write_csv(batch / "comparison.csv", flat)
        render_suite(comparisons, batch, seed)
        manifest["status"] = "complete"
        write_json(batch / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="incomplete", reason=type(error).__name__)
        write_json(batch / "manifest.json", manifest)
        raise
    print(f"Wind-test report: {batch / 'report.html'}", flush=True)
    return batch


def render_suite(rows, output, seed):
    output = Path(output)
    labels = list(dict.fromkeys(row["model"] for row in rows))
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        selected = [next(row for row in rows if row["model"] == label and
                         row["scenario"] == scenario) for scenario in scenarios]
        positions = x - 0.4 + width / 2 + i * width
        rates = np.array([row["landed_rate"] for row in selected])
        lower = np.array([row["landed_ci95"][0] for row in selected])
        upper = np.array([row["landed_ci95"][1] for row in selected])
        axes[0].bar(positions, rates * 100, width, label=label,
                    yerr=np.maximum(0, np.array([rates - lower, upper - rates])) * 100,
                    capsize=3)
        axes[1].bar(positions, [row["mean_base_return"] for row in selected],
                    width, label=label)
    axes[0].set(title="Landing rate with 95% Wilson intervals", ylabel="Episodes landed (%)",
                ylim=(0, 105))
    axes[1].set(title="Original reward in each weather condition", ylabel="Mean score")
    for ax in axes:
        ax.set_xticks(x, [SCENARIOS[name].title for name in scenarios])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7)
    fig.savefig(output / "weather-comparison.png", dpi=160)
    plt.close(fig)
    table = "".join(
        f"<tr><td>{html.escape(row['model'])}</td><td>{SCENARIOS[row['scenario']].title}</td>"
        f"<td>{row['mean_base_return']:.1f}</td><td>{row['landed_rate']:.0%}</td>"
        f"<td>{row['landed_ci95'][0]:.0%}–{row['landed_ci95'][1]:.0%}</td>"
        f"<td>{row['timeout_rate']:.0%}</td><td>{row['episodes']}</td></tr>"
        for row in rows)
    (output / "report.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rocket weather tests</title>
<style>body{background:#0b1120;color:#e2e8f0;font:17px/1.6 system-ui;max-width:1200px;
margin:40px auto;padding:0 24px}h1{color:#67e8f9}img{width:100%;border-radius:12px}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #334155}
.table{overflow-x:auto}a{color:#67e8f9}</style><h1>How well does the rocket handle wind?</h1>
<p>These are tests of existing policies, with their weights unchanged. All scores use the original
game reward. Calm is the original task; breezy and gusty are additional stress tests. The same
episode seed numbers are used across every model and condition.</p>
<img src="weather-comparison.png" alt="Landing rates and original scores across weather conditions">
<div class="table"><table><thead><tr><th>Model</th><th>Weather</th><th>Mean score</th>
<th>Landed</th><th>95% interval</th><th>Timeout</th><th>Episodes</th></tr></thead><tbody>'''
        + table + '''</tbody></table></div><p>Calm: wind off. Breezy: wind power 8, turbulence 0.5.
Gusty: wind power 15, turbulence 1.5. These are simulator settings, not real-world wind speeds.</p>
<p>The intervals describe uncertainty across episode outcomes for each fixed policy. They do not
measure variation across training seeds. Small tests can have wide intervals; an apparent difference
does not establish that one policy is better.</p>'''
        + f'<p>Episode seeds start at {seed}. Results are saved in '
          '<a href="comparison.csv">comparison.csv</a> and '
          '<a href="episodes.csv">episodes.csv</a>.</p></html>', encoding="utf-8")
