"""Reproducible acceptance checks for saved simulation policies."""

import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import write_json


DEFAULT_POLICY = {
    "minimum_episodes": 50,
    "minimum_landing_ci_lower": {"calm": 0.70, "breezy": 0.60, "gusty": 0.50},
    "maximum_timeout_rate": 0.20,
    "minimum_calm_score": 200.0,
}


def assess(rows, policy):
    """Fail closed on missing/duplicate cases and insufficient evidence."""
    checks = []

    def check(name, passed, observed, required):
        checks.append(dict(check=name, passed=bool(passed), observed=observed, required=required))

    for scenario, minimum in policy["minimum_landing_ci_lower"].items():
        cases = [row for row in rows if row["scenario"] == scenario]
        check(f"{scenario}: exactly one result", len(cases) == 1, len(cases), 1)
        if len(cases) != 1:
            continue
        row = cases[0]
        check(f"{scenario}: sample size", row["episodes"] >= policy["minimum_episodes"],
              row["episodes"], policy["minimum_episodes"])
        lower = row["landed_ci95"][0]
        check(f"{scenario}: landing confidence lower bound", math.isfinite(lower) and lower >= minimum,
              lower, minimum)
        timeout = row["timeout_rate"]
        check(f"{scenario}: timeout rate", math.isfinite(timeout) and 0 <= timeout <= policy["maximum_timeout_rate"],
              timeout, policy["maximum_timeout_rate"])
        if scenario == "calm":
            score = row["mean_base_return"]
            check("calm: original score", math.isfinite(score) and score >= policy["minimum_calm_score"],
                  score, policy["minimum_calm_score"])
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def audit(batch, policy=DEFAULT_POLICY):
    batch = Path(batch)
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((batch / "results.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or not rows or manifest["completed_cases"] != len(rows):
        raise ValueError("A complete weather evaluation is required")
    models = []
    for path in manifest["models"]:
        selected = [row for row in rows if row["model_path"] == path]
        result = assess(selected, policy)
        model_file = Path(path)
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest() if model_file.is_file() else None
        integrity = bool(selected) and digest is not None and all(row["model_sha256"] == digest for row in selected)
        result["checks"].append(dict(check="Saved weights match evaluated weights", passed=integrity,
                                     observed=digest, required="matching SHA-256"))
        result["passed"] = result["passed"] and integrity
        models.append(dict(model=selected[0]["model"] if selected else path, model_sha256=digest, **result))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
              "first_episode_seed": manifest["first_episode_seed"], "models": models,
              "passed": bool(models) and all(model["passed"] for model in models),
              "scope": "Project simulation acceptance criteria, not a real-flight certification"}
    write_json(batch / "quality-gate.json", result)
    cards = []
    for model in models:
        details = "".join(f"<tr><td>{html.escape(c['check'])}</td><td>{'PASS' if c['passed'] else 'FAIL'}</td>"
                          f"<td>{html.escape(str(c['observed']))}</td><td>{html.escape(str(c['required']))}</td></tr>"
                          for c in model["checks"])
        cards.append(f"<h2>{html.escape(model['model'])}: {'PASS' if model['passed'] else 'FAIL'}</h2>"
                     f"<table><tr><th>Check</th><th>Result</th><th>Observed</th><th>Required threshold</th></tr>{details}</table>")
    (batch / "quality-gate.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Model quality gate</title>
<style>body{font:16px/1.6 system-ui;background:#0b1120;color:#e2e8f0;max-width:1200px;margin:40px auto;padding:24px}
h1,h2,a{color:#67e8f9}table{border-collapse:collapse;width:100%;margin-bottom:32px}td,th{padding:10px;text-align:left;border-bottom:1px solid #334155;overflow-wrap:anywhere}td:nth-child(3){max-width:220px}</style>
<h1>Model quality gate</h1><p>Each model must pass every check in calm, breezy and gusty weather.
Landing checks use the lower end of the 95% Wilson confidence interval, rather than the success rate alone.
These are explicit project acceptance criteria, not industry standards or real-flight certification.</p>
<p>Results cover one trained policy across simulation episodes. They do not measure training-seed variability.
The intentionally hovering model is expected to fail. Models are never deployed or replaced automatically.</p>
<p><a href="report.html">Weather charts and episode data</a> · <a href="quality-gate.json">Machine-readable checks</a></p>'''
        + "".join(cards) + "</html>", encoding="utf-8")
    for model in models:
        print(f"{'PASS' if model['passed'] else 'FAIL'}: {model['model']}", flush=True)
    print(f"Quality report: {batch / 'quality-gate.html'}", flush=True)
    return result


def run_gate(models, output, episodes=50, seed=300_000):
    from .robustness import run_suite
    # Persist the criteria before measuring any flights.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "acceptance-policy.json", DEFAULT_POLICY)
    batch = run_suite(models, output, episodes, seed)
    return audit(batch)
