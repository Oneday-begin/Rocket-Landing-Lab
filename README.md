# Rocket Landing Lab

A local deep reinforcement learning lab that trains PPO to land a rocket, demonstrates reward hacking, and evaluates reliability across weather conditions.

Built with Python, Gymnasium, Stable-Baselines3, PyTorch and Tkinter. Inspired by Hari (@harixganesh)'s Lunar Lander reward-shaping project. Implementation developed with AI assistance.

## Features

- Four saved policies: original reward, time penalty, deliberate hover exploit and corrected progress shaping.
- Desktop controls for watching, recording and evaluating flights.
- Calm, breezy and gusty stress tests with confidence intervals and CSV exports.
- Automated quality gate: sample size, landing reliability, timeouts, original score and model identity.
- Repeatable evaluation seeds, model checksums and 33 automated tests.

## Quick start — Windows

Install Python 3.10 or newer (the saved build used Python 3.12), then open PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m rocket_landing panel
```

After setup, double-click **Open-Control-Panel.cmd**. Select a final model and click **Watch AI**. Four small trained models are included, so retraining is optional. Setup needs internet access; installed inference runs locally on CPU.

The package ranges are in requirements.txt. requirements-lock.txt records the original tested direct versions; it is a reference, not a guarantee of identical behavior on every platform. GIF recording is included. MP4 dependencies are optional in requirements-video.txt.

## Measured results

Original calm evaluation: 100 episodes per policy, seeds 100000–100099, one training seed.

| Policy | Mean original score | Landed |
|---|---:|---:|
| Baseline | 238.8 | 94% |
| Time penalty | 235.9 | 89% |
| Hover reward | -8.2 | 0% |
| Fixed reward | 221.3 | 92% |

The separate quality run tested 600 flights: four policies × three weather conditions × 50 episodes. Only the fixed-reward policy passed every project acceptance check. These are simulation results for particular saved policies, not real-flight certification or a general ranking of algorithms.

Open **View-Results.cmd** for the original report and **View-Quality-Results.cmd** for the quality report. Reports include per-episode data. **Run-Quality-Check.cmd** runs a fresh suite; it may take several minutes. The intentionally faulty hover policy is expected to fail.

## Reproduce and extend

```powershell
# New training seed; existing seed_0 models remain intact
.\.venv\Scripts\python.exe -m rocket_landing train --preset fixed --seed 1 --steps 1000000

# Run an automation-compatible acceptance check (exit 2 on rejection)
.\.venv\Scripts\python.exe -m rocket_landing quality --models runs/fixed/seed_0/final_model.zip --seed 400000 --fail-on-rejection

# Run tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Repository contents

- `rocket_landing/`: application, training and evaluation code.
- `tests/`: behavior and regression tests.
- `runs/`: four final models, settings and original measured evaluations.
- `artifacts/`: selected demonstration reports, media and quality-test evidence.
- [Placement preparation report](docs/Rocket-Landing-Placement-Report.pdf): 29 pages covering libraries, PPO mathematics, rewards, architecture, results, interview questions, exercises and a seven-day revision plan.

Virtual environments, caches, intermediate checkpoints, temporary experiments and machine-specific helper scripts are excluded. New generated outputs are ignored by Git; explicitly add only results you intend to publish. This is the main working project and GitHub source folder. Local environments and generated files stay excluded by .gitignore.
