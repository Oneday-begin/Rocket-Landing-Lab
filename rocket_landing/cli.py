"""Run `python -m rocket_landing --help` for all commands."""

import argparse
from pathlib import Path

from .evaluation import evaluate_model, rollout, summarize, write_csv, write_json
from .rewards import PRESETS
from .scenarios import SCENARIOS


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    p = argparse.ArgumentParser(description="Train a rocket and investigate reward shaping.")
    sub = p.add_subparsers(dest="command", required=True)
    random = sub.add_parser("random", help="Watch or record an untrained rocket")
    random.add_argument("--episodes", type=positive, default=3)
    random.add_argument("--seed", type=int, default=100_000)
    random.add_argument("--human", action="store_true", help="Open the game window")
    random.add_argument("--output", default="artifacts/random")
    random.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    random.add_argument("--scenario", choices=SCENARIOS, default="calm")
    train = sub.add_parser("train", help="Train one reward setup")
    train.add_argument("--preset", choices=PRESETS, default="baseline")
    training_arguments(train)
    train.add_argument("--resume", help="Path to a checkpoint in this same run directory")
    train.add_argument("--eval-every", type=positive, default=50_000)
    train.add_argument("--eval-episodes", type=positive, default=10)
    evaluate = sub.add_parser("evaluate", help="Test using original rewards only")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=positive, default=100)
    evaluate.add_argument("--seed", type=int, default=100_000)
    evaluate.add_argument("--output", help="Default: model's directory")
    evaluate.add_argument("--scenario", choices=SCENARIOS, default="calm")
    watch = sub.add_parser("watch", help="Watch or record a saved model")
    watch.add_argument("--model", required=True)
    watch.add_argument("--episodes", type=positive, default=3)
    watch.add_argument("--seed", type=int, default=100_000)
    watch.add_argument("--human", action="store_true")
    watch.add_argument("--output", help="Default: model's directory/videos")
    watch.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    watch.add_argument("--scenario", choices=SCENARIOS, default="calm")
    experiment = sub.add_parser("experiment", help="Train, test, and record all four setups")
    training_arguments(experiment)
    experiment.add_argument("--presets", choices=PRESETS, nargs="+", default=list(PRESETS))
    experiment.add_argument("--episodes", type=positive, default=100)
    experiment.add_argument("--video-episodes", type=positive, default=3)
    experiment.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    experiment.add_argument("--selection", choices=("final", "best"), default="final",
                            help="Use final models (article's experiment) or best raw-validation models")
    experiment.add_argument("--report-output", default="artifacts")
    experiment.add_argument("--skip-existing", action="store_true",
                            help="Reuse completed matching runs; interrupted runs require --resume")
    report = sub.add_parser("report", help="Rebuild a report from saved evaluations")
    report.add_argument("--runs", default="runs")
    report.add_argument("--output", default="artifacts")
    wind = sub.add_parser("robustness", help="Compare saved policies in calm, breezy, and gusty weather")
    wind.add_argument("--models", nargs="+", help="Default: all final models in runs/")
    wind.add_argument("--episodes", type=positive, default=20, help="Episodes per model per weather condition")
    wind.add_argument("--seed", type=int, default=200_000)
    wind.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    wind.add_argument("--output", default="artifacts/robustness")
    gate = sub.add_parser("quality", help="Run automated model acceptance checks")
    gate.add_argument("--models", nargs="+", help="Default: all final models")
    gate.add_argument("--episodes", type=positive, default=50)
    gate.add_argument("--seed", type=int, default=300_000)
    gate.add_argument("--output", default="artifacts/quality")
    gate.add_argument("--fail-on-rejection", action="store_true", help="Exit 2 if any model fails; useful in CI")
    sub.add_parser("panel", help="Open the desktop control panel")
    return p


def training_arguments(p):
    p.add_argument("--steps", type=positive, default=1_000_000,
                   help="Total additional environment frames, not frames per environment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="runs")
    p.add_argument("--n-envs", type=positive, default=4)
    p.add_argument("--threads", type=positive, default=1)


def show(summary, scenario=None):
    if scenario is not None:
        summary = {**summary, "benchmark_applicable": scenario == "calm"}
    print(f"Original score: {summary['mean_base_return']:.1f} "
          f"+/- {summary['std_base_return']:.1f} (standard deviation)")
    print(f"Landed: {summary['landed_rate']:.0%}; timeouts: {summary['timeout_rate']:.0%}")
    if "landed_ci95" in summary:
        low, high = summary["landed_ci95"]
        print(f"95% landing-rate interval: {low:.0%} to {high:.0%}")
    if summary.get("benchmark_applicable", True):
        print(f"200-point mean benchmark: {'MET' if summary['benchmark_met'] else 'NOT MET'}")
    else:
        print("Weather stress test; original calm-weather benchmark not applied.")


def record(model, output, episodes, seed, video_format="gif", scenario="calm"):
    output = Path(output)
    # Keep each recording batch distinct, so a repeat never overwrites clips.
    from datetime import datetime
    batch = output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    rows = rollout(model, episodes, seed, video_dir=batch, video_format=video_format,
                   scenario=scenario)
    write_csv(batch / "episodes.csv", rows)
    stats = summarize(rows)
    write_json(batch / "summary.json", {
        **stats, "training_steps": model.num_timesteps if model is not None else 0,
        "first_episode_seed": seed, "reward": "original environment reward",
        "scenario": scenario,
        "benchmark_applicable": scenario == "calm",
        "benchmark_met": stats["benchmark_met"] if scenario == "calm" else None,
    })
    print(f"Videos: {batch.resolve()}")
    return batch


def main(argv=None):
    args = parser().parse_args(argv)
    # Small policies are generally faster on CPU with one compute thread.
    import torch
    torch.set_num_threads(getattr(args, "threads", 1))
    if args.command == "random":
        if args.human:
            show(summarize(rollout(episodes=args.episodes, seed=args.seed,
                                  human=True, print_observation=True, scenario=args.scenario)), args.scenario)
        else:
            record(None, args.output, args.episodes, args.seed, args.video_format, args.scenario)
    elif args.command == "train":
        from .training import train
        directory = train(args.preset, args.steps, args.seed, args.output,
                          args.n_envs, args.threads, args.resume,
                          args.eval_every, args.eval_episodes)
        print(f"Saved model: {(directory / 'final_model.zip').resolve()}")
    elif args.command == "evaluate":
        from datetime import datetime
        output = args.output or (Path(args.model).parent if args.scenario == "calm" else
            Path("artifacts/evaluations") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        show(evaluate_model(args.model, output, args.episodes, args.seed, args.scenario))
    elif args.command == "watch":
        from .models import load_ppo
        model = load_ppo(args.model, device="cpu")
        if args.human:
            show(summarize(rollout(model, args.episodes, args.seed, human=True,
                                   scenario=args.scenario)), args.scenario)
        else:
            record(model, args.output or Path(args.model).parent / "videos",
                   args.episodes, args.seed, args.video_format, args.scenario)
    elif args.command == "experiment":
        import json
        from .models import load_ppo
        from .training import train
        from .reporting import build_report
        record(None, Path(args.report_output) / "random", 2, 100_000, args.video_format)
        for preset in args.presets:
            directory = Path(args.output) / preset / f"seed_{args.seed}"
            training_file = directory / "training.json"
            if args.skip_existing and training_file.exists():
                saved = json.loads(training_file.read_text())
                if saved["interrupted"] or saved["total_steps"] < args.steps:
                    raise ValueError(f"Incomplete run {directory}; continue with train --resume first.")
                print(f"Reusing completed {directory}", flush=True)
            else:
                directory = train(preset, args.steps, args.seed, args.output,
                                  args.n_envs, args.threads)
            state = json.loads((directory / "training.json").read_text())
            if state["interrupted"]:
                print("Experiment stopped after saving the interrupted run.")
                return
            # Final models reveal what each reward actually taught. Selecting by
            # raw validation is optional; it can hide a late-learned hover exploit.
            model_path = directory / f"{'best' if args.selection == 'best' else 'final'}_model.zip"
            if not model_path.exists():
                model_path = directory / "final_model.zip"
            show(evaluate_model(model_path, directory, args.episodes, 100_000))
            record(load_ppo(model_path, device="cpu"), directory / "videos",
                   args.video_episodes, 100_000, args.video_format)
            build_report(args.output, args.report_output)
    elif args.command == "report":
        from .reporting import build_report
        build_report(args.runs, args.output)
    elif args.command == "robustness":
        from .robustness import run_suite
        models = args.models or sorted(Path("runs").glob("*/seed_*/final_model.zip"))
        run_suite(models, args.output, args.episodes, args.seed, args.scenarios)
    elif args.command == "quality":
        from .quality import run_gate
        models = args.models or sorted(Path("runs").glob("*/seed_*/final_model.zip"))
        result = run_gate(models, args.output, args.episodes, args.seed)
        if args.fail_on_rejection and not result["passed"]:
            raise SystemExit(2)
    elif args.command == "panel":
        from .panel import main as launch_panel
        launch_panel()


if __name__ == "__main__":
    main()
