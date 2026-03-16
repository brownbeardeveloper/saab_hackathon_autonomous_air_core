#!/usr/bin/env python3
"""Run profile benchmarks and optional final training from a shared YAML config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from discord_notify import DiscordNotifier, resolve_discord_webhook_url


LOWER_IS_BETTER_TOKENS = (
    "missed",
    "invalid",
    "loss",
    "std",
    "error",
    "penalty",
)


def format_compact(value: object) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.2e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark training profiles and optionally launch a final run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Scenario config path.",
    )
    parser.add_argument(
        "--missions-file",
        type=Path,
        default=Path("generated_missions_100.json"),
        help="Fixed mission manifest for every episode.",
    )
    parser.add_argument(
        "--profiles-config",
        type=Path,
        default=Path("training_profiles.yml"),
        help="YAML file containing benchmark/final settings and profile definitions.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Python interpreter to use for child training processes. "
            "Defaults to the active interpreter."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where sweep summaries are written.",
    )
    parser.add_argument(
        "--discord-webhook-url",
        default=None,
        help="Optional Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL.",
    )
    return parser.parse_args()


def load_profiles_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    if not profiles:
        raise SystemExit(f"No profiles found in {path}")
    return data


def get_profiles_to_run(data: dict) -> list[str]:
    profiles = data["profiles"]
    requested = data.get("benchmark", {}).get("profiles_to_run")
    if not requested:
        return list(profiles)

    missing = [profile for profile in requested if profile not in profiles]
    if missing:
        raise SystemExit(
            "Unknown profiles in benchmark.profiles_to_run: "
            + ", ".join(sorted(missing))
        )
    return list(requested)


def get_stage_seeds(stage_settings: dict) -> list[int]:
    if "seeds" in stage_settings:
        seeds = stage_settings["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise SystemExit("benchmark.seeds must be a non-empty list.")
        return [int(seed) for seed in seeds]
    if "seed" in stage_settings:
        return [int(stage_settings["seed"])]
    return [42]


def run_training(
    python_bin: Path,
    config_path: Path,
    missions_file: Path,
    profiles_config: Path,
    profile_name: str,
    stage_settings: dict,
    seed: int,
    run_name: str,
    discord_webhook_url: str | None = None,
    discord_progress_interval_percent: int = 0,
    discord_notify_completion: bool = False,
) -> Path:
    cmd = [
        str(python_bin),
        "train.py",
        "--config",
        str(config_path),
        "--missions-file",
        str(missions_file),
        "--profiles-config",
        str(profiles_config),
        "--profile",
        profile_name,
        "--total-timesteps",
        str(stage_settings["total_timesteps"]),
        "--eval-freq",
        str(stage_settings["eval_freq"]),
        "--n-eval-episodes",
        str(stage_settings["n_eval_episodes"]),
        "--seed",
        str(seed),
        "--device",
        str(stage_settings.get("device", "auto")),
        "--verbose",
        str(stage_settings.get("verbose", 1)),
        "--run-name",
        run_name,
    ]

    if "max_episode_hours" in stage_settings:
        cmd.extend(
            [
                "--max-episode-hours",
                str(stage_settings["max_episode_hours"]),
            ]
        )
    if "save_best_model" in stage_settings:
        cmd.append(
            "--save-best-model"
            if stage_settings["save_best_model"]
            else "--no-save-best-model"
        )
    if stage_settings.get("save_replay_metrics", False):
        cmd.append("--save-replay-metrics")
    if discord_webhook_url:
        cmd.extend(["--discord-webhook-url", discord_webhook_url])
    if discord_progress_interval_percent > 0:
        cmd.extend(
            [
                "--discord-progress-interval-percent",
                str(discord_progress_interval_percent),
            ]
        )
    if discord_notify_completion:
        cmd.append("--discord-notify-completion")

    subprocess.run(cmd, check=True)
    save_dir = load_profiles_config(profiles_config).get("logging", {}).get("save_dir", "artifacts")
    return Path(save_dir) / run_name / "training_summary.json"


def read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def aggregate_profile_results(profile_name: str, seed_results: list[dict]) -> dict:
    if not seed_results:
        raise SystemExit(f"No benchmark seed results found for profile '{profile_name}'.")

    first = seed_results[0]
    aggregate = {
        "profile_name": profile_name,
        "num_seeds": len(seed_results),
        "seeds": [result["seed"] for result in seed_results],
        "seed_results": seed_results,
        "learning_rate": first["learning_rate"],
        "n_steps": first["n_steps"],
        "batch_size": first["batch_size"],
        "n_epochs": first["n_epochs"],
        "gamma": first["gamma"],
        "gae_lambda": first["gae_lambda"],
        "ent_coef": first["ent_coef"],
        "clip_range": first["clip_range"],
        "vf_coef": first["vf_coef"],
        "max_grad_norm": first["max_grad_norm"],
        "target_kl": first["target_kl"],
        "hidden_size": first["hidden_size"],
        "hidden_layers": first["hidden_layers"],
        "policy": first["policy"],
        "device": first["device"],
        "max_episode_hours": first["max_episode_hours"],
    }

    metrics_to_aggregate = (
        "eval_mean_reward",
        "eval_std_reward",
        "episode_length",
        "missions_completed",
        "missions_missed",
        "fuel_used",
        "fuel_left",
        "maintenance_events",
        "transfers",
        "invalid_action_rate",
        "completion_rate",
        "truncation_rate",
    )

    for metric_name in metrics_to_aggregate:
        values = np.array(
            [float(result[metric_name]) for result in seed_results],
            dtype=np.float64,
        )
        aggregate[metric_name] = float(values.mean())
        aggregate[f"{metric_name}_seed_std"] = float(values.std(ddof=0))

    return aggregate


def metric_score(metric_name: str, value: float) -> float:
    lower_is_better = any(token in metric_name for token in LOWER_IS_BETTER_TOKENS)
    return -value if lower_is_better else value


def rank_results(results: list[dict], model_selection: dict) -> list[dict]:
    primary_metric = model_selection.get("metric", "eval_mean_reward")
    tie_breakers = model_selection.get(
        "tie_breakers",
        ["missions_completed", "missions_missed", "invalid_action_rate"],
    )

    for metric_name in [primary_metric, *tie_breakers]:
        missing = [result["profile_name"] for result in results if metric_name not in result]
        if missing:
            raise SystemExit(
                f"Metric '{metric_name}' missing from benchmark results for profiles: "
                + ", ".join(sorted(missing))
            )

    return sorted(
        results,
        key=lambda result: tuple(
            metric_score(metric_name, float(result[metric_name]))
            for metric_name in [primary_metric, *tie_breakers]
        ),
        reverse=True,
    )


def write_markdown_summary(summary_dir: Path, payload: dict) -> Path:
    lines = [
        "# Hyperparameter Sweep Summary",
        "",
        f"- Sweep id: {payload['sweep_id']}",
        f"- Missions file: `{payload['missions_file']}`",
        f"- Profiles config: `{payload['profiles_config']}`",
        f"- Benchmark enabled: `{payload['benchmark_enabled']}`",
        f"- Final training enabled: `{payload['final_training_enabled']}`",
    ]

    if payload["benchmark_enabled"]:
        lines.extend(
            [
                f"- Benchmark seeds: `{payload['benchmark_seeds']}`",
                f"- Profiles screened: `{payload['profiles_to_run']}`",
                f"- Best benchmark profile: `{payload.get('best_profile')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Benchmark settings",
            "",
        ]
    )

    for key, value in payload.get("benchmark_settings", {}).items():
        lines.append(f"- {key}: `{format_compact(value)}`")

    lines.extend(
        [
            "",
            "## Model selection",
            "",
        ]
    )
    for key, value in payload.get("model_selection", {}).items():
        lines.append(f"- {key}: `{value}`")

    if payload.get("ranked_results"):
        lines.extend(
            [
                "",
                "## Ranked results",
                "",
                "| Rank | Profile | Reward | Reward seed std | Missions done | Missions missed | Invalid rate | Network | LR | Steps | Batch | Seeds |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )

        for idx, item in enumerate(payload["ranked_results"], start=1):
            network = f"{item['hidden_layers']} x {item['hidden_size']}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        item["profile_name"],
                        format_compact(item["eval_mean_reward"]),
                        format_compact(item["eval_mean_reward_seed_std"]),
                        format_compact(item["missions_completed"]),
                        format_compact(item["missions_missed"]),
                        format_compact(item["invalid_action_rate"]),
                        network,
                        format_compact(item["learning_rate"]),
                        format_compact(item["n_steps"]),
                        format_compact(item["batch_size"]),
                        str(item["num_seeds"]),
                    ]
                )
                + " |"
            )

    if "final_run" in payload:
        final_run = payload["final_run"]
        lines.extend(
            [
                "",
                "## Final run",
                "",
                f"- Profile: `{final_run['profile_name']}`",
                f"- Selection source: `{payload.get('final_profile_source')}`",
                f"- Reward: `{format_compact(final_run['eval_mean_reward'])}` +/- `{format_compact(final_run['eval_std_reward'])}`",
                f"- Missions completed: `{format_compact(final_run['missions_completed'])}`",
                f"- Missions missed: `{format_compact(final_run['missions_missed'])}`",
                f"- Model: `{final_run['final_model_path']}`",
            ]
        )

    markdown_path = summary_dir / "sweep_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def send_profile_completion_notification(
    notifier: DiscordNotifier,
    sweep_id: str,
    aggregate: dict,
) -> None:
    if not notifier.enabled:
        return

    notifier.send(
        "\n".join(
            [
                "**Benchmark profile complete**",
                f"- Sweep: `{sweep_id}`",
                f"- Profile: `{aggregate['profile_name']}`",
                f"- Seeds: `{aggregate['num_seeds']}`",
                f"- Reward: `{aggregate['eval_mean_reward']:.3f}` +/- `{aggregate['eval_mean_reward_seed_std']:.3f}`",
                f"- Missions completed: `{aggregate['missions_completed']:.2f}`",
                f"- Missions missed: `{aggregate['missions_missed']:.2f}`",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    notifier = DiscordNotifier(resolve_discord_webhook_url(args.discord_webhook_url))
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.missions_file.exists():
        raise SystemExit(f"Missions file not found: {args.missions_file}")
    if not args.profiles_config.exists():
        raise SystemExit(f"Profiles config not found: {args.profiles_config}")
    if not args.python_bin.exists():
        raise SystemExit(f"Python interpreter not found: {args.python_bin}")

    data = load_profiles_config(args.profiles_config)
    benchmark = data.get("benchmark", {})
    final_training = data.get("final_training", {})
    model_selection = data.get("model_selection", {})
    profiles = data["profiles"]
    profiles_to_run = get_profiles_to_run(data)

    benchmark_enabled = benchmark.get("enabled", True)
    final_training_enabled = final_training.get("enabled", False)
    benchmark_seeds = get_stage_seeds(benchmark) if benchmark_enabled else []
    benchmark_run_total = len(profiles_to_run) * len(benchmark_seeds) if benchmark_enabled else 0
    final_run_total = 1 if final_training_enabled else 0
    total_scheduled_runs = benchmark_run_total + final_run_total
    benchmark_step_total = benchmark_run_total * int(benchmark.get("total_timesteps", 0))
    final_step_total = int(final_training.get("total_timesteps", 0)) if final_training_enabled else 0
    total_scheduled_steps = benchmark_step_total + final_step_total

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = args.output_dir / f"hackathon_sweep_{sweep_id}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "sweep_summary.json"

    summary_payload = {
        "sweep_id": sweep_id,
        "missions_file": str(args.missions_file),
        "profiles_config": str(args.profiles_config),
        "benchmark_enabled": benchmark_enabled,
        "final_training_enabled": final_training_enabled,
        "benchmark_settings": benchmark,
        "final_training_settings": final_training,
        "model_selection": model_selection,
        "profiles_to_run": profiles_to_run,
        "benchmark_seeds": benchmark_seeds,
        "profiles": profiles,
        "benchmark_seed_runs": [],
        "ranked_results": [],
    }

    ranked = []
    best_profile = None
    completed_runs = 0

    if benchmark_enabled:
        print(f"Sweep id: {sweep_id}")
        print(f"Profiles: {', '.join(profiles_to_run)}")
        print(f"Seeds: {benchmark_seeds}")
        print(f"Using missions: {args.missions_file}")
        print(
            f"Scheduled runs: {total_scheduled_runs} total "
            f"({benchmark_run_total} benchmark + {final_run_total} final)"
        )
        print(
            f"Scheduled env steps: {total_scheduled_steps:,} total "
            f"({benchmark_step_total:,} benchmark + {final_step_total:,} final)"
        )

        benchmark_results = []
        for profile_name in profiles_to_run:
            seed_results = []
            for seed in benchmark_seeds:
                run_name = f"hackathon_sweep_{sweep_id}_{profile_name}_seed{seed}"
                current_run = completed_runs + 1
                print()
                print(
                    f"[benchmark {current_run}/{total_scheduled_runs}] "
                    f"Running profile '{profile_name}' with seed {seed} -> {run_name}"
                )
                summary_file = run_training(
                    python_bin=args.python_bin,
                    config_path=args.config,
                    missions_file=args.missions_file,
                    profiles_config=args.profiles_config,
                    profile_name=profile_name,
                    stage_settings=benchmark,
                    seed=seed,
                    run_name=run_name,
                    discord_webhook_url=args.discord_webhook_url,
                )
                summary = read_summary(summary_file)
                seed_results.append(summary)
                summary_payload["benchmark_seed_runs"].append(summary)
                completed_runs += 1
                print(
                    f"[benchmark] {profile_name} seed {seed}: reward "
                    f"{summary['eval_mean_reward']:.3f} +/- {summary['eval_std_reward']:.3f}, "
                    f"missions {summary['missions_completed']:.2f}/{summary['missions_completed'] + summary['missions_missed']:.2f}"
                )
                print(
                    f"[sweep progress] completed {completed_runs}/{total_scheduled_runs} runs; "
                    f"{total_scheduled_runs - completed_runs} remaining"
                )

            aggregate = aggregate_profile_results(profile_name, seed_results)
            benchmark_results.append(aggregate)
            print(
                f"[benchmark] aggregate {profile_name}: reward {aggregate['eval_mean_reward']:.3f} "
                f"(seed std {aggregate['eval_mean_reward_seed_std']:.3f})"
            )
            send_profile_completion_notification(
                notifier=notifier,
                sweep_id=sweep_id,
                aggregate=aggregate,
            )

        ranked = rank_results(benchmark_results, model_selection)
        best_profile = ranked[0]["profile_name"] if ranked else None
        summary_payload["ranked_results"] = ranked
        summary_payload["best_profile"] = best_profile

        print()
        print("Benchmark ranking:")
        for idx, item in enumerate(ranked, start=1):
            print(
                f"{idx}. {item['profile_name']}: reward {item['eval_mean_reward']:.3f} "
                f"(seed std {item['eval_mean_reward_seed_std']:.3f}), "
                f"missions {item['missions_completed']:.2f}, missed {item['missions_missed']:.2f}"
            )

    if final_training_enabled:
        selected_profile = final_training.get("selected_profile")
        if selected_profile in (None, "", "best"):
            if best_profile is None:
                raise SystemExit(
                    "final_training.selected_profile is not set and no benchmark winner is available."
                )
            final_profile = best_profile
            final_profile_source = "benchmark_best"
        else:
            if selected_profile not in profiles:
                raise SystemExit(
                    f"final_training.selected_profile '{selected_profile}' is not defined in profiles."
                )
            final_profile = selected_profile
            final_profile_source = "configured"

        final_seed = int(final_training.get("seed", 42))
        final_run_name = f"hackathon_final_{sweep_id}_{final_profile}_seed{final_seed}"
        current_run = completed_runs + 1
        print()
        print(
            f"[final {current_run}/{total_scheduled_runs}] Launching final training "
            f"with profile '{final_profile}' "
            f"({final_profile_source}) -> {final_run_name}"
        )
        final_summary_path = run_training(
            python_bin=args.python_bin,
            config_path=args.config,
            missions_file=args.missions_file,
            profiles_config=args.profiles_config,
            profile_name=final_profile,
            stage_settings=final_training,
            seed=final_seed,
            run_name=final_run_name,
            discord_webhook_url=args.discord_webhook_url,
            discord_progress_interval_percent=10,
            discord_notify_completion=True,
        )
        final_summary = read_summary(final_summary_path)
        completed_runs += 1
        summary_payload["final_profile_source"] = final_profile_source
        summary_payload["final_run"] = final_summary
        print(
            f"[sweep progress] completed {completed_runs}/{total_scheduled_runs} runs; "
            f"{total_scheduled_runs - completed_runs} remaining"
        )

    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    markdown_summary_path = write_markdown_summary(summary_dir, summary_payload)

    print(f"Sweep summary saved to: {summary_path}")
    print(f"Sweep markdown summary saved to: {markdown_summary_path}")


if __name__ == "__main__":
    main()
