#!/usr/bin/env python3
"""Train a mask-aware PPO agent for the FleetEnv environment.

This script is designed for sb3-contrib's MaskablePPO, which supports:
- Dict observations via MultiInputPolicy
- MultiDiscrete action spaces
- boolean action masks supplied by the environment

Example:
    python train.py --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from discord_notify import DiscordNotifier, resolve_discord_webhook_url
from fleet_env import FleetEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO policy for the fleet-management environment.",
    )
    parser.add_argument(
        "--profiles-config",
        type=Path,
        default=None,
        help="Optional YAML file with named training hyperparameter profiles.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile name to load from --profiles-config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to the scenario config file.",
    )
    parser.add_argument(
        "--missions-file",
        type=Path,
        default=None,
        help="Optional JSON file with a fixed mission list to use for every episode.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Number of environment timesteps to train for.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate every N environment steps.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to average per checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout length before each policy update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per update.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clipping range.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value-function loss coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Optional PPO target KL for early stopping.",
    )
    parser.add_argument(
        "--max-episode-hours",
        type=float,
        default=2000.0,
        help="Maximum simulated hours before an episode is truncated.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Width of each policy/value hidden layer.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Number of policy/value hidden layers.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to Stable-Baselines3 (for example: auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--policy",
        default="MultiInputPolicy",
        help="SB3 policy type. Dict observations require MultiInputPolicy in this project.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where trained models and logs are written.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="Optional root directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--csv-log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-run CSV summaries.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to a timestamped MaskablePPO name.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity level.",
    )
    parser.add_argument(
        "--save-best-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the best checkpoint from evaluation callbacks.",
    )
    parser.add_argument(
        "--save-replay-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-episode evaluation metrics after training.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the Stable-Baselines3 progress bar during training.",
    )
    parser.add_argument(
        "--discord-webhook-url",
        default=None,
        help="Optional Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL.",
    )
    parser.add_argument(
        "--discord-progress-interval-percent",
        type=int,
        default=0,
        help="Optional progress notification interval percentage (for example: 10).",
    )
    parser.add_argument(
        "--discord-notify-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send a Discord message when the training run finishes.",
    )
    return parser.parse_args()


def load_profiles_config_data(profiles_config: Path) -> dict:
    return yaml.safe_load(profiles_config.read_text(encoding="utf-8")) or {}


def load_profile(profiles_config: Path, profile_name: str) -> dict:
    data = load_profiles_config_data(profiles_config)
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise SystemExit(
            f"Profile '{profile_name}' not found in {profiles_config}. Available: {available}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise SystemExit(
            f"Profile '{profile_name}' in {profiles_config} must be a mapping."
        )
    return profile


def apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile is None:
        return args
    if args.profiles_config is None:
        raise SystemExit("--profile requires --profiles-config")
    if not args.profiles_config.exists():
        raise SystemExit(f"Profiles config not found: {args.profiles_config}")

    data = load_profiles_config_data(args.profiles_config)
    common_ppo = data.get("common_ppo", {})
    logging_cfg = data.get("logging", {})

    for field, cfg_key in (
        ("policy", "policy"),
        ("clip_range", "clip_range"),
        ("vf_coef", "vf_coef"),
        ("max_grad_norm", "max_grad_norm"),
        ("target_kl", "target_kl"),
    ):
        if cfg_key in common_ppo:
            setattr(args, field, common_ppo[cfg_key])

    if args.output_dir == Path("artifacts") and "save_dir" in logging_cfg:
        args.output_dir = Path(logging_cfg["save_dir"])
    if args.tensorboard_log_dir is None and "tensorboard_log_dir" in logging_cfg:
        args.tensorboard_log_dir = Path(logging_cfg["tensorboard_log_dir"])
    if args.csv_log_dir is None and "csv_log_dir" in logging_cfg:
        args.csv_log_dir = Path(logging_cfg["csv_log_dir"])

    profile = load_profile(args.profiles_config, args.profile)
    for field in (
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "max_episode_hours",
        "hidden_size",
        "hidden_layers",
    ):
        if field in profile:
            setattr(args, field, profile[field])
    return args


def import_training_deps():
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise SystemExit(
            "Training dependencies are missing. Install the packages in "
            "`requirements.txt` inside a Python 3.12 or 3.13 environment "
            "(or create the `mamba` environment from `environment.yml`) "
            f"(current interpreter: {version}). Original import error: {exc}"
        ) from exc

    return (
        MaskablePPO,
        MaskableEvalCallback,
        evaluate_policy,
        ActionMasker,
        Monitor,
        BaseCallback,
        CallbackList,
    )


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return requested_device

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return requested_device


def resolve_policy_name(requested_policy: str) -> str:
    if requested_policy == "MultiInputPolicy":
        return requested_policy

    print(
        "Warning: FleetEnv uses Dict observations, so overriding requested policy "
        f"'{requested_policy}' with 'MultiInputPolicy'.",
        file=sys.stderr,
    )
    return "MultiInputPolicy"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_progress_callback(
    base_callback_cls,
    total_timesteps: int,
    enabled: bool,
    notifier: DiscordNotifier,
    progress_interval_percent: int,
    run_label: str,
    profile_name: str | None,
):
    class TrainingProgressCallback(base_callback_cls):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.total_timesteps = max(1, int(total_timesteps))
            self.enabled = enabled
            self.start_time = 0.0
            self.last_reported_step = -1
            self.notifier = notifier
            self.progress_interval_percent = max(0, int(progress_interval_percent))
            self.run_label = run_label
            self.profile_name = profile_name
            self.next_progress_mark = self.progress_interval_percent

        def _emit_progress(self) -> None:
            completed_steps = min(int(self.num_timesteps), self.total_timesteps)
            elapsed = max(time.monotonic() - self.start_time, 1e-9)
            avg_fps = completed_steps / elapsed if completed_steps > 0 else 0.0
            remaining_steps = max(self.total_timesteps - completed_steps, 0)
            eta_seconds = (
                remaining_steps / avg_fps if remaining_steps > 0 and avg_fps > 0 else 0.0
            )
            if self.enabled and completed_steps > self.last_reported_step:
                print(
                    f"[progress] {completed_steps}/{self.total_timesteps} steps "
                    f"({completed_steps / self.total_timesteps:.1%}) | "
                    f"elapsed {format_duration(elapsed)} | "
                    f"eta {format_duration(eta_seconds)} | avg fps {avg_fps:.1f}"
                )
            completed_percent = (completed_steps / self.total_timesteps) * 100.0
            while (
                self.notifier.enabled
                and self.progress_interval_percent > 0
                and self.next_progress_mark < 100
                and completed_percent >= self.next_progress_mark
            ):
                lines = [
                    f"**Training progress: {self.next_progress_mark}%**",
                    f"- Run: `{self.run_label}`",
                ]
                if self.profile_name is not None:
                    lines.append(f"- Profile: `{self.profile_name}`")
                lines.extend(
                    [
                        f"- Steps: `{completed_steps}/{self.total_timesteps}`",
                        f"- Elapsed: `{format_duration(elapsed)}`",
                        f"- ETA: `{format_duration(eta_seconds)}`",
                    ]
                )
                self.notifier.send("\n".join(lines))
                self.next_progress_mark += self.progress_interval_percent
            self.last_reported_step = completed_steps

        def _on_training_start(self) -> None:
            self.start_time = time.monotonic()
            if self.enabled:
                print(
                    f"[progress] 0/{self.total_timesteps} steps (0.0%) | "
                    "elapsed 00:00:00 | eta unknown | avg fps 0.0"
                )

        def _on_rollout_end(self) -> None:
            self._emit_progress()

        def _on_training_end(self) -> None:
            self._emit_progress()

        def _on_step(self) -> bool:
            return True

    return TrainingProgressCallback()


def send_training_completion_notification(
    notifier: DiscordNotifier,
    run_label: str,
    profile_name: str | None,
    aggregate_metrics: dict,
    total_timesteps: int,
    final_model_path: Path,
) -> None:
    if not notifier.enabled:
        return

    profile_line = f"- Profile: `{profile_name}`" if profile_name is not None else None
    lines = [
        "**Training complete**",
        f"- Run: `{run_label}`",
        profile_line,
        f"- Timesteps: `{total_timesteps}`",
        f"- Reward: `{aggregate_metrics['eval_mean_reward']:.3f}` +/- `{aggregate_metrics['eval_std_reward']:.3f}`",
        f"- Missions completed: `{aggregate_metrics['missions_completed']:.2f}`",
        f"- Missions missed: `{aggregate_metrics['missions_missed']:.2f}`",
        f"- Model: `{final_model_path.with_suffix('.zip')}`",
    ]
    notifier.send("\n".join(line for line in lines if line is not None))


def send_training_failure_notification(
    notifier: DiscordNotifier,
    run_label: str,
    profile_name: str | None,
    exc: Exception,
) -> None:
    if not notifier.enabled:
        return

    lines = [
        "**Training failed**",
        f"- Run: `{run_label}`",
    ]
    if profile_name is not None:
        lines.append(f"- Profile: `{profile_name}`")
    lines.append(f"- Error: `{type(exc).__name__}: {exc}`")
    notifier.send("\n".join(lines))


def build_run_dir(output_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"maskable_ppo_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_env(
    config_path: Path,
    missions_file: Path | None,
    max_episode_hours: float,
    record_events: bool,
    action_masker_cls,
    monitor_cls,
):
    env = FleetEnv(
        config_path,
        mission_manifest_path=missions_file,
        max_episode_hours=max_episode_hours,
        record_events=record_events,
    )
    env = monitor_cls(env)
    return action_masker_cls(env, lambda wrapped_env: wrapped_env.unwrapped.action_masks())


def _initial_total_fuel(env: FleetEnv) -> float:
    aircraft_fuel = sum(ac.fuel_level for ac in env._fleet_template.values())
    base_fuel = sum(base.fuel for base in env._base_template.values())
    return float(aircraft_fuel + base_fuel)


def _current_total_fuel(env: FleetEnv) -> float:
    aircraft_fuel = sum(ac.fuel_level for ac in env.fleet.values())
    base_fuel = sum(base.fuel for base in env.bases.values())
    return float(aircraft_fuel + base_fuel)


def extract_episode_metrics(
    env: FleetEnv,
    info: dict,
    episode_reward: float,
    terminated: bool,
    truncated: bool,
) -> dict:
    events = info.get("episode_events") or []
    transfers = sum(1 for event in events if event.get("event_type") == "transfer")
    maintenance_events = sum(
        1 for event in events if event.get("event_type") == "maintenance"
    )
    missions_completed = int(info.get("missions_completed", 0))
    missions_missed = max(0, env.total_missions - missions_completed)
    fuel_left = _current_total_fuel(env)
    fuel_used = max(0.0, _initial_total_fuel(env) - fuel_left)

    return {
        "episode_reward": float(episode_reward),
        "episode_length": int(info.get("step_index", 0)),
        "missions_completed": missions_completed,
        "missions_missed": missions_missed,
        "fuel_used": fuel_used,
        "fuel_left": fuel_left,
        "maintenance_events": maintenance_events,
        "transfers": transfers,
        # Action masking removes invalid choices before sampling, so this remains 0.
        "invalid_action_rate": 0.0,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def aggregate_episode_metrics(per_episode: list[dict]) -> dict:
    if not per_episode:
        return {}

    aggregate = {}
    metric_names = (
        "episode_reward",
        "episode_length",
        "missions_completed",
        "missions_missed",
        "fuel_used",
        "fuel_left",
        "maintenance_events",
        "transfers",
        "invalid_action_rate",
    )
    for metric_name in metric_names:
        values = np.array([episode[metric_name] for episode in per_episode], dtype=np.float64)
        aggregate[metric_name] = float(values.mean())
        aggregate[f"{metric_name}_std"] = float(values.std(ddof=0))

    aggregate["completion_rate"] = float(
        np.mean([1.0 if episode["terminated"] else 0.0 for episode in per_episode])
    )
    aggregate["truncation_rate"] = float(
        np.mean([1.0 if episode["truncated"] else 0.0 for episode in per_episode])
    )
    return aggregate


def evaluate_model(model, eval_env, n_eval_episodes: int, seed: int) -> tuple[dict, list[dict]]:
    base_env = eval_env.unwrapped
    per_episode = []
    rewards = []

    for episode_idx in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=seed + episode_idx)
        episode_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            action_masks = base_env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += float(reward)

        rewards.append(episode_reward)
        per_episode.append(
            extract_episode_metrics(
                base_env,
                info,
                episode_reward,
                terminated,
                truncated,
            )
        )

    aggregate = aggregate_episode_metrics(per_episode)
    aggregate["eval_mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
    aggregate["eval_std_reward"] = float(np.std(rewards, ddof=0)) if rewards else 0.0
    return aggregate, per_episode


def write_evaluation_metrics(run_dir: Path, aggregate_metrics: dict, per_episode: list[dict]) -> Path:
    metrics_path = run_dir / "evaluation_metrics.json"
    payload = {
        "aggregate": aggregate_metrics,
        "episodes": per_episode,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path


def write_csv_summary(csv_log_dir: Path, summary: dict) -> Path:
    csv_log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_log_dir / "training_runs.csv"
    row = {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def main() -> None:
    args = apply_profile_overrides(parse_args())
    notifier = DiscordNotifier(resolve_discord_webhook_url(args.discord_webhook_url))
    run_dir: Path | None = None
    (
        MaskablePPO,
        MaskableEvalCallback,
        evaluate_policy,
        ActionMasker,
        Monitor,
        BaseCallback,
        CallbackList,
    ) = import_training_deps()

    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if args.missions_file is not None and not args.missions_file.exists():
        raise SystemExit(f"Missions file not found: {args.missions_file}")
    if args.profiles_config is not None and not args.profiles_config.exists():
        raise SystemExit(f"Profiles config not found: {args.profiles_config}")

    if sys.version_info >= (3, 14):
        print(
            "Warning: Python 3.14 is ahead of the versions commonly supported by "
            "PyTorch / Stable-Baselines3 right now. Python 3.12 is recommended "
            "for the training environment.",
            file=sys.stderr,
        )

    try:
        run_dir = build_run_dir(args.output_dir, args.run_name)
        run_label = run_dir.name
        best_model_dir = run_dir / "best_model"
        eval_log_dir = run_dir / "eval"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        if args.save_best_model:
            best_model_dir.mkdir(parents=True, exist_ok=True)
        resolved_device = resolve_device(args.device)
        resolved_policy = resolve_policy_name(args.policy)
        tensorboard_log_dir = (
            args.tensorboard_log_dir / run_dir.name
            if args.tensorboard_log_dir is not None
            else run_dir / "tensorboard"
        )
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        train_env = make_env(
            args.config,
            args.missions_file,
            args.max_episode_hours,
            False,
            ActionMasker,
            Monitor,
        )
        eval_env = make_env(
            args.config,
            args.missions_file,
            args.max_episode_hours,
            False,
            ActionMasker,
            Monitor,
        )
        analysis_eval_env = make_env(
            args.config,
            args.missions_file,
            args.max_episode_hours,
            True,
            ActionMasker,
            Monitor,
        )

        hidden_layers = [args.hidden_size] * max(1, args.hidden_layers)
        policy_kwargs = {
            "net_arch": {
                "pi": hidden_layers,
                "vf": hidden_layers,
            },
        }

        model = MaskablePPO(
            policy=resolved_policy,
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            seed=args.seed,
            device=resolved_device,
            verbose=args.verbose,
            tensorboard_log=str(tensorboard_log_dir),
            policy_kwargs=policy_kwargs,
        )

        eval_callback = MaskableEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_model_dir) if args.save_best_model else None,
            log_path=str(eval_log_dir),
            eval_freq=max(1, args.eval_freq),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        progress_callback = build_progress_callback(
            BaseCallback,
            total_timesteps=args.total_timesteps,
            enabled=args.verbose > 0,
            notifier=notifier,
            progress_interval_percent=args.discord_progress_interval_percent,
            run_label=run_label,
            profile_name=args.profile,
        )
        callback = CallbackList([progress_callback, eval_callback])

        print(f"Training run directory: {run_dir}")
        print(f"Config: {args.config}")
        if args.profile is not None:
            print(f"Profile: {args.profile}")
        if args.profiles_config is not None:
            print(f"Profiles config: {args.profiles_config}")
        if args.missions_file is not None:
            print(f"Missions file: {args.missions_file}")
        print(f"Total timesteps: {args.total_timesteps}")
        print(f"Max episode hours: {args.max_episode_hours}")
        print(f"Device: {resolved_device}")
        print(f"Policy: {resolved_policy}")
        print(f"Seed: {args.seed}")

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
            tb_log_name="fleet",
        )

        final_model_path = run_dir / "final_model"
        model.save(str(final_model_path))

        aggregate_metrics, per_episode_metrics = evaluate_model(
            model,
            analysis_eval_env,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
        )
        evaluation_metrics_path = None
        if args.save_replay_metrics:
            evaluation_metrics_path = write_evaluation_metrics(
                run_dir,
                aggregate_metrics,
                per_episode_metrics,
            )

        summary = {
            "algorithm": "MaskablePPO",
            "policy": resolved_policy,
            "config_path": str(args.config),
            "profile_name": args.profile,
            "profiles_config": str(args.profiles_config) if args.profiles_config is not None else None,
            "missions_file": str(args.missions_file) if args.missions_file is not None else None,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "clip_range": args.clip_range,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "target_kl": args.target_kl,
            "max_episode_hours": args.max_episode_hours,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "device": resolved_device,
            "final_model_path": str(final_model_path.with_suffix(".zip")),
            "best_model_path": (
                str(best_model_dir / "best_model.zip") if args.save_best_model else None
            ),
            "eval_mean_reward": aggregate_metrics["eval_mean_reward"],
            "eval_std_reward": aggregate_metrics["eval_std_reward"],
            "evaluation_metrics": aggregate_metrics,
            "evaluation_metrics_path": (
                str(evaluation_metrics_path) if evaluation_metrics_path is not None else None
            ),
        }
        summary.update(aggregate_metrics)
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        csv_summary_path = None
        if args.csv_log_dir is not None:
            csv_summary_path = write_csv_summary(args.csv_log_dir, summary)

        print(f"Final model saved to: {final_model_path.with_suffix('.zip')}")
        if args.save_best_model:
            print(f"Best model saved to: {best_model_dir / 'best_model.zip'}")
        if evaluation_metrics_path is not None:
            print(f"Evaluation metrics saved to: {evaluation_metrics_path}")
        if csv_summary_path is not None:
            print(f"CSV summary updated at: {csv_summary_path}")
        print(
            f"Evaluation mean reward: {aggregate_metrics['eval_mean_reward']:.3f} "
            f"+/- {aggregate_metrics['eval_std_reward']:.3f}"
        )
        print(f"Training summary saved to: {run_dir / 'training_summary.json'}")

        if args.discord_notify_completion:
            send_training_completion_notification(
                notifier=notifier,
                run_label=run_label,
                profile_name=args.profile,
                aggregate_metrics=aggregate_metrics,
                total_timesteps=args.total_timesteps,
                final_model_path=final_model_path,
            )
    except Exception as exc:
        run_label = run_dir.name if run_dir is not None else (args.run_name or "unknown_run")
        if (
            args.discord_notify_completion
            or args.discord_progress_interval_percent > 0
        ):
            send_training_failure_notification(
                notifier=notifier,
                run_label=run_label,
                profile_name=args.profile,
                exc=exc,
            )
        raise


if __name__ == "__main__":
    main()
