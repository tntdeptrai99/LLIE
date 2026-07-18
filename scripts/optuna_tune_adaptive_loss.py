from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import optuna
import torch


def checkpoint_val(path: Path) -> dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    val = checkpoint.get("val", {})
    return {
        "loss": float(val.get("loss", 0.0)),
        "psnr": float(val.get("psnr", 0.0)),
        "ssim": float(val.get("ssim", 0.0)),
    }


def score(
    val: dict[str, float],
    ssim_scale: float,
    min_ssim: float,
    penalty_scale: float,
) -> float:
    ssim_penalty = max(0.0, min_ssim - val["ssim"])
    return val["psnr"] + ssim_scale * val["ssim"] - penalty_scale * ssim_penalty


def write_trials_csv(study: optuna.Study, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update(trial.params)
        row.update({f"user_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="ghost_esp_dark_adaptive_loss")
    parser.add_argument("--storage", type=Path, default=Path("experiments/optuna/ghost_esp_dark_adaptive_loss.db"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/optuna/ghost_esp_dark_adaptive_loss"))
    parser.add_argument("--pretrained", type=Path, default=Path("experiments/ghost_esp_dark_rgb_kd_96_run1/best.pt"))
    parser.add_argument("--train-split", type=Path, nargs="+", default=[Path("splits/lol_train_retinexformer_rgb.txt")])
    parser.add_argument("--val-split", type=Path, nargs="+", default=[Path("splits/lol_test_retinexformer_rgb.txt")])
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-ssim-scale", type=float, default=10.0)
    parser.add_argument("--score-min-ssim", type=float, default=0.80)
    parser.add_argument("--score-penalty-scale", type=float, default=20.0)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--mid-channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{args.storage.as_posix()}"

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            "kd_weight": trial.suggest_float("kd_weight", 0.20, 0.50),
            "kd_dark_gain": trial.suggest_float("kd_dark_gain", 0.0, 1.2),
            "ssim_weight": trial.suggest_float("ssim_weight", 0.10, 0.35),
            "ssim_dark_gain": trial.suggest_float("ssim_dark_gain", 0.0, 1.0),
            "teacher_ssim_weight": trial.suggest_float("teacher_ssim_weight", 0.02, 0.10),
            "edge_kd_weight": trial.suggest_float("edge_kd_weight", 0.0, 0.05),
            "edge_gt_weight": trial.suggest_float("edge_gt_weight", 0.02, 0.12),
            "color_gt_weight": trial.suggest_float("color_gt_weight", 0.02, 0.12),
            "adaptive_dark_weight": trial.suggest_float("adaptive_dark_weight", 0.005, 0.08, log=True),
            "adaptive_dark_base": trial.suggest_float("adaptive_dark_base", 0.0, 0.6),
            "adaptive_dark_gain": trial.suggest_float("adaptive_dark_gain", 0.4, 2.0),
            "adaptive_dark_gamma": trial.suggest_float("adaptive_dark_gamma", 0.6, 2.2),
            "adaptive_bright_gamma": trial.suggest_float("adaptive_bright_gamma", 0.6, 2.0),
            "adaptive_bright_target": trial.suggest_categorical("adaptive_bright_target", ["low", "target"]),
        }
        trial_dir = args.out_dir / f"trial_{trial.number:03d}"
        command = [
            sys.executable,
            "scripts/train_distill.py",
            "--student-variant",
            "GHOST-ESP-DARK",
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--train-limit",
            str(args.train_limit),
            "--val-limit",
            str(args.val_limit),
            "--out-dir",
            str(trial_dir),
            "--pretrained",
            str(args.pretrained),
            "--seed",
            str(args.seed + trial.number),
            "--num-workers",
            str(args.num_workers),
            "--gt-weight",
            "1.0",
            "--contrast-gt-weight",
            "0.0",
            "--chroma-gt-weight",
            "0.0",
            "--blocks",
            str(args.blocks),
            "--base-channels",
            str(args.base_channels),
            "--mid-channels",
            str(args.mid_channels),
            "--gain-max",
            str(args.gain_max),
            "--residual-scale",
            str(args.residual_scale),
        ]
        command.append("--train-split")
        command.extend(str(split) for split in args.train_split)
        command.append("--val-split")
        command.extend(str(split) for split in args.val_split)
        for key, value in params.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])

        completed = subprocess.run(command, cwd=Path.cwd(), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Trial {trial.number} failed with exit code {completed.returncode}")

        best = checkpoint_val(trial_dir / "best.pt")
        best_ssim = checkpoint_val(trial_dir / "best_ssim.pt")
        trial.set_user_attr("best_psnr", best["psnr"])
        trial.set_user_attr("best_ssim_at_best_psnr", best["ssim"])
        trial.set_user_attr("best_ssim_psnr", best_ssim["psnr"])
        trial.set_user_attr("best_ssim", best_ssim["ssim"])
        trial.set_user_attr("trial_dir", str(trial_dir))
        best_score = score(
            best,
            args.score_ssim_scale,
            args.score_min_ssim,
            args.score_penalty_scale,
        )
        best_ssim_score = score(
            best_ssim,
            args.score_ssim_scale,
            args.score_min_ssim,
            args.score_penalty_scale,
        )
        trial.set_user_attr("best_psnr_score", best_score)
        trial.set_user_attr("best_ssim_score", best_ssim_score)
        trial.set_user_attr(
            "selected_checkpoint",
            "best_ssim.pt" if best_ssim_score > best_score else "best.pt",
        )
        return max(best_score, best_ssim_score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    (args.out_dir / "best_params.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )
    write_trials_csv(study, args.out_dir / "trials.csv")
    print(json.dumps(best, indent=2))
    print(f"best_params={args.out_dir / 'best_params.json'}")
    print(f"trials_csv={args.out_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
