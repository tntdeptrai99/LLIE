import subprocess
import sys
from pathlib import Path
import shutil

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def copy_checkpoint(src_dir, dst_dir):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    if not (Path(dst_dir) / "last.pt").exists():
        for file in Path(src_dir).glob("*.pt"):
            shutil.copy(file, Path(dst_dir) / file.name)
        if (Path(src_dir) / "train_log.csv").exists():
            shutil.copy(Path(src_dir) / "train_log.csv", Path(dst_dir) / "train_log.csv")

def main():
    common_args_supervised = [
        "--train-split", "splits/lol_train.txt", "splits/lolv2_real_train.txt",
        "--val-split", "splits/lol_val.txt", "splits/lolv2_real_val.txt",
        "--base-channels", "12",
        "--mid-channels", "24",
        "--gain-max", "3.0",
        "--residual-scale", "0.35",
        "--batch-size", "32",
        "--lr", "3e-4",
        "--image-size", "96",
        "--num-workers", "0",
        "--resume"
    ]

    distill_args = [
        "--train-split", "splits/lol_train_retinexformer_rgb.txt",
        "--val-split", "splits/lol_test_retinexformer_rgb.txt",
        "--base-channels", "12",
        "--mid-channels", "24",
        "--gain-max", "3.0",
        "--residual-scale", "0.35",
        "--batch-size", "32",
        "--lr", "3e-4",
        "--image-size", "96",
        "--num-workers", "0",
        "--resume",
        "--student-variant", "GHOST-ESP-DARK",
        "--blocks", "3"
    ]

    print("--- 1. MAE Baseline (Resume 30 -> 300) ---")
    src = "experiments/component_ablation/mae_baseline_w12m24_20260806"
    dst = "experiments/component_ablation/mae_baseline_w12m24_300e"
    copy_checkpoint(src, dst)
    run_cmd([sys.executable, "scripts/train_supervised.py"] + common_args_supervised + [
        "--model-variant", "GHOST-ESP-DARK", 
        "--e-blocks", "3",
        "--epochs", "300",
        "--pretrained", f"{dst}/last.pt",
        "--out-dir", dst
    ])

    print("--- 2. Base Model (Resume 80 -> 300) ---")
    src = "experiments/ghost_esp_dark_w12_m24_gain3_res035_96_run1"
    dst = "experiments/component_ablation/base_model_w12m24_300e"
    copy_checkpoint(src, dst)
    run_cmd([sys.executable, "scripts/train_distill.py"] + distill_args + [
        "--epochs", "300",
        "--pretrained", f"{dst}/last.pt",
        "--out-dir", dst
    ])

    print("--- 3. No Dark-Map Ablation (Resume 30 -> 300) ---")
    src = "experiments/component_ablation/no_adaptive_dark_loss_w12m24_20260726"
    dst = "experiments/component_ablation/no_dark_w12m24_300e"
    copy_checkpoint(src, dst)
    run_cmd([sys.executable, "scripts/train_distill.py"] + distill_args + [
        "--adaptive-dark-weight", "0.0",
        "--epochs", "300",
        "--pretrained", f"{dst}/last.pt",
        "--out-dir", dst
    ])

    print("--- 4. No Teacher KD Ablation (Resume 30 -> 300) ---")
    src = "experiments/component_ablation/no_teacher_kd_w12m24_20260726"
    dst = "experiments/component_ablation/no_teacher_w12m24_300e"
    copy_checkpoint(src, dst)
    run_cmd([sys.executable, "scripts/train_distill.py"] + distill_args + [
        "--kd-weight", "0.0",
        "--epochs", "300",
        "--pretrained", f"{dst}/last.pt",
        "--out-dir", dst
    ])

    print("--- 5. Optuna Best Model (Resume 180 -> 220, total 300) ---")
    src = "experiments/refine/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_from_long80_best_ssim"
    dst = "experiments/component_ablation/optuna_best_w12m24_300e"
    copy_checkpoint(src, dst)
    run_cmd([sys.executable, "scripts/train_distill.py"] + distill_args + [
        "--kd-weight", "0.414907",
        "--ssim-weight", "0.101260",
        "--teacher-ssim-weight", "0.059698",
        "--edge-kd-weight", "0.023192",
        "--edge-gt-weight", "0.024092",
        "--color-gt-weight", "0.055340",
        "--adaptive-dark-weight", "0.005375",
        "--adaptive-dark-base", "0.399048",
        "--adaptive-dark-gain", "1.279561",
        "--epochs", "300",
        "--pretrained", f"{dst}/last.pt",
        "--out-dir", dst
    ])

    print("--- 6. Evaluation ---")
    eval_cmds = [
        [sys.executable, "scripts/eval_model.py", "--checkpoint", "experiments/component_ablation/mae_baseline_w12m24_300e/best.pt", "--metric-name", "mae_baseline_300e", "--name", "lol_test", "--figure-dir", "reports/figures/mae_baseline_300e"],
        [sys.executable, "scripts/eval_model.py", "--checkpoint", "experiments/component_ablation/base_model_w12m24_300e/best.pt", "--metric-name", "base_model_300e", "--name", "lol_test", "--figure-dir", "reports/figures/base_model_300e"],
        [sys.executable, "scripts/eval_model.py", "--checkpoint", "experiments/component_ablation/no_dark_w12m24_300e/best.pt", "--metric-name", "no_dark_300e", "--name", "lol_test", "--figure-dir", "reports/figures/no_dark_300e"],
        [sys.executable, "scripts/eval_model.py", "--checkpoint", "experiments/component_ablation/no_teacher_w12m24_300e/best.pt", "--metric-name", "no_teacher_300e", "--name", "lol_test", "--figure-dir", "reports/figures/no_teacher_300e"],
        [sys.executable, "scripts/eval_model.py", "--checkpoint", "experiments/component_ablation/optuna_best_w12m24_300e/best.pt", "--metric-name", "optuna_best_300e", "--name", "lol_test", "--figure-dir", "reports/figures/optuna_best_300e"],
    ]
    for cmd in eval_cmds:
        run_cmd(cmd)

    print("--- 7. Generate Figures ---")
    run_cmd([sys.executable, "scripts/make_paper_style_figures.py"])

    print("--- 8. Generate Final Report ---")
    run_cmd([sys.executable, "scripts/create_final_comprehensive_report.py"])

if __name__ == "__main__":
    main()
