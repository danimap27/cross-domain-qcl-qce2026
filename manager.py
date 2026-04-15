"""
manager.py — Generic Experiment Control Hub for Hercules HPC.

Reads phases and labels from config.yaml. Compatible with any paper
that follows the runner.py + config.yaml + slurm_generic.sh structure.

Usage:
    python manager.py
    python manager.py --config config.yaml
"""

import os
import sys
import subprocess
import glob
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(experiment_name: str):
    width = 70
    print("=" * width)
    title = f"EXPERIMENT HUB — {experiment_name.upper()}"
    print(f"{title:^{width}}")
    print("=" * width)


def run_cmd(cmd: str, capture: bool = False):
    try:
        if capture:
            r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return r.stdout.strip()
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {cmd}\n{e}")
        return None


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    for enc in ["utf-8-sig", "utf-16", "latin1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return len([l for l in f if l.strip()])
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 0


def refresh_commands(config_path: str, cfg: dict):
    """Regenerate all phase command files via runner.py --export-commands."""
    print(f"\n[INFO] Regenerating command files from {config_path}...")
    ok = run_cmd(f"python runner.py --config {config_path} --export-commands")
    if ok:
        print("[OK] Command files regenerated.")
        for phase in cfg.get("phases", []):
            n = count_lines(phase["file"])
            print(f"  - [{phase['id']}] {phase['description']}: {n} tasks")
    else:
        print("[FAIL] Check runner.py output.")
    input("\nEnter to return...")


def submit_phase(phase: dict, dependency_id: Optional[str] = None) -> Optional[str]:
    """Submit a phase as a SLURM array job."""
    n = count_lines(phase["file"])
    if n == 0:
        print(f"\n[WARN] {phase['file']} is empty or missing. Run [R] first.")
        return None

    dep = f"--dependency=afterok:{dependency_id}" if dependency_id else ""
    job_name = f"QCL_{phase['id']}_{phase['name']}"
    cmd = (
        f"sbatch --parsable --job-name='{job_name}' "
        f"--array=1-{n}%30 {dep} "
        f"--export=CMD_FILE={phase['file']} slurm_generic.sh"
    )
    print(f"\n[SUBMIT] {phase['description']} ({n} tasks)...")
    job_id = run_cmd(cmd, capture=True)
    if job_id:
        print(f"[OK] Job ID: {job_id}")
    return job_id


def show_monitor(cfg: dict):
    """Scan results directory and report progress."""
    results_dir = cfg.get("output_dir", "./results")
    expected = cfg.get("expected_runs", 0)

    print(f"\n[MONITOR] Scanning {results_dir}/...")
    pattern = os.path.join(results_dir, "*", "runs.csv")
    csv_files = glob.glob(pattern)

    completed = 0
    if HAS_PANDAS and csv_files:
        try:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
            completed = len(df.drop_duplicates(subset=["run_id"]))
            print(f"\n{'Metric':<25} {'Value':>10}")
            print("-" * 37)
            print(f"{'Unique runs completed':<25} {completed:>10}")
            if "forgetting_drop" in df.columns:
                mean_drop = df.groupby("ansatz")["forgetting_drop"].mean()
                print("\nMean forgetting drop by ansatz:")
                for ansatz, val in mean_drop.items():
                    label = cfg.get("labels", {}).get("ansatze", {}).get(ansatz, ansatz)
                    print(f"  {label:<20}: {val*100:>6.1f}%")
        except Exception as e:
            print(f"[WARN] {e}")
            completed = len(csv_files)
    else:
        completed = len(csv_files)
        print(f"Completed run folders: {completed}")

    pct = (completed / expected * 100) if expected > 0 else 0
    print(f"\nOverall progress: {pct:.1f}% ({completed}/{expected})")

    print("\n--- Active SLURM jobs ---")
    out = run_cmd("squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M %.6D' 2>/dev/null", capture=True)
    print(out or "No active jobs or squeue unavailable.")
    input("\nEnter to return...")


def launch_full_pipeline(cfg: dict):
    """Submit all phases with sequential SLURM dependencies."""
    phases = cfg.get("phases", [])
    print(f"\n[PIPELINE] Submitting {len(phases)} phases with sequential dependencies...")
    prev_id = None
    ids = []
    for phase in phases:
        job_id = submit_phase(phase, dependency_id=prev_id)
        ids.append(job_id)
        if job_id:
            prev_id = job_id
    print(f"\n[OK] Chain: {' -> '.join(str(i) for i in ids)}")
    input("\nEnter to return...")


def generate_tables(config_path: str):
    """Run generate_tables.py."""
    print("\n[TABLES] Generating LaTeX tables...")
    run_cmd(f"python generate_tables.py --config {config_path}")
    input("\nEnter to return...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment HUB")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment_name", "experiment")
    phases = cfg.get("phases", [])

    os.makedirs("logs", exist_ok=True)
    os.makedirs(cfg.get("output_dir", "./results"), exist_ok=True)

    while True:
        clear_screen()
        print_header(experiment_name)
        print()
        print("  [R]  Refresh command files from config.yaml")
        print()
        for phase in phases:
            n = count_lines(phase["file"])
            print(f"  [{phase['id']}]  {phase['description']}  ({n} tasks)")
        print()
        print("  [F]  Launch FULL PIPELINE (all phases, sequential deps)")
        print("  [M]  Monitor progress and SLURM queue")
        print("  [T]  Generate LaTeX tables")
        print("  [X]  Exit")
        print("-" * 70)

        choice = input("Option: ").strip().upper()

        if choice == "R":
            refresh_commands(args.config, cfg)
        elif choice == "F":
            launch_full_pipeline(cfg)
        elif choice == "M":
            show_monitor(cfg)
        elif choice == "T":
            generate_tables(args.config)
        elif choice == "X":
            print("\nExiting.\n")
            break
        elif choice in {p["id"] for p in phases}:
            phase = next(p for p in phases if p["id"] == choice)
            submit_phase(phase)
            input("\nEnter to return...")
        else:
            print("Invalid option.")
            time.sleep(1)


if __name__ == "__main__":
    main()
