"""
runner.py — Experiment orchestrator for Cross-Domain QCL.

Generates the cartesian product of (seed x ansatz x noise_model x source),
applies phase filters from config.yaml, and executes runs sequentially or
exports command lists for SLURM array jobs.

Usage:
    # Export command files for all phases
    python runner.py --config config.yaml --dry-run --export-commands

    # Run phase 1 locally (topology ideal)
    python runner.py --config config.yaml --phase topology_ideal

    # Run a single experiment directly
    python runner.py --config config.yaml \
        --ansatz ttn --noise ideal --source synthetic_gaussian --seed 42

    # Run inside a SLURM array (called by slurm_generic.sh)
    python runner.py --config config.yaml --run-id <run_id>
"""

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def make_run_id(ansatz: str, noise: str, source: str, seed: int) -> str:
    return f"{ansatz}__{noise}__{source}__s{seed}"


# ---------------------------------------------------------------------------
# Run enumeration
# ---------------------------------------------------------------------------

def iter_all_runs(cfg: dict) -> Iterator[dict]:
    """Yield all (ansatz, noise, source, seed) combinations from config."""
    ansatze = [a["name"] for a in cfg["ansatze"]]
    noise_models = [n["name"] for n in cfg["noise_models"]]
    sources = [s["name"] for s in cfg["sources"]]
    seeds = cfg["seeds"]

    for ansatz in ansatze:
        for noise in noise_models:
            for source in sources:
                for seed in seeds:
                    run_id = make_run_id(ansatz, noise, source, seed)
                    yield {
                        "run_id": run_id,
                        "ansatz": ansatz,
                        "noise_model": noise,
                        "source": source,
                        "seed": seed,
                    }


def apply_phase_filter(runs: list[dict], phase: dict) -> list[dict]:
    """Filter runs according to a phase's filter specification."""
    filters = phase.get("filters", {})
    result = []
    for r in runs:
        match = True
        for key, val in filters.items():
            if key == "noise" and r["noise_model"] != val:
                match = False
                break
            if key == "source" and r["source"] != val:
                match = False
                break
            if key == "ansatz" and r["ansatz"] != val:
                match = False
                break
        if match:
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def is_completed(run_id: str, results_dir: str) -> bool:
    """Check if a run has already produced a result CSV."""
    csv_path = Path(results_dir) / run_id / "runs.csv"
    return csv_path.exists()


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result, results_dir: str, machine_id: str = "local"):
    """Write result to results/<run_id>/runs.csv."""
    run_dir = Path(results_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "runs.csv"

    row = result.to_dict()
    row["machine_id"] = machine_id
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Command export (for SLURM dry-run)
# ---------------------------------------------------------------------------

def export_commands(runs: list[dict], out_path: str, config_path: str):
    """Write one python runner.py command per run to a text file."""
    lines = []
    for r in runs:
        cmd = (
            f"python runner.py --config {config_path} "
            f"--run-id {r['run_id']} "
            f"--ansatz {r['ansatz']} "
            f"--noise {r['noise_model']} "
            f"--source {r['source']} "
            f"--seed {r['seed']}"
        )
        lines.append(cmd)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Exported {len(lines)} commands to {out_path}")


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def execute_run(run_spec: dict, cfg: dict, machine_id: str = "local"):
    """Build QCLRunConfig and execute a single run."""
    from trainer import QCLRunConfig, run_qcl

    training = cfg.get("training", {})
    noise_cfg = next((n for n in cfg["noise_models"] if n["name"] == run_spec["noise_model"]), {})
    channels = noise_cfg.get("channels", ["amplitude_damping", "phase_damping", "depolarizing"])

    run_cfg = QCLRunConfig(
        run_id=run_spec["run_id"],
        ansatz=run_spec["ansatz"],
        noise_model=run_spec["noise_model"],
        source=run_spec["source"],
        seed=run_spec["seed"],
        n_qubits=cfg.get("n_qubits", 4),
        n_layers=cfg.get("n_layers", 2),
        lr=training.get("lr", 0.05),
        epochs=training.get("epochs", 10),
        pretrain_epochs=training.get("pretrain_epochs", 10),
        batch_size=training.get("batch_size", 32),
        data_dir=cfg.get("data_dir", "./data/raw"),
        results_dir=cfg.get("output_dir", "./results"),
        noise_channels=channels,
        machine_id=machine_id,
    )

    logger.info(f"Starting run: {run_cfg.run_id}")
    result = run_qcl(run_cfg)
    save_result(result, cfg.get("output_dir", "./results"), machine_id)
    logger.info(f"Completed: {run_cfg.run_id} | drop={result.forgetting_drop:.4f}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-Domain QCL experiment runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--phase", default=None, help="Phase name to run (from config phases)")
    parser.add_argument("--run-id", default=None, help="Execute a specific run by its ID")
    parser.add_argument("--ansatz", default=None)
    parser.add_argument("--noise", default=None, help="Noise model name")
    parser.add_argument("--source", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, only show planned runs")
    parser.add_argument("--export-commands", action="store_true", help="Export commands to phase .txt files")
    parser.add_argument("--machine-id", default="local")
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_runs = list(iter_all_runs(cfg))

    # --- Export commands for all phases ---
    if args.export_commands:
        for phase in cfg.get("phases", []):
            filtered = apply_phase_filter(all_runs, phase)
            export_commands(filtered, phase["file"], args.config)
        return

    # --- Specific run by ID (called by SLURM) ---
    if args.run_id:
        run_spec = next((r for r in all_runs if r["run_id"] == args.run_id), None)
        if run_spec is None:
            # Reconstruct from CLI args
            run_spec = {
                "run_id": args.run_id,
                "ansatz": args.ansatz,
                "noise_model": args.noise,
                "source": args.source,
                "seed": args.seed,
            }
        results_dir = cfg.get("output_dir", "./results")
        if is_completed(args.run_id, results_dir):
            logger.info(f"Run {args.run_id} already completed. Skipping.")
            return
        execute_run(run_spec, cfg, args.machine_id)
        return

    # --- Phase run ---
    if args.phase:
        phase_cfg = next((p for p in cfg.get("phases", []) if p["name"] == args.phase), None)
        if phase_cfg is None:
            logger.error(f"Phase '{args.phase}' not found in config.")
            sys.exit(1)
        runs = apply_phase_filter(all_runs, phase_cfg)
    elif args.ansatz or args.noise or args.source or args.seed is not None:
        # Manual filter from CLI
        runs = all_runs
        if args.ansatz:
            runs = [r for r in runs if r["ansatz"] == args.ansatz]
        if args.noise:
            runs = [r for r in runs if r["noise_model"] == args.noise]
        if args.source:
            runs = [r for r in runs if r["source"] == args.source]
        if args.seed is not None:
            runs = [r for r in runs if r["seed"] == args.seed]
    else:
        runs = all_runs

    results_dir = cfg.get("output_dir", "./results")
    pending = [r for r in runs if not is_completed(r["run_id"], results_dir)]

    logger.info(f"Total planned: {len(runs)} | Pending: {len(pending)} | Done: {len(runs) - len(pending)}")

    if args.dry_run:
        for r in pending:
            print(f"  {r['run_id']}")
        return

    for r in pending:
        try:
            execute_run(r, cfg, args.machine_id)
        except Exception as e:
            logger.error(f"Run {r['run_id']} failed: {e}")


if __name__ == "__main__":
    main()
