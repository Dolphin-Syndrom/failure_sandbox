"""
experiments/run_sweep.py
=========================
Runs all experiment conditions (nominal + perturbations) and saves telemetry CSVs.

Usage:
    cd failure_sandbox
    python experiments/run_sweep.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import time
import json
import subprocess
from simulation.task import PickPlaceTask, load_config
from telemetry.logger import EpisodeLogger


def run_episode(config: dict, episode_id: str, output_dir: str) -> dict:
    """Run one episode, log telemetry, return summary."""
    task = PickPlaceTask(config)
    task.reset()

    logger = EpisodeLogger(
        episode_id=episode_id,
        scenario_id=config.get("scenario_id", "unknown"),
        output_dir=output_dir,
    )

    # Add episode/scenario IDs to every row
    for _ in range(3000):
        row = task.step()
        row["episode_id"]  = episode_id
        row["scenario_id"] = config.get("scenario_id", "unknown")
        logger.log_step(row)
        if task.done:
            break

    success, fail_stage = task.outcome()
    logger.close(success=success, failure_stage=fail_stage)

    return {
        "episode_id": episode_id,
        "scenario_id": config.get("scenario_id"),
        "success": success,
        "failure_stage": fail_stage,
        "steps": task.step_count,
    }


def run_condition(config: dict, output_dir: str) -> list:
    """Run N episodes for one condition."""
    scenario = config["scenario_id"]
    n = config.get("num_episodes", 5)
    base_seed = config.get("random_seed", 42)
    results = []

    for i in range(n):
        cfg = dict(config)
        cfg["random_seed"] = base_seed + i   # different seed per episode
        ep_id = f"{scenario}_ep{i:03d}"
        summary = run_episode(cfg, ep_id, output_dir)
        results.append(summary)

    return results


def main():
    output_dir = "data/episodes"
    all_results = []

    # ── 1. Nominal baseline ──────────────────────────
    print("=" * 60)
    print("NOMINAL BASELINE")
    print("=" * 60)
    nominal = load_config("configs/nominal.yaml")
    nominal["scenario_id"] = "nominal"
    nominal["num_episodes"] = 8
    results = run_condition(nominal, output_dir)
    all_results.extend(results)

    # ── 2. Perturbations ─────────────────────────────
    with open("configs/perturbations.yaml") as f:
        perturb_cfg = yaml.safe_load(f)

    for p in perturb_cfg["perturbations"]:
        print("=" * 60)
        print(f"PERTURBATION: {p['scenario_id']}")
        print("=" * 60)
        results = run_condition(p, output_dir)
        all_results.extend(results)

    # ── 3. Summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    # Group by scenario
    scenarios = {}
    for r in all_results:
        s = r["scenario_id"]
        if s not in scenarios:
            scenarios[s] = {"total": 0, "success": 0, "failures": {}}
        scenarios[s]["total"] += 1
        if r["success"]:
            scenarios[s]["success"] += 1
        else:
            stage = r["failure_stage"]
            scenarios[s]["failures"][stage] = scenarios[s]["failures"].get(stage, 0) + 1

    for s, info in scenarios.items():
        rate = info["success"] / info["total"] * 100
        fail_str = ", ".join(f"{k}:{v}" for k, v in info["failures"].items()) or "none"
        print(f"  {s:18s} | {info['success']}/{info['total']} success ({rate:5.1f}%) | failures: {fail_str}")

    # Save summary JSON
    summary_path = Path("data/sweep_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "scenarios": scenarios}, f, indent=2, default=str)
    print(f"\nSummary saved → {summary_path}")
    print(f"Total episodes: {len(all_results)}")


if __name__ == "__main__":
    main()
