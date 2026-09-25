"""
experiments/run_sweep.py  —  Run all experiment conditions and save telemetry CSVs.

Usage:
    cd failure_sandbox
    python experiments/run_sweep.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json
from simulation.task import PickPlaceTask, load_config
from telemetry.logger import EpisodeLogger


def run_episode(config: dict, episode_id: str, output_dir: str) -> dict:
    """Run one episode, log telemetry to CSV, return outcome summary."""
    task   = PickPlaceTask(config)
    logger = EpisodeLogger(episode_id, config.get("scenario_id", "unknown"), output_dir)
    task.reset()

    for _ in range(3000):
        row = task.step()
        row["episode_id"]  = episode_id
        row["scenario_id"] = config.get("scenario_id", "unknown")
        logger.log_step(row)
        if task.done:
            break

    success, fail_stage = task.outcome()
    logger.close(success=success, failure_stage=fail_stage)
    return {"episode_id": episode_id, "scenario_id": config.get("scenario_id"),
            "success": success, "failure_stage": fail_stage, "steps": task.step_count}


def run_condition(config: dict, output_dir: str) -> list:
    """Run N episodes for one condition with incrementing seeds."""
    n         = config.get("num_episodes", 5)
    base_seed = config.get("random_seed", 42)
    results   = []
    for i in range(n):
        cfg              = dict(config)
        cfg["random_seed"] = base_seed + i
        results.append(run_episode(cfg, f"{config['scenario_id']}_ep{i:03d}", output_dir))
    return results


def print_summary(results: list):
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    scenarios = {}
    for r in results:
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
        rate     = info["success"] / info["total"] * 100
        fail_str = ", ".join(f"{k}:{v}" for k, v in info["failures"].items()) or "none"
        print(f"  {s:18s} | {info['success']}/{info['total']} ({rate:5.1f}%) | {fail_str}")
    print(f"\nTotal episodes: {sum(s['total'] for s in scenarios.values())}")
    return scenarios


def main():
    output_dir  = "data/episodes"
    all_results = []

    # Nominal baseline
    print("=" * 60 + "\nNOMINAL BASELINE\n" + "=" * 60)
    nominal = load_config("configs/nominal.yaml")
    nominal.update({"scenario_id": "nominal", "num_episodes": 8})
    all_results.extend(run_condition(nominal, output_dir))

    # Perturbations
    with open("configs/perturbations.yaml") as f:
        perturbs = yaml.safe_load(f)["perturbations"]
    for p in perturbs:
        print("=" * 60 + f"\nPERTURBATION: {p['scenario_id']}\n" + "=" * 60)
        all_results.extend(run_condition(p, output_dir))

    scenarios = print_summary(all_results)

    # Save machine-readable summary
    out = Path("data/sweep_summary.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"results": all_results, "scenarios": scenarios}, f, indent=2, default=str)
    print(f"Summary → {out}")


if __name__ == "__main__":
    main()
