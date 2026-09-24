"""
experiments/counterfactual.py
==============================
Phase 8: Counterfactual Diagnosis Demo

CONCEPT:
  Counterfactual = "What would have happened if I changed ONE variable?"

  Steps:
    1. Pick a failed episode (low_friction, ep000)
    2. Identify the suspected cause (friction = 0.15)
    3. Create counterfactual: keep everything else identical, restore friction = 0.8
    4. Run with the same seed → compare outcome
    5. If the failure disappears → friction is a plausible causal variable

Run:
    cd failure_sandbox
    python -m experiments.counterfactual
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json
import numpy as np
from simulation.task import PickPlaceTask, load_config
from telemetry.logger import EpisodeLogger
from evaluation.first_divergence import build_healthy_envelope, find_first_divergence
from evaluation.task_phases import load_episode


def run_episode_collect(config: dict, episode_id: str) -> tuple:
    """Run one episode, return (success, failure_stage, rows, task)."""
    task = PickPlaceTask(config)
    task.reset()
    rows = []
    for _ in range(3000):
        row = task.step()
        row["episode_id"]  = episode_id
        row["scenario_id"] = config.get("scenario_id", "unknown")
        rows.append(row)
        if task.done:
            break
    s, f = task.outcome()
    return s, f, rows, task


def print_episode_summary(label: str, success: bool, fail_stage: str,
                           rows: list, div: dict):
    phases = list(dict.fromkeys(r["task_phase"] for r in rows if r["task_phase"]))
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Result        : {'✅ SUCCESS' if success else f'❌ FAILED at [{fail_stage}]'}")
    print(f"  Steps         : {len(rows)}")
    print(f"  Phases hit    : {' → '.join(phases)}")
    if div.get("first_divergence_step") is not None:
        print(f"  First div     : step {div['first_divergence_step']} "
              f"| signal={div['first_divergence_signal']} "
              f"| phase={div['first_divergence_phase']}")
    else:
        print(f"  First div     : none detected (within normal envelope)")


def main():
    print("=" * 55)
    print("COUNTERFACTUAL DIAGNOSIS DEMO")
    print("=" * 55)
    print("""
Hypothesis: The low_friction condition (friction=0.15) caused
the slip failure at the LIFT phase.

Test: Keep everything else identical (same seed, same mass,
same gains, same delay). Restore friction to nominal (0.8).
If failure disappears → friction is the causal variable.
""")

    # ── Base config shared by both runs ──────────────────────────
    BASE_CFG = {
        "object_mass":  0.5,
        "action_delay": 0.0,
        "kp":           500.0,
        "kd":           50.0,
        "pose_offset":  0.0,
        "lift_speed":   0.3,
        "random_seed":  42,          # SAME seed for both runs
        "simulation_timestep": 0.002,
        "control_frequency":   50,
        "damping":      1.0,
        "success_distance_threshold": 0.20,
    }

    # ── Run A: Original failed condition ─────────────────────────
    cfg_a = dict(BASE_CFG, friction=0.15, scenario_id="cf_low_friction")
    print("[Running] A — low friction (friction=0.15)...")
    success_a, stage_a, rows_a, _ = run_episode_collect(cfg_a, "cf_low_friction_ep000")

    # ── Run B: Counterfactual (friction restored) ─────────────────
    cfg_b = dict(BASE_CFG, friction=0.80, scenario_id="cf_restored_friction")
    print("[Running] B — restored friction (friction=0.80)...")
    success_b, stage_b, rows_b, _ = run_episode_collect(cfg_b, "cf_restored_friction_ep000")

    # ── First-divergence analysis ─────────────────────────────────
    import pandas as pd
    envelope = build_healthy_envelope()

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    div_a = find_first_divergence(df_a, envelope) if envelope else {}
    div_b = find_first_divergence(df_b, envelope) if envelope else {}

    # ── Print comparison ──────────────────────────────────────────
    print_episode_summary("A — low_friction (friction=0.15)", success_a, stage_a, rows_a, div_a)
    print_episode_summary("B — counterfactual (friction=0.80)", success_b, stage_b, rows_b, div_b)

    # ── Causal conclusion ─────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  CAUSAL CONCLUSION")
    print(f"{'═'*55}")
    if not success_a and success_b:
        print("""
  ✅ HYPOTHESIS CONFIRMED

  Changing ONLY friction (0.15 → 0.80), keeping all else
  fixed (same seed, mass, delay, gains), eliminated the
  failure. This is strong evidence that LOW FRICTION is a
  causal variable in this failure mode.

  Physical explanation:
    max_grip_force  = μ × m × g × safety_factor
    At friction=0.15: max_grip ≈ 0.15 × 0.5 × 9.81 × 8 = 5.9N
    Required for lift: m × (g + a)  ≈ 0.5 × (9.81 + 15) ≈ 12.4N
    → Required > max_grip → SLIP (failure confirmed by physics)

    At friction=0.80: max_grip ≈ 0.8 × 0.5 × 9.81 × 8 = 31.4N
    → Required (12.4N) < max_grip → NO SLIP → SUCCESS
""")
    elif not success_a and not success_b:
        print("""
  ⚠️  HYPOTHESIS INCONCLUSIVE

  Restoring friction did not eliminate the failure.
  The root cause may be multi-variate. Try adjusting
  action_delay or pose_offset as the next hypothesis.
""")
    else:
        print("  Baseline succeeded — no failure to diagnose.")

    # ── Save counterfactual result ────────────────────────────────
    result = {
        "experiment": "counterfactual_friction",
        "hypothesis": "low_friction caused slip during lift",
        "variable_changed": "friction",
        "A_value": 0.15, "A_success": success_a, "A_failure_stage": stage_a,
        "B_value": 0.80, "B_success": success_b, "B_failure_stage": stage_b,
        "conclusion": "confirmed" if (not success_a and success_b) else "inconclusive",
        "first_div_A": div_a,
        "first_div_B": div_b,
    }
    out = Path("data/counterfactual_result.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result saved → {out}")


if __name__ == "__main__":
    main()
