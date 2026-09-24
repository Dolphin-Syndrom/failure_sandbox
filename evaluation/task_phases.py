"""
evaluation/task_phases.py
==========================
Labels each timestep with a task phase and computes phase-level statistics.

WHAT IT DOES:
The task.py already writes 'task_phase' per row during simulation.
This module reads that column and provides:
  - Phase sequence (which phases were reached)
  - Time spent in each phase
  - Which phase the failure occurred in
"""

import pandas as pd
from pathlib import Path

PHASE_ORDER = ["approach", "pre_grasp", "contact", "grasp",
               "lift", "transport", "place", "release", "done"]


def load_episode(csv_path: str) -> pd.DataFrame:
    """Load one episode CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    # Convert boolean-ish columns
    df["success"] = df["success"].map({"True": True, "False": False, True: True, False: False})
    df["in_contact"] = df["in_contact"].astype(float)
    return df


def get_phase_summary(df: pd.DataFrame) -> dict:
    """
    Return a dict describing phase progression for this episode.
    
    KEY CONCEPT: Task phases let you say "the failure happened during LIFT"
    not just "the task failed." This is the first step in localizing failure.
    """
    phases_hit   = list(dict.fromkeys(df["task_phase"].dropna().tolist()))  # ordered unique
    failure_stage = df["failure_stage"].dropna().iloc[-1] if not df["failure_stage"].dropna().empty else "none"
    success       = bool(df["success"].dropna().iloc[-1]) if not df["success"].dropna().empty else False

    # Steps per phase
    phase_steps = df.groupby("task_phase")["step"].count().to_dict()

    return {
        "phases_hit":    phases_hit,
        "failure_stage": failure_stage,
        "success":       success,
        "phase_steps":   phase_steps,
        "total_steps":   len(df),
    }


def summarize_all_episodes(data_dir: str = "data/episodes") -> pd.DataFrame:
    """
    Load all episode CSVs and return a summary DataFrame.
    One row per episode with: scenario, success, failure_stage, phases_hit.
    """
    records = []
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        df  = load_episode(str(csv_path))
        if df.empty:
            continue
        summary = get_phase_summary(df)
        records.append({
            "episode_id":    csv_path.stem,
            "scenario_id":   df["scenario_id"].iloc[0] if "scenario_id" in df.columns else "unknown",
            "success":       summary["success"],
            "failure_stage": summary["failure_stage"],
            "phases_hit":    " → ".join(summary["phases_hit"]),
            "total_steps":   summary["total_steps"],
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = summarize_all_episodes()
    print(df.to_string(index=False))
