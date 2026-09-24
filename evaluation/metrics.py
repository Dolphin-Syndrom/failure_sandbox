"""
evaluation/metrics.py
======================
Aggregates all evaluation outputs into one report.

WHAT IT PRODUCES:
1. Success rate per condition
2. Failure stage distribution per condition
3. Mean tracking residuals (commanded vs measured)
4. First-divergence summary per condition
5. Failure family labels (contact/slip, control, timing, target/pose, dynamics)
6. Prints the report to stdout (also returns as dict for plotting)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.task_phases import load_episode, summarize_all_episodes
from evaluation.first_divergence import analyze_all_failed_episodes, build_healthy_envelope

# Map scenario → failure family (for human-readable labels)
FAILURE_FAMILY = {
    "nominal":      "none",
    "low_friction": "contact/slip",
    "pose_error":   "target/pose",
    "low_gain":     "control",
    "action_delay": "timing",
    "heavy_object": "dynamics",
    "fast_lift":    "contact/slip",
}


def compute_tracking_residuals(data_dir: str = "data/episodes") -> pd.DataFrame:
    """
    Mean tracking error norm per scenario.
    tracking_err_norm = |commanded_q - measured_q|  (joint space proxy)
    
    WHY: High residual = controller couldn't follow the command.
    Low residual on a failed run = the COMMAND was wrong, not execution.
    """
    records = []
    for csv in Path(data_dir).glob("*.csv"):
        df = load_episode(str(csv))
        if df.empty or "tracking_err_norm" not in df.columns:
            continue
        scenario = df["scenario_id"].iloc[0] if "scenario_id" in df.columns else "unknown"
        records.append({
            "scenario_id": scenario,
            "mean_tracking_err": df["tracking_err_norm"].mean(),
            "max_tracking_err":  df["tracking_err_norm"].max(),
            "episode_id":        csv.stem,
        })
    return pd.DataFrame(records)


def compute_scenario_stats(data_dir: str = "data/episodes") -> pd.DataFrame:
    """
    Per-scenario: success rate, failure stage counts, mean tracking error.
    """
    ep_summary = summarize_all_episodes(data_dir)
    residuals  = compute_tracking_residuals(data_dir)

    if ep_summary.empty:
        return pd.DataFrame()

    stats = []
    for scenario, grp in ep_summary.groupby("scenario_id"):
        n_total   = len(grp)
        n_success = grp["success"].sum()
        success_rate = n_success / n_total if n_total > 0 else 0

        fail_stages = grp[~grp["success"]]["failure_stage"].value_counts().to_dict()

        # Mean tracking error for this scenario
        res = residuals[residuals["scenario_id"] == scenario]["mean_tracking_err"]
        mean_res = float(res.mean()) if not res.empty else None

        stats.append({
            "scenario_id":   scenario,
            "failure_family": FAILURE_FAMILY.get(scenario, "unknown"),
            "n_total":       n_total,
            "n_success":     int(n_success),
            "success_rate":  round(success_rate * 100, 1),
            "failure_stages": fail_stages,
            "mean_tracking_err": round(mean_res, 4) if mean_res else None,
        })

    return pd.DataFrame(stats).sort_values("scenario_id")


def run_full_evaluation(data_dir: str = "data/episodes") -> dict:
    """
    Run all evaluation modules and print a human-readable report.
    Returns a dict with all results for downstream plotting.
    """
    print("=" * 65)
    print("OBLIVIQ FAILURE SANDBOX v0 — EVALUATION REPORT")
    print("=" * 65)

    # ── 1. Per-scenario stats ─────────────────────────────────
    print("\n[1] SCENARIO STATISTICS")
    print("-" * 65)
    stats = compute_scenario_stats(data_dir)
    if not stats.empty:
        for _, row in stats.iterrows():
            bar = "█" * int(row["success_rate"] / 10)
            print(f"  {row['scenario_id']:18s} | {row['n_success']}/{row['n_total']} "
                  f"({row['success_rate']:5.1f}%) {bar:<10} | "
                  f"family={row['failure_family']:15s} | "
                  f"err={row['mean_tracking_err']}")

    # ── 2. First-divergence summary ───────────────────────────
    print("\n[2] FIRST-DIVERGENCE ANALYSIS (failed episodes only)")
    print("-" * 65)
    div_df = analyze_all_failed_episodes(data_dir)
    if not div_df.empty:
        # Summarise per scenario: most common first-divergence signal and phase
        for scenario, grp in div_df.groupby("scenario_id"):
            top_signal = grp["first_div_signal"].value_counts().idxmax() if not grp["first_div_signal"].isna().all() else "N/A"
            top_phase  = grp["first_div_phase"].value_counts().idxmax()  if not grp["first_div_phase"].isna().all()  else "N/A"
            mean_step  = grp["first_div_step"].mean()
            print(f"  {scenario:18s} | first_div_signal={top_signal:22s} | "
                  f"phase={top_phase:12s} | avg_step={mean_step:.0f}")
    else:
        print("  No failed episodes found or no envelope computed.")

    # ── 3. Tracking residuals table ───────────────────────────
    print("\n[3] TRACKING RESIDUALS (mean |q_cmd - q_actual|)")
    print("-" * 65)
    res = compute_tracking_residuals(data_dir)
    if not res.empty:
        by_scenario = res.groupby("scenario_id")["mean_tracking_err"].mean().sort_values(ascending=False)
        for s, v in by_scenario.items():
            print(f"  {s:18s} | {v:.4f} rad")

    print("\n" + "=" * 65)
    print("END OF REPORT")
    print("=" * 65)

    return {
        "scenario_stats":  stats.to_dict("records") if not stats.empty else [],
        "first_divergence": div_df.to_dict("records") if not div_df.empty else [],
        "residuals":       res.groupby("scenario_id")["mean_tracking_err"].mean().to_dict() if not res.empty else {},
    }


if __name__ == "__main__":
    run_full_evaluation()
