"""
evaluation/first_divergence.py
================================
Finds the EARLIEST timestep where a failed episode's signals leave normal range.

CONCEPT:
- "First divergence" = the root cause window (not the final symptom)
- Normal range is computed from the NOMINAL baseline episodes (mean ± 2σ)
- For each failed episode, scan each signal left-to-right and find the first
  timestep where any signal exits the normal band
- That timestamp is when the failure BEGAN, not when it was detected

WHY THIS MATTERS:
A slip during lift might be detected at t=2.0s (cube hits floor),
but first divergence may be at t=1.3s (tracking_err_norm spikes).
That 700ms gap is where you look for root cause.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.task_phases import load_episode

# Signals to monitor for divergence
# These are the most informative channels for detecting failures early
MONITORED_SIGNALS = [
    "tracking_err_norm",   # joint tracking error — bad control or delay
    "obj_z",               # cube height — detects drop/slip
    "obj_vz",              # cube vertical velocity — sudden fall
    "contact_force",       # gripper contact — detects slip
    "in_contact",          # binary contact state
    "eef_z",               # EEF height — control failure indicator
]


def build_healthy_envelope(nominal_dir: str = "data/episodes",
                           scenario: str = "nominal") -> dict:
    """
    Compute mean ± 2σ of each signal from nominal baseline episodes.
    
    WHY 2σ: Under a normal distribution, 95% of values fall within 2σ.
    Anything outside is statistically unusual — a candidate divergence.
    
    Returns dict: { signal_name: {"mean": Series, "std": Series} }
    """
    dfs = []
    for csv in Path(nominal_dir).glob(f"{scenario}_ep*.csv"):
        df = load_episode(str(csv))
        dfs.append(df)

    if not dfs:
        print(f"[Warning] No nominal episodes found in {nominal_dir}")
        return {}

    envelope = {}
    for sig in MONITORED_SIGNALS:
        if sig not in dfs[0].columns:
            continue
        # Stack all nominal runs and compute step-wise stats
        all_vals = pd.concat([df[sig].reset_index(drop=True) for df in dfs], axis=1)
        envelope[sig] = {
            "mean": all_vals.mean(axis=1),
            "std":  all_vals.std(axis=1).fillna(0),
        }
    return envelope


def find_first_divergence(df: pd.DataFrame, envelope: dict,
                          sigma_threshold: float = 2.5) -> dict:
    """
    Scan a failed episode and find the earliest timestep where any
    monitored signal exits the normal envelope (mean ± sigma_threshold * std).

    Args:
        df:              Episode DataFrame (one row per step)
        envelope:        Healthy range from build_healthy_envelope()
        sigma_threshold: How many σ beyond mean counts as divergence

    Returns:
        {
          "first_divergence_step":      int or None,
          "first_divergence_timestamp": float or None,
          "first_divergence_signal":    str or None,
          "first_divergence_phase":     str or None,
        }
    """
    earliest_step = None
    earliest_signal = None

    for sig, stats in envelope.items():
        if sig not in df.columns:
            continue

        mean = stats["mean"]
        std  = stats["std"]

        # Align envelope length to episode length (may differ)
        n = min(len(df), len(mean))
        values = df[sig].iloc[:n].values
        lo = (mean.iloc[:n] - sigma_threshold * std.iloc[:n]).values
        hi = (mean.iloc[:n] + sigma_threshold * std.iloc[:n]).values

        # Floor hi-lo to avoid zero-width bands on constant signals
        band_min = 0.01
        lo = np.minimum(lo, mean.iloc[:n].values - band_min)
        hi = np.maximum(hi, mean.iloc[:n].values + band_min)

        # Find first index outside band
        outside = np.where((values < lo) | (values > hi))[0]
        if len(outside) > 0:
            first = int(outside[0])
            if earliest_step is None or first < earliest_step:
                earliest_step   = first
                earliest_signal = sig

    if earliest_step is None:
        return {
            "first_divergence_step":      None,
            "first_divergence_timestamp": None,
            "first_divergence_signal":    None,
            "first_divergence_phase":     None,
        }

    first_div_ts = float(df["timestamp"].iloc[earliest_step]) if "timestamp" in df.columns else None
    first_div_phase = str(df["task_phase"].iloc[earliest_step]) if "task_phase" in df.columns else None

    return {
        "first_divergence_step":      earliest_step,
        "first_divergence_timestamp": first_div_ts,
        "first_divergence_signal":    earliest_signal,
        "first_divergence_phase":     first_div_phase,
    }


def analyze_all_failed_episodes(data_dir: str = "data/episodes") -> pd.DataFrame:
    """
    Run first-divergence analysis on every failed episode.
    Returns a DataFrame with one row per failed episode.
    """
    envelope = build_healthy_envelope(data_dir)
    if not envelope:
        print("Cannot compute envelope — no nominal episodes found.")
        return pd.DataFrame()

    records = []
    for csv in sorted(Path(data_dir).glob("*.csv")):
        df = load_episode(str(csv))
        if df.empty:
            continue

        # Skip nominal and successful episodes
        scenario = df["scenario_id"].iloc[0] if "scenario_id" in df.columns else "unknown"
        success  = df["success"].dropna().iloc[-1] if not df["success"].dropna().empty else True
        if scenario == "nominal" or success is True or success == "True":
            continue

        div = find_first_divergence(df, envelope)
        failure_stage = str(df["failure_stage"].dropna().iloc[-1]) if not df["failure_stage"].dropna().empty else "none"

        records.append({
            "episode_id":             csv.stem,
            "scenario_id":            scenario,
            "failure_stage":          failure_stage,
            "first_div_step":         div["first_divergence_step"],
            "first_div_signal":       div["first_divergence_signal"],
            "first_div_phase":        div["first_divergence_phase"],
            "total_steps":            len(df),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Building healthy envelope from nominal episodes...")
    results = analyze_all_failed_episodes()
    if not results.empty:
        print("\nFirst-Divergence Analysis:")
        print(results.to_string(index=False))
    else:
        print("No failed episodes found.")
