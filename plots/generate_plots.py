"""
plots/generate_plots.py
========================
Generates all 6 required plots from episode telemetry CSVs.

Run:
    cd failure_sandbox
    python -m plots.generate_plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from evaluation.task_phases import load_episode
from evaluation.first_divergence import build_healthy_envelope, find_first_divergence

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

# Color palette per scenario
COLORS = {
    "nominal":      "#4CAF50",
    "low_friction": "#F44336",
    "pose_error":   "#FF9800",
    "low_gain":     "#9C27B0",
    "action_delay": "#2196F3",
    "heavy_object": "#795548",
    "fast_lift":    "#E91E63",
}

def load_one(scenario: str, ep_idx: int = 0) -> pd.DataFrame:
    p = Path(f"data/episodes/{scenario}_ep{ep_idx:03d}.csv")
    return load_episode(str(p)) if p.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Commanded vs Measured Trajectory (joint tracking)
# ─────────────────────────────────────────────────────────────────────────────
def plot_commanded_vs_measured():
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Plot 1 — Commanded vs Measured Joint Positions (nominal)", fontsize=13, fontweight="bold")

    df = load_one("nominal", 0)
    if df.empty:
        plt.close(); return

    steps = df["step"]
    joints = [("q_0", "cmd_eef_x", "Joint 0 (yaw)"),
              ("q_1", "cmd_eef_y", "Joint 1 (shoulder)"),
              ("q_2", "cmd_eef_z", "Joint 2 (elbow)")]

    for ax, (measured_col, cmd_col, label) in zip(axes, joints):
        ax.plot(steps, df[measured_col], color="#4CAF50", lw=1.5, label="Measured")
        ax.plot(steps, df[cmd_col],      color="#F44336", lw=1.2, ls="--", label="Commanded")
        ax.set_ylabel(f"{label} (rad)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        # Shade task phases
        _shade_phases(ax, df)

    axes[-1].set_xlabel("Step", fontsize=10)
    plt.tight_layout()
    p = OUT / "plot1_commanded_vs_measured.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Object Position & Velocity vs Time
# ─────────────────────────────────────────────────────────────────────────────
def plot_object_motion():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Plot 2 — Object Position & Velocity vs Time", fontsize=13, fontweight="bold")

    scenarios = ["nominal", "low_friction", "fast_lift", "pose_error"]
    for s in scenarios:
        df = load_one(s)
        if df.empty: continue
        c = COLORS.get(s, "#888")
        axes[0].plot(df["step"], df["obj_z"],  color=c, lw=1.5, label=s, alpha=0.85)
        axes[1].plot(df["step"], df["obj_vz"], color=c, lw=1.2, alpha=0.85)

    axes[0].axhline(0.447, color="gray", ls=":", lw=1, label="table_z")
    axes[0].set_ylabel("Cube Z position (m)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Cube Z velocity (m/s)", fontsize=10)
    axes[1].set_xlabel("Step", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = OUT / "plot2_object_motion.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Tracking Error vs Time (all scenarios)
# ─────────────────────────────────────────────────────────────────────────────
def plot_tracking_error():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Plot 3 — Joint Tracking Error vs Time (all conditions)", fontsize=13, fontweight="bold")

    all_scenarios = ["nominal", "low_friction", "pose_error", "low_gain",
                     "action_delay", "heavy_object", "fast_lift"]
    for s in all_scenarios:
        df = load_one(s)
        if df.empty or "tracking_err_norm" not in df.columns: continue
        ax.plot(df["step"], df["tracking_err_norm"],
                color=COLORS.get(s, "#888"), lw=1.4, label=s, alpha=0.85)

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("|q_cmd − q_actual| (rad)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT / "plot3_tracking_error.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Contact / Gripper Force + First-Divergence Marker
# ─────────────────────────────────────────────────────────────────────────────
def plot_contact_and_divergence():
    envelope = build_healthy_envelope()
    if not envelope:
        print("  [Skip] No envelope — plot 4 skipped"); return

    # Show 3 contrasting scenarios side-by-side
    scenarios = ["low_friction", "fast_lift", "action_delay"]
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Plot 4 — Contact Force & First-Divergence Marker", fontsize=13, fontweight="bold")

    for ax, s in zip(axes, scenarios):
        df = load_one(s)
        if df.empty: continue
        div = find_first_divergence(df, envelope)

        ax.plot(df["step"], df["in_contact"],      color=COLORS[s], lw=1.5, label="in_contact", alpha=0.9)
        ax.plot(df["step"], df["tracking_err_norm"] / 5,
                color="#607D8B", lw=1.0, ls="--", label="tracking_err/5", alpha=0.7)

        # First-divergence vertical line
        if div["first_divergence_step"] is not None:
            fds = div["first_divergence_step"]
            ax.axvline(fds, color="red", lw=2, ls="--")
            ax.text(fds + 5, ax.get_ylim()[1] * 0.85,
                    f"↑ first div\n(step {fds})\n{div['first_divergence_signal']}",
                    color="red", fontsize=7.5)

        _shade_phases(ax, df)
        ax.set_title(f"{s}  (failure_stage={df['failure_stage'].dropna().iloc[-1] if not df['failure_stage'].dropna().empty else '?'})",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("value", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Step", fontsize=10)
    plt.tight_layout()
    p = OUT / "plot4_contact_divergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: All-scenario overlaid comparison (obj_z)
# ─────────────────────────────────────────────────────────────────────────────
def plot_scenario_overlay():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Plot 5 — Scenario Overlay: Cube Z & EEF Z", fontsize=13, fontweight="bold")

    all_scenarios = ["nominal", "low_friction", "pose_error", "low_gain",
                     "action_delay", "heavy_object", "fast_lift"]

    for s in all_scenarios:
        df = load_one(s)
        if df.empty: continue
        c   = COLORS.get(s, "#888")
        ls  = "-" if s == "nominal" else "--"
        lw  = 2.0 if s == "nominal" else 1.2
        axes[0].plot(df["step"], df["obj_z"],  color=c, lw=lw, ls=ls, label=s, alpha=0.85)
        axes[1].plot(df["step"], df["eef_z"],  color=c, lw=lw, ls=ls, alpha=0.85)

    axes[0].axhline(0.447, color="black", ls=":", lw=1.0, label="table_z")
    axes[0].set_ylabel("Cube Z (m)", fontsize=10)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("EEF Z (m)", fontsize=10)
    axes[1].set_xlabel("Step", fontsize=10)
    axes[1].grid(alpha=0.3)

    # Add legend patches for scenarios
    patches = [mpatches.Patch(color=COLORS[s], label=s) for s in all_scenarios]
    axes[1].legend(handles=patches, fontsize=7, ncol=2)

    plt.tight_layout()
    p = OUT / "plot5_scenario_overlay.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: 2D Failure Boundary (friction × lift_speed)
# Run a mini-sweep varying two parameters simultaneously
# ─────────────────────────────────────────────────────────────────────────────
def plot_2d_failure_boundary():
    """
    2D parameter sweep: friction (x-axis) × lift_speed (y-axis)
    Each cell = success rate over 2 mini-episodes.
    Color: green=success, red=failure.
    """
    from simulation.task import PickPlaceTask

    frictions   = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    lift_speeds  = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    n_per_cell  = 2   # 2 episodes per cell → 72 total

    print(f"  Running 2D sweep ({len(frictions)}×{len(lift_speeds)}×{n_per_cell} = "
          f"{len(frictions)*len(lift_speeds)*n_per_cell} episodes)...")

    grid = np.zeros((len(lift_speeds), len(frictions)))

    for fi, fr in enumerate(frictions):
        for li, ls in enumerate(lift_speeds):
            successes = 0
            for seed in range(n_per_cell):
                cfg = {
                    "scenario_id": "sweep",
                    "friction": fr, "object_mass": 0.5,
                    "action_delay": 0.0, "kp": 500.0, "kd": 50.0,
                    "pose_offset": 0.0, "lift_speed": ls,
                    "random_seed": 42 + seed,
                    "simulation_timestep": 0.002,
                    "control_frequency": 50,
                    "success_distance_threshold": 0.20,
                    "damping": 1.0,
                }
                task = PickPlaceTask(cfg)
                task.reset()
                for _ in range(3000):
                    task.step()
                    if task.done: break
                s, _ = task.outcome()
                successes += int(s)
            grid[li, fi] = successes / n_per_cell

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[frictions[0]-0.05, frictions[-1]+0.05,
                           lift_speeds[0]-0.05, lift_speeds[-1]+0.05])

    plt.colorbar(im, ax=ax, label="Success rate")
    ax.set_xlabel("Friction coefficient", fontsize=11)
    ax.set_ylabel("Lift speed (m/s)", fontsize=11)
    ax.set_title("Plot 6 — 2D Failure Boundary: Friction × Lift Speed", fontsize=12, fontweight="bold")
    ax.set_xticks(frictions)
    ax.set_yticks(lift_speeds)

    # Annotate cells
    for li, ls in enumerate(lift_speeds):
        for fi, fr in enumerate(frictions):
            val = grid[li, fi]
            ax.text(fr, ls, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="black" if 0.3 < val < 0.7 else "white")

    plt.tight_layout()
    p = OUT / "plot6_2d_failure_boundary.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: shade background by task phase
# ─────────────────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "approach":   "#E3F2FD", "pre_grasp":  "#E8F5E9",
    "contact":    "#FFF9C4", "grasp":      "#FFF3E0",
    "lift":       "#FCE4EC", "transport":  "#F3E5F5",
    "place":      "#E0F7FA", "release":    "#EFEBE9",
}

def _shade_phases(ax, df: pd.DataFrame):
    """Add light background shading for each task phase."""
    if "task_phase" not in df.columns:
        return
    phase_col = df["task_phase"].fillna("unknown")
    changes = phase_col[phase_col != phase_col.shift()].index.tolist()
    changes.append(len(df))
    for i, start_idx in enumerate(changes[:-1]):
        end_idx = changes[i + 1]
        phase   = phase_col.iloc[start_idx]
        color   = PHASE_COLORS.get(phase, "#FAFAFA")
        x_start = df["step"].iloc[start_idx]
        x_end   = df["step"].iloc[min(end_idx, len(df)-1)]
        ax.axvspan(x_start, x_end, alpha=0.15, color=color, linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating all plots...")
    plot_commanded_vs_measured()
    plot_object_motion()
    plot_tracking_error()
    plot_contact_and_divergence()
    plot_scenario_overlay()
    plot_2d_failure_boundary()
    print(f"\nAll plots saved to {OUT.resolve()}/")
