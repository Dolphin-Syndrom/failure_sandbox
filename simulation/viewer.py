"""
simulation/viewer.py  —  Watch a single episode in the MuJoCo 3D GUI.

Usage:
    cd failure_sandbox
    python -m simulation.viewer                          # nominal run
    python -m simulation.viewer --scenario low_friction  # slip failure
    python -m simulation.viewer --scenario fast_lift     # speed failure
    python -m simulation.viewer --speed 0.2              # slow-motion (0.2× real-time)

Controls (MuJoCo viewer):
    Left-drag   Rotate camera
    Right-drag  Pan
    Scroll      Zoom
    Space       Pause / resume
    Esc         Quit
"""

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer
import yaml
from simulation.task import PickPlaceTask, load_config


def load_scenario(scenario: str) -> dict:
    """Load nominal config, then override with the named perturbation if given."""
    cfg = load_config("configs/nominal.yaml")
    cfg["scenario_id"] = "nominal"

    if scenario and scenario != "nominal":
        with open("configs/perturbations.yaml") as f:
            all_perturbs = yaml.safe_load(f)["perturbations"]
        match = next((p for p in all_perturbs if p["scenario_id"] == scenario), None)
        if match is None:
            print(f"[viewer] Unknown scenario '{scenario}'. Available:")
            for p in all_perturbs:
                print(f"  {p['scenario_id']}")
            sys.exit(1)
        cfg.update(match)

    return cfg


def run_with_viewer(scenario: str = "nominal", speed: float = 0.5):
    cfg  = load_scenario(scenario)
    task = PickPlaceTask(cfg)
    task.reset()

    dt = cfg.get("simulation_timestep", 0.002)  # seconds per physics step
    render_interval = dt / speed                  # wall-clock seconds per step

    print(f"\n{'='*50}")
    print(f"  MuJoCo Viewer — {cfg['scenario_id']}")
    print(f"{'='*50}")
    print(f"  friction={cfg.get('friction',0.8)}  mass={cfg.get('object_mass',0.5)}  "
          f"delay={cfg.get('action_delay',0.0)}s  kp={cfg.get('kp',500)}")
    print(f"  Playback speed: {speed}× real-time")
    print(f"  Controls: SPACE=pause  ESC=quit")
    print(f"{'='*50}\n")

    with mujoco.viewer.launch_passive(task.model, task.data) as v:
        # ── Run episode at controlled speed ──
        while v.is_running() and not task.done:
            step_start = time.time()

            task.step()
            v.sync()

            # Sleep to maintain real-time pacing
            elapsed    = time.time() - step_start
            sleep_time = render_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ── Print result ──
        s, f = task.outcome()
        result = '✅ SUCCESS' if s else f'❌ FAILED at [{f}]'
        print(f"\n  Result : {result}")
        print(f"  Steps  : {task.step_count}")

        # ── Hold window open so user can inspect final state ──
        if v.is_running():
            print("  (Window stays open — press ESC to close)")
            while v.is_running():
                v.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a pick-and-place episode in MuJoCo GUI")
    parser.add_argument("--scenario", default="nominal",
                        help="nominal | low_friction | pose_error | low_gain | "
                             "action_delay | heavy_object | fast_lift")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed multiplier (default=0.5, i.e. half real-time)")
    args = parser.parse_args()
    run_with_viewer(args.scenario, args.speed)

