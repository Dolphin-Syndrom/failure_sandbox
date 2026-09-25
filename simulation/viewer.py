"""
simulation/viewer.py  —  Watch a single episode in the MuJoCo 3D GUI.

Usage:
    cd failure_sandbox
    python -m simulation.viewer                          # nominal run
    python -m simulation.viewer --scenario low_friction  # perturbed run
    python -m simulation.viewer --scenario fast_lift

Controls (MuJoCo viewer):
    Left-drag   Rotate camera
    Right-drag  Pan
    Scroll      Zoom
    Space       Pause / resume
    Esc         Quit
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer
import yaml
import numpy as np
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


def run_with_viewer(scenario: str = "nominal"):
    cfg  = load_scenario(scenario)
    task = PickPlaceTask(cfg)
    task.reset()

    print(f"\n[viewer] Scenario : {cfg['scenario_id']}")
    print(f"[viewer] friction={cfg.get('friction',0.8)}  "
          f"mass={cfg.get('object_mass',0.5)}  "
          f"delay={cfg.get('action_delay',0.0)}s  "
          f"kp={cfg.get('kp',500)}")
    print("[viewer] Press SPACE to pause, ESC to quit.\n")

    with mujoco.viewer.launch_passive(task.model, task.data) as v:
        while v.is_running() and not task.done:
            task.step()
            v.sync()

    s, f = task.outcome()
    print(f"\n[viewer] Result: {'✅ SUCCESS' if s else f'❌ FAILED at [{f}]'}")
    print(f"[viewer] Steps: {task.step_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a pick-and-place episode in MuJoCo GUI")
    parser.add_argument("--scenario", default="nominal",
                        help="nominal | low_friction | pose_error | low_gain | "
                             "action_delay | heavy_object | fast_lift")
    args = parser.parse_args()
    run_with_viewer(args.scenario)
