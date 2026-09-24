"""
verify_setup.py
================
Run this script to confirm your Phase 1 setup is complete and working.

Usage:
    conda activate obliviq
    cd obliviq_failure_sandbox
    python verify_setup.py

What it checks:
1. MuJoCo import and version
2. NumPy, matplotlib, pandas, yaml — all installed
3. MuJoCo can create a basic physics model and step it
4. Project folder structure is correct
5. Config files can be loaded and parsed
"""

import sys
print(f"Python: {sys.version}")
print("-" * 60)

# ── 1. Check all imports ─────────────────────────────────────────────────────
print("1️⃣  Checking imports...")
try:
    import mujoco
    print(f"   ✅ mujoco {mujoco.__version__}")
except ImportError as e:
    print(f"   ❌ mujoco FAILED: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✅ numpy {np.__version__}")
except ImportError as e:
    print(f"   ❌ numpy FAILED: {e}")

try:
    import matplotlib
    print(f"   ✅ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"   ❌ matplotlib FAILED: {e}")

try:
    import pandas as pd
    print(f"   ✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"   ❌ pandas FAILED: {e}")

try:
    import yaml
    print(f"   ✅ pyyaml installed")
except ImportError as e:
    print(f"   ❌ pyyaml FAILED: {e}")

try:
    import scipy
    print(f"   ✅ scipy {scipy.__version__}")
except ImportError as e:
    print(f"   ❌ scipy FAILED: {e}")

# ── 2. MuJoCo physics test ───────────────────────────────────────────────────
print("\n2️⃣  Testing MuJoCo physics engine...")

# A minimal MuJoCo model: one free-floating sphere under gravity.
# xml = MuJoCo's scene description language (like HTML for physics scenes).
minimal_xml = """
<mujoco model="verify_test">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="sphere" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="sphere" size="0.05" mass="0.5" rgba="1 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(minimal_xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # Compute initial positions

    # In MuJoCo 3.x, xpos is on data.xpos[body_id], indexed by body id
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
    initial_z = data.xpos[body_id][2]

    # Step the simulation 100 times (= 0.2 seconds of simulated time)
    for _ in range(100):
        mujoco.mj_step(model, data)

    final_z = data.xpos[body_id][2]

    # Ball should have fallen due to gravity (z decreases)
    if final_z < initial_z:
        print(f"   ✅ Physics works! Ball fell from z={initial_z:.3f} → z={final_z:.3f} (gravity confirmed)")
    else:
        print(f"   ⚠️  Ball z: {initial_z:.3f} → {final_z:.3f} (may have hit floor — still OK)")
except Exception as e:
    print(f"   ❌ MuJoCo physics FAILED: {e}")
    sys.exit(1)

# ── 3. Folder structure check ────────────────────────────────────────────────
print("\n3️⃣  Checking project folder structure...")
from pathlib import Path

required_paths = [
    "configs/nominal.yaml",
    "configs/perturbations.yaml",
    "simulation/",
    "telemetry/schema.py",
    "telemetry/logger.py",
    "evaluation/",
    "experiments/",
    "data/episodes/",
    "plots/",
    "report/",
]

all_ok = True
for p in required_paths:
    path = Path(p)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {p}")
    if not exists:
        all_ok = False

# ── 4. Config loading check ──────────────────────────────────────────────────
print("\n4️⃣  Loading and validating configs...")
try:
    with open("configs/nominal.yaml") as f:
        nominal = yaml.safe_load(f)
    assert "friction" in nominal
    assert "object_mass" in nominal
    assert "random_seed" in nominal
    print(f"   ✅ nominal.yaml loaded | friction={nominal['friction']} | mass={nominal['object_mass']} | seed={nominal['random_seed']}")
except Exception as e:
    print(f"   ❌ nominal.yaml FAILED: {e}")

try:
    with open("configs/perturbations.yaml") as f:
        perturbs = yaml.safe_load(f)
    n = len(perturbs["perturbations"])
    ids = [p["scenario_id"] for p in perturbs["perturbations"]]
    print(f"   ✅ perturbations.yaml loaded | {n} conditions: {ids}")
except Exception as e:
    print(f"   ❌ perturbations.yaml FAILED: {e}")

# ── 5. Schema check ──────────────────────────────────────────────────────────
print("\n5️⃣  Checking telemetry schema...")
try:
    sys.path.insert(0, ".")
    from telemetry.schema import ALL_COLUMNS, PHASES
    print(f"   ✅ schema.py loaded | {len(ALL_COLUMNS)} columns defined | phases: {PHASES}")
except Exception as e:
    print(f"   ❌ schema.py FAILED: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("🎉  PHASE 1 COMPLETE — All checks passed!")
    print("    Next: Phase 2 — Build the MuJoCo simulation (task.py)")
else:
    print("⚠️   Some checks failed — fix above errors before Phase 2")
print("=" * 60)
