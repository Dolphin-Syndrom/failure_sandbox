# Week 1 Failure Analysis — Obliviq Failure Debugging Sandbox v0

## Overview

This report documents the failure analysis performed on a simulated 3-DOF robot arm
performing pick-and-place using MuJoCo 3.14. Six failure modes were injected, and
one failure was diagnosed end-to-end via counterfactual testing.

---

## 1. System Description

- **Robot**: 3-DOF arm (yaw + 2 pitch joints), scripted IK controller
- **Task**: Pick a cube at x=0.30m, lift, transport, place
- **Controller**: Analytical IK + position servos (kp=500, kd=50)
- **Grasp**: Kinematic attachment (cube follows EEF after grasp phase)
- **Simulator**: MuJoCo 3.14.0, timestep=2ms, 500Hz physics

---

## 2. Experiment Conditions

| Condition | Changed Variable | Value |
|-----------|-----------------|-------|
| nominal | — | baseline |
| low_friction | friction | 0.15 (nominal: 0.8) |
| pose_error | pose_offset | +0.06m (nominal: 0) |
| low_gain | kp / kd | 30 / 3 (nominal: 500 / 50) |
| action_delay | action_delay | 0.15s (nominal: 0) |
| heavy_object | object_mass | 8.0kg (nominal: 0.5kg) |
| fast_lift | lift_speed | 2.0 m/s (nominal: 0.3) |

---

## 3. Results Summary

| Condition | Success Rate | Failure Stage | First Div Signal |
|-----------|-------------|---------------|-----------------|
| nominal | 8/8 (100%) | — | — |
| low_friction | 0/7 (0%) | lift | in_contact (step 417) |
| pose_error | 0/7 (0%) | approach | tracking_err_norm (step 0) |
| low_gain | 0/7 (0%) | approach | tracking_err_norm (step 0) |
| action_delay | 4/7 (57%) | place | tracking_err_norm (step 0) |
| heavy_object | 0/7 (0%) | approach | obj_vz (step 3) |
| fast_lift | 0/7 (0%) | lift | tracking_err_norm (step 417) |

---

## 4. Failure Analysis: `low_friction` (Primary Case Study)

### What failed
- **Stage**: LIFT — the arm grasped the cube but it immediately slipped during the lift motion.
- **First divergence**: Step 417 (the first step of the lift phase), signal = `in_contact`.
  The contact state changed from 1→0 immediately at lift onset.

### Why it failed (physical reasoning)

The Coulomb friction model gives:

```
max_holding_force = μ × N = μ × m × g × grip_factor
                  = 0.15 × 0.5 × 9.81 × 8 ≈ 5.9 N

required_force    = m × (g + a_lift)
                  = 0.5 × (9.81 + 15.0) ≈ 12.4 N
```

Since `required > max_holding → slip occurs`. The gripper cannot generate enough
friction to overcome gravity + lift acceleration. This is a **contact/slip** failure.

### Evidence from telemetry
- `in_contact` drops from 1 to 0 at step 417 (first lift step)
- `obj_vz` shows no upward velocity (cube stays on table after release)
- `tracking_err_norm` is stable (arm moved correctly — the command was fine)
- This tells us: **execution was correct, physics was the cause**

---

## 5. Counterfactual Test

**Hypothesis**: Low friction (μ=0.15) is the causal variable.

**Test design**:
- Run A: friction=0.15, seed=42, all else nominal → `❌ FAILED at lift`
- Run B: friction=0.80, seed=42, all else nominal → `✅ SUCCESS`

**Result**: Changing ONLY friction eliminated the failure. All other variables
(mass, delay, gains, pose, seed) were identical.

**Conclusion**: ✅ Friction is a **confirmed causal variable** for this failure mode.

This rules out:
- Delay (same delay=0 in both)
- Gains (same kp=500 in both)
- Pose error (same offset=0 in both)
- Random variation (same seed)

---

## 6. Observations Across All Conditions

### approach failures (pose_error, low_gain, heavy_object)
- All three fail at step 0–3 of approach.
- `tracking_err_norm` diverges immediately for low_gain and pose_error
  → the arm cannot converge to the pre-grasp position.
- `obj_vz` diverges at step 3 for heavy_object → the cube's physics differ
  immediately (extra inertia causes bounce on table).
- **Key insight**: These are all approach-time failures. The arm never even
  reaches the cube. Root cause is in the command target or joint response.

### lift failures (low_friction, fast_lift)
- Both fail at step 417 (exact start of lift phase).
- `in_contact` drops to 0 for low_friction; `tracking_err_norm` spikes for fast_lift.
- **Key distinction**: low_friction = can't hold the cube; fast_lift = arm moves
  too fast, slip threshold exceeded by acceleration (same physics, different driver).

### timing failure (action_delay)
- Partial failure: 4/7 succeed, 3/7 fail at place.
- 57% success shows the failure is **probabilistic** — delay sometimes matters,
  sometimes not, depending on whether the arm's buffered command lands near the
  right position during placement.
- This is a **timing-sensitivity** failure: the system works if timing is lucky,
  fails if it isn't.

---

## 7. Key Learnings

| Learning | Evidence |
|----------|----------|
| Failure stage ≠ failure cause | low_friction fails at lift, but the cause is physics (friction) set at config time |
| First divergence precedes final outcome | `in_contact` drops at step 417 before the episode is "done" |
| Commanded vs measured separates control from physics | tracking_err is normal in low_friction — arm was fine, physics wasn't |
| Counterfactual confirms causation | Restoring one variable while holding all else fixed eliminates the failure |
| Partial failures need more episodes | action_delay's 57% rate needs more episodes for statistical confidence |

---

## 8. Reproducibility

Every result in this report is reproduced by:

```bash
git clone https://github.com/Dolphin-Syndrom/failure_sandbox
cd failure_sandbox
conda create -n sandbox python=3.11 -y && conda activate sandbox
pip install -r requirements.txt
python experiments/run_sweep.py       # reproduces all 50 episodes
python -m experiments.counterfactual  # reproduces the counterfactual
python -m evaluation.metrics          # reproduces the report table
python -m plots.generate_plots        # reproduces all 6 figures
```

All random seeds are pinned in `configs/perturbations.yaml`.
Physics engine: MuJoCo 3.14.0 (version in `requirements.txt`).
