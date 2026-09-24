"""
telemetry/schema.py
====================
Defines the telemetry schema — the exact columns (fields) logged per timestep.

WHAT IS A SCHEMA?
-----------------
A schema is a blueprint that says "every row of our data must have exactly
these fields." Think of it like column headers in a spreadsheet. If you don't
define this upfront, you'll end up with inconsistent data across episodes
and debugging becomes a nightmare.

WHY DOES THIS MATTER?
---------------------
When you're trying to find "first divergence" (the earliest moment a signal
went wrong), you need every episode to have identical, time-aligned columns.
This schema enforces that contract.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import time


# ── Column name constants ───────────────────────────────────────────────────
# Using constants avoids typos like "timestep" vs "time_step" vs "timestamp"
COL_TIMESTAMP          = "timestamp"          # Wall-clock time of this step
COL_EPISODE_ID         = "episode_id"         # Unique episode identifier
COL_SCENARIO_ID        = "scenario_id"        # Which config was used
COL_STEP               = "step"               # Integer step index within episode

# Robot state — joint space (q = position, qd = velocity, tau = torque/effort)
COL_Q_0                = "q_0"               # Joint 0 position (rad)
COL_Q_1                = "q_1"               # Joint 1 position (rad)
COL_Q_2                = "q_2"               # Joint 2 position (rad)
COL_Q_3                = "q_3"               # Joint 3 position (rad)
COL_Q_4                = "q_4"               # Joint 4 position (rad)
COL_Q_5                = "q_5"               # Joint 5 position (rad)
COL_QD_0               = "qd_0"             # Joint 0 velocity (rad/s)
COL_QD_1               = "qd_1"
COL_QD_2               = "qd_2"
COL_QD_3               = "qd_3"
COL_QD_4               = "qd_4"
COL_QD_5               = "qd_5"
COL_TAU_0              = "tau_0"             # Joint 0 torque (N·m)
COL_TAU_1              = "tau_1"
COL_TAU_2              = "tau_2"
COL_TAU_3              = "tau_3"
COL_TAU_4              = "tau_4"
COL_TAU_5              = "tau_5"

# Commanded targets (what we TOLD the robot to do)
COL_CMD_EEF_X          = "cmd_eef_x"         # Commanded EEF x position (m)
COL_CMD_EEF_Y          = "cmd_eef_y"         # Commanded EEF y position (m)
COL_CMD_EEF_Z          = "cmd_eef_z"         # Commanded EEF z position (m)
COL_CMD_GRIPPER        = "cmd_gripper"        # Commanded gripper (0=open, 1=closed)

# Measured end-effector state (what the robot ACTUALLY did)
COL_EEF_X              = "eef_x"             # Measured EEF x position (m)
COL_EEF_Y              = "eef_y"             # Measured EEF y position (m)
COL_EEF_Z              = "eef_z"             # Measured EEF z position (m)
COL_EEF_VX             = "eef_vx"            # EEF velocity x (m/s)
COL_EEF_VY             = "eef_vy"
COL_EEF_VZ             = "eef_vz"

# Tracking error (commanded - measured) — the gap we want to minimize
COL_TRACKING_ERR_X     = "tracking_err_x"
COL_TRACKING_ERR_Y     = "tracking_err_y"
COL_TRACKING_ERR_Z     = "tracking_err_z"
COL_TRACKING_ERR_NORM  = "tracking_err_norm"  # Euclidean distance error

# Object state (the cube being manipulated)
COL_OBJ_X              = "obj_x"             # Object x position (m)
COL_OBJ_Y              = "obj_y"             # Object y position (m)
COL_OBJ_Z              = "obj_z"             # Object z position (m)
COL_OBJ_VX             = "obj_vx"            # Object velocity x (m/s)
COL_OBJ_VY             = "obj_vy"
COL_OBJ_VZ             = "obj_vz"

# Gripper / contact state
COL_GRIPPER_STATE      = "gripper_state"      # Actual gripper opening (m)
COL_CONTACT_FORCE      = "contact_force"      # Estimated contact force magnitude (N)
COL_IN_CONTACT         = "in_contact"         # Boolean: is gripper touching object?

# Task phase label (approach/pre_grasp/contact/grasp/lift/transport/place/release)
COL_TASK_PHASE         = "task_phase"

# Scenario parameters (copied from config for easy analysis later)
COL_FRICTION           = "friction"
COL_OBJECT_MASS        = "object_mass"
COL_ACTION_DELAY       = "action_delay"
COL_KP                 = "kp"
COL_KD                 = "kd"
COL_POSE_OFFSET        = "pose_offset"
COL_LIFT_SPEED         = "lift_speed"

# Episode outcome (filled in at end of episode)
COL_SUCCESS            = "success"            # True/False
COL_FAILURE_STAGE      = "failure_stage"      # Which phase failure occurred in
COL_FIRST_DIV_TIME     = "first_divergence_t" # Time of first divergence (filled by evaluator)
COL_EPISODE_DURATION   = "episode_duration"   # Total time taken (s)


# ── All columns in order ────────────────────────────────────────────────────
ALL_COLUMNS = [
    COL_TIMESTAMP, COL_EPISODE_ID, COL_SCENARIO_ID, COL_STEP,
    COL_Q_0, COL_Q_1, COL_Q_2, COL_Q_3, COL_Q_4, COL_Q_5,
    COL_QD_0, COL_QD_1, COL_QD_2, COL_QD_3, COL_QD_4, COL_QD_5,
    COL_TAU_0, COL_TAU_1, COL_TAU_2, COL_TAU_3, COL_TAU_4, COL_TAU_5,
    COL_CMD_EEF_X, COL_CMD_EEF_Y, COL_CMD_EEF_Z, COL_CMD_GRIPPER,
    COL_EEF_X, COL_EEF_Y, COL_EEF_Z,
    COL_EEF_VX, COL_EEF_VY, COL_EEF_VZ,
    COL_TRACKING_ERR_X, COL_TRACKING_ERR_Y, COL_TRACKING_ERR_Z, COL_TRACKING_ERR_NORM,
    COL_OBJ_X, COL_OBJ_Y, COL_OBJ_Z,
    COL_OBJ_VX, COL_OBJ_VY, COL_OBJ_VZ,
    COL_GRIPPER_STATE, COL_CONTACT_FORCE, COL_IN_CONTACT,
    COL_TASK_PHASE,
    COL_FRICTION, COL_OBJECT_MASS, COL_ACTION_DELAY, COL_KP, COL_KD,
    COL_POSE_OFFSET, COL_LIFT_SPEED,
    COL_SUCCESS, COL_FAILURE_STAGE, COL_FIRST_DIV_TIME, COL_EPISODE_DURATION,
]

# Task phase names (used as string labels in COL_TASK_PHASE)
PHASES = ["approach", "pre_grasp", "contact", "grasp", "lift", "transport", "place", "release"]
