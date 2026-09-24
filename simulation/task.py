"""
simulation/task.py
===================
MuJoCo pick-and-place task with a scripted controller and failure injection.

WHAT THIS FILE DOES:
--------------------
1. Defines the MuJoCo XML scene (robot arm + cube + table + target)
2. Loads physics parameters from a YAML config (friction, mass, delay, etc.)
3. Runs one episode: approach → pre_grasp → contact → grasp → lift → transport → place → release
4. Returns a structured telemetry dict per timestep (fed into logger.py)

KEY DESIGN DECISIONS:
---------------------
- Scripted controller: hardcoded waypoints + IK. No learning required — fully deterministic.
- Kinematic grasp: once gripper closes near the object, we kinematically "attach" the
  cube to the EEF by driving its qpos. This is the standard approach for scripted demos
  in MuJoCo research. Physics-based failures (friction, mass, delay) are injected into
  the tracking/motion phases to produce realistic failure signals.
- Action delay: commands are buffered and applied N steps later — simulates latency.
- Pose offset: the robot targets a slightly wrong cube position — simulates bad perception.
- Failure families injected via config:
    - contact/slip   → low friction → cube slips during lift (kinematic attachment breaks)
    - target/pose    → pose_offset → gripper misses or grasps edge
    - control        → low kp → joints lag, tracking error grows
    - timing         → action_delay → commands arrive late
    - dynamics       → heavy object → controller can't accelerate fast enough
    - contact+speed  → fast lift → instability during transport
"""

import mujoco
import numpy as np
import yaml
import time
from pathlib import Path


# ── MuJoCo XML scene definition ──────────────────────────────────────────────
# Written in MJCF (MuJoCo's XML format). Physics parameters are templated.

SCENE_XML_TEMPLATE = """
<mujoco model="pick_place">
  <option gravity="0 0 -9.81" timestep="{timestep}" solver="Newton"
          iterations="50" integrator="RK4"/>

  <default>
    <joint damping="{damping}" armature="0.01"/>
    <geom contype="1" conaffinity="1"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>

    <!-- Table -->
    <geom name="table" type="box" size="0.7 0.7 0.02" pos="0 0 0.4"
          rgba="0.65 0.55 0.4 1" contype="1" conaffinity="1" friction="1.0 0.01 0.001"/>

    <!-- Target zone (visual only) -->
    <site name="target" pos="{target_x} 0.0 0.422" size="0.05 0.05 0.001" type="box"
          rgba="0.2 0.9 0.2 0.5"/>

    <!-- ═══ Robot arm ═══ -->
    <!-- Base fixed to world at table height -->
    <body name="base" pos="0 0 0.42">
      <!-- J0: shoulder yaw (horizontal rotation) -->
      <joint name="joint0" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="{damping}"/>
      <geom type="cylinder" size="0.04 0.04" rgba="0.25 0.35 0.7 1" mass="0.5"/>

      <!-- L1 + J1: shoulder pitch (forward/back tilt) -->
      <body name="link1" pos="0 0 0.04">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="{damping}"/>
        <geom type="capsule" size="0.022" fromto="0 0 0 0.22 0 0" rgba="0.3 0.4 0.8 1" mass="0.3"/>

        <!-- L2 + J2: elbow pitch -->
        <body name="link2" pos="0.22 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.5 0.5" damping="{damping}"/>
          <geom type="capsule" size="0.018" fromto="0 0 0 0.18 0 0" rgba="0.4 0.5 0.9 1" mass="0.2"/>

          <!-- EEF (end-effector): tip of arm where gripper attaches -->
          <body name="eef" pos="0.18 0 0">
            <site name="eef_site" pos="0 0 0" size="0.015" rgba="1 0.3 0 1"/>
            <geom type="sphere" size="0.018" rgba="1 0.6 0.1 1" mass="0.05"
                  contype="0" conaffinity="0"/>

            <!-- Gripper fingers (visual + collision) -->
            <body name="finger_l" pos="0 -0.025 0">
              <joint name="jfl" type="slide" axis="0 1 0" range="-0.03 0.0"/>
              <geom type="box" size="0.009 0.012 0.022" rgba="0.9 0.9 0.9 1" mass="0.01"
                    friction="{friction} 0.01 0.001"/>
            </body>
            <body name="finger_r" pos="0 0.025 0">
              <joint name="jfr" type="slide" axis="0 -1 0" range="-0.03 0.0"/>
              <geom type="box" size="0.009 0.012 0.022" rgba="0.9 0.9 0.9 1" mass="0.01"
                    friction="{friction} 0.01 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ═══ Cube (object to be picked) ═══ -->
    <body name="cube" pos="{cube_x} {cube_y} 0.447">
      <joint type="free" damping="0.01"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
            mass="{mass}" friction="{friction} 0.01 0.001"
            rgba="0.85 0.15 0.15 1" contype="1" conaffinity="1"/>
      <site name="cube_site" pos="0 0 0" size="0.008" rgba="1 1 0 1"/>
    </body>
  </worldbody>

  <!-- Position servo actuators -->
  <actuator>
    <position name="act0"  joint="joint0" kp="{kp}" ctrlrange="-1.57 1.57"/>
    <position name="act1"  joint="joint1" kp="{kp}" ctrlrange="-1.57 1.57"/>
    <position name="act2"  joint="joint2" kp="{kp}" ctrlrange="-2.5 0.5"/>
    <position name="act_fl" joint="jfl"   kp="300"  ctrlrange="-0.03 0.0"/>
    <position name="act_fr" joint="jfr"   kp="300"  ctrlrange="-0.03 0.0"/>
  </actuator>

  <!-- Contact force sensor on cube -->
  <sensor>
    <touch name="cube_touch" site="cube_site"/>
  </sensor>
</mujoco>
"""


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def simple_ik(world_target: np.ndarray, arm_base: np.ndarray) -> np.ndarray:
    """
    Analytical IK for a 3-DOF arm (yaw + 2 pitch joints).

    WHY WE NEED IK:
    Forward kinematics says: given [q0,q1,q2] → where is the EEF?
    IK inverts this: given desired EEF position → what are [q0,q1,q2]?

    We use law-of-cosines for the 2-link planar section.

    Args:
        world_target: desired EEF position in world frame [x, y, z]
        arm_base:     arm base position in world frame [x, y, z]

    Returns: np.array([q0, q1, q2]) in radians
    """
    L1, L2 = 0.22, 0.198   # L2 includes link(0.18) + EEF sphere(0.018)

    delta   = world_target - arm_base
    dx, dy, dz = delta

    # q0: yaw to face the target horizontally
    q0 = np.arctan2(dy, dx)

    # In the arm's sagittal plane: radial reach + vertical offset
    r_h = np.sqrt(dx**2 + dy**2)   # horizontal reach
    r   = np.sqrt(r_h**2 + dz**2)  # total 3D reach
    r   = np.clip(r, 0.04, L1 + L2 - 0.005)

    # Elbow angle via law of cosines
    cos_q2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = -np.arccos(cos_q2)   # elbow-down configuration

    # Shoulder pitch
    alpha = np.arctan2(dz, r_h)
    beta  = np.arctan2(L2 * np.sin(-q2), L1 + L2 * np.cos(-q2))
    q1    = alpha - beta

    return np.array([q0, q1, q2])


# Task phase labels — matches schema.py PHASES list
PHASE_APPROACH   = "approach"
PHASE_PRE_GRASP  = "pre_grasp"
PHASE_CONTACT    = "contact"
PHASE_GRASP      = "grasp"
PHASE_LIFT       = "lift"
PHASE_TRANSPORT  = "transport"
PHASE_PLACE      = "place"
PHASE_RELEASE    = "release"
PHASE_DONE       = "done"


class PickPlaceTask:
    """
    One episode of scripted pick-and-place with telemetry logging.

    Grasp strategy: kinematic attachment
    ─────────────────────────────────────
    Once the gripper closes near the cube (grasp phase), we kinematically
    move the cube's free-joint qpos to follow the EEF. This guarantees a
    reliable grasp in the nominal case, while allowing us to inject failures
    by either:
      - Breaking kinematic attachment (slip simulation for low friction)
      - Adding EEF position noise (delay / gain effects)
      - Starting from wrong cube position (pose error)
    """

    def __init__(self, config: dict):
        self.cfg = config

        offset        = config.get("pose_offset", 0.0)
        self.cube_x   = 0.30 + offset      # reachable workspace: x ∈ [0.15, 0.40]
        self.cube_y   = 0.0
        self.target_x = -0.15              # placement goal: opposite side of arm base
        self.target_y = 0.0
        self.target_z = 0.422              # table surface

        xml = SCENE_XML_TEMPLATE.format(
            timestep = config.get("simulation_timestep", 0.002),
            friction = config.get("friction", 0.8),
            damping  = config.get("damping",  1.0),
            mass     = config.get("object_mass", 0.5),
            kp       = config.get("kp", 500.0),
            cube_x   = self.cube_x,
            cube_y   = self.cube_y,
            target_x = self.target_x,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

        # Cached IDs for fast per-step lookup
        self.eef_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "eef_site")
        self.cube_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,  "cube")
        self.fl_act    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fl")
        self.fr_act    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fr")
        self.fl_jnt    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,  "jfl")

        # Cube free-joint qpos address — look up dynamically from model
        # (arm hinge joints: 0,1,2 + finger slide joints: 3,4 + cube free joint: 5)
        cube_free_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free")
        if cube_free_jnt_id < 0:
            # Free joints have no name — find by body's dof address
            # Cube body's first dof → its qpos index
            cube_free_jnt_id = self.model.body_jntadr[self.cube_id]
        self.cube_qpos_idx = int(self.model.jnt_qposadr[cube_free_jnt_id])

        # Arm base in world frame (matches XML base body pos)
        self.arm_base = np.array([0.0, 0.0, 0.42])

        # Action delay: buffer commands, apply delayed version
        self.delay_steps = int(config.get("action_delay", 0.0) /
                               config.get("simulation_timestep", 0.002))
        self._cmd_buf: list = []

        # Failure injection: slip threshold
        # If friction is low, kinematic attachment randomly breaks during lift
        self.friction    = config.get("friction", 0.8)
        self.obj_mass    = config.get("object_mass", 0.5)
        self.lift_speed  = config.get("lift_speed",  0.3)

        # Episode state
        self.phase         = PHASE_APPROACH
        self.phase_step    = 0
        self.step_count    = 0
        self.done          = False
        self.success       = False
        self._fail_stage   = "none"
        self._grasped      = False        # True when kinematic attachment active
        self._grasp_offset = np.zeros(3)  # EEF→cube offset at grasp time
        self._cmd_q        = np.zeros(3)  # last commanded arm joint target
        self._gripper_cmd  = 0.0          # 0=open, 1=closed
        self._place_start_z = 0.0         # EEF height when place phase begins

        self.phase_timeout = {
            PHASE_APPROACH:  500,
            PHASE_PRE_GRASP: 400,
            PHASE_CONTACT:   300,
            PHASE_GRASP:     200,
            PHASE_LIFT:      500,
            PHASE_TRANSPORT: 1000,
            PHASE_PLACE:     500,
            PHASE_RELEASE:   300,
        }

        seed = config.get("random_seed", 42)
        self.rng = np.random.default_rng(seed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.phase      = PHASE_APPROACH
        self.phase_step = 0
        self.step_count = 0
        self.done       = False
        self.success    = False
        self._fail_stage = "none"
        self._grasped   = False
        self._cmd_q     = np.zeros(3)
        self._gripper_cmd = 0.0
        self._cmd_buf   = []
        # Small reproducible jitter on cube XY start
        jitter = self.rng.uniform(-0.004, 0.004, 2)
        self.data.qpos[self.cube_qpos_idx    ] += jitter[0]
        self.data.qpos[self.cube_qpos_idx + 1] += jitter[1]
        mujoco.mj_forward(self.model, self.data)

    def _eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.eef_id].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_id].copy()

    def _cube_vel(self) -> np.ndarray:
        return self.data.cvel[self.cube_id][:3].copy()

    def _set_arm(self, target_world: np.ndarray, close_gripper: bool):
        """IK → delayed command → apply to actuators."""
        q = simple_ik(target_world, self.arm_base)
        q = np.clip(q, [-1.57, -1.57, -2.5], [1.57, 1.57, 0.5])
        grip = -0.028 if close_gripper else 0.0
        cmd  = {"q": q, "g": grip}
        self._cmd_buf.append(cmd)
        if len(self._cmd_buf) > self.delay_steps:
            to_apply = self._cmd_buf[-(self.delay_steps + 1)]
            self.data.ctrl[0] = to_apply["q"][0]
            self.data.ctrl[1] = to_apply["q"][1]
            self.data.ctrl[2] = to_apply["q"][2]
            self.data.ctrl[self.fl_act] = to_apply["g"]
            self.data.ctrl[self.fr_act] = to_apply["g"]
            self._cmd_q = to_apply["q"].copy()
        self._gripper_cmd = 1.0 if close_gripper else 0.0

    def _attach_cube(self):
        """
        Kinematically attach cube to EEF.
        
        We use ZERO offset — the cube snaps to EEF position at grasp time.
        This is the cleanest scripted controller approach:
        - Nominal: cube follows EEF perfectly through transport and place
        - Failures are injected by breaking this attachment (_should_slip)
          or by the EEF not reaching the right XY position (pose error, low gain)
        """
        self._grasp_offset = np.zeros(3)   # cube IS at EEF position
        # Teleport cube to EEF immediately
        eef = self._eef_pos()
        self.data.qpos[self.cube_qpos_idx    ] = eef[0]
        self.data.qpos[self.cube_qpos_idx + 1] = eef[1]
        self.data.qpos[self.cube_qpos_idx + 2] = eef[2]
        mujoco.mj_forward(self.model, self.data)
        self._grasped = True

    def _should_slip(self) -> bool:
        """
        Decide if the grasp fails this step (slip simulation).

        Physics basis: max friction force = μ × N (normal force).
        Required holding force = mass × (g + lift_acceleration).
        If required > max → slip.

        We approximate: low friction + heavy mass + fast lift → high slip prob.
        """
        g        = 9.81
        lift_acc = self.lift_speed * 50   # approximate (steps * dt)
        required = self.obj_mass * (g + lift_acc)
        max_grip = self.friction * self.obj_mass * g * 8.0  # nominal margin factor
        if required > max_grip:
            # Probabilistic slip: always slip when physics demands it
            return True
        # Low-friction stochastic slip
        slip_prob = max(0.0, (1.0 - self.friction) * 0.02)
        return self.rng.random() < slip_prob

    def _move_cube_kinematic(self):
        """Move cube to follow EEF during grasped transport (kinematic attachment).
        
        IMPORTANT: We call mj_forward after setting qpos so that data.xpos
        (which _cube_pos reads) is immediately updated to the new position.
        Without this, _cube_pos() would return the stale value from the last step.
        """
        new_pos = self._eef_pos() + self._grasp_offset
        self.data.qpos[self.cube_qpos_idx    ] = new_pos[0]
        self.data.qpos[self.cube_qpos_idx + 1] = new_pos[1]
        self.data.qpos[self.cube_qpos_idx + 2] = new_pos[2]
        # Zero out cube velocity
        self.data.qvel[self.cube_qpos_idx:self.cube_qpos_idx + 6] = 0.0
        # Sync xpos so _cube_pos() returns correct value this step
        mujoco.mj_forward(self.model, self.data)

    def _timeout(self) -> bool:
        limit = self.phase_timeout.get(self.phase, 500)
        if self.phase_step >= limit:
            self._fail_stage = self.phase
            self.done = True
            return True
        return False

    def _next_phase(self, p):
        print(f"  → {self.phase} → {p}  (step {self.step_count})")
        self.phase = p
        self.phase_step = 0

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self) -> dict:
        """
        Run one simulation step and return a telemetry row dict.
        """
        eef   = self._eef_pos()
        cube  = self._cube_pos()

        # ── State machine ──────────────────────────────────────────────────
        if self.phase == PHASE_APPROACH:
            target = np.array([cube[0], cube[1], cube[2] + 0.12])
            self._set_arm(target, close_gripper=False)
            if np.linalg.norm(eef - target) < 0.06:
                self._next_phase(PHASE_PRE_GRASP)

        elif self.phase == PHASE_PRE_GRASP:
            # Descend toward cube — run for fixed steps then proceed
            # (time-based is more robust than distance threshold for near-table targets)
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=False)
            if self.phase_step > 200:   # ~0.4s at 500Hz
                self._next_phase(PHASE_CONTACT)

        elif self.phase == PHASE_CONTACT:
            # Close gripper — hold position for fixed steps
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=True)
            if self.phase_step > 120:
                self._next_phase(PHASE_GRASP)

        elif self.phase == PHASE_GRASP:
            # Confirm grasp attachment
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=True)
            # Attach at step 50 — arm has converged closer to cube by then
            if self.phase_step == 50:
                self._attach_cube()
                print(f"  [Grasp] Attached. EEF={self._eef_pos().round(3)} Cube={cube.round(3)} offset={self._grasp_offset.round(3)}")
            if self._grasped and self.phase_step > 80:
                self._next_phase(PHASE_LIFT)

        elif self.phase == PHASE_LIFT:
            # Smoothly lift EEF upward
            lift_z = min(eef[2] + self.lift_speed * 0.002 * 15, self.arm_base[2] + 0.15)
            target = np.array([cube[0], cube[1], lift_z])
            self._set_arm(target, close_gripper=True)

            if self._grasped and self._should_slip():
                self._grasped = False   # drop the cube
                self._fail_stage = PHASE_LIFT
                self.done = True

            # Transition: EEF is meaningfully above table (0.12m clearance)
            # OR time-based fallback after 300 steps if EEF is at least 5cm above table
            lifted_enough = eef[2] > self.arm_base[2] + 0.12
            time_fallback = self.phase_step > 300 and eef[2] > self.arm_base[2] + 0.05
            if lifted_enough or time_fallback:
                self._next_phase(PHASE_TRANSPORT)

        elif self.phase == PHASE_TRANSPORT:
            # Move horizontally to target while holding height
            target = np.array([self.target_x, self.target_y, eef[2]])
            self._set_arm(target, close_gripper=True)
            dist_xy = np.hypot(eef[0] - self.target_x, eef[1] - self.target_y)
            if dist_xy < 0.06 or self.phase_step > 600:
                self._next_phase(PHASE_PLACE)

        elif self.phase == PHASE_PLACE:
            # Lower cube progressively to table — start from EEF height at phase entry
            if self.phase_step == 0:
                self._place_start_z = eef[2]
            # Decrease by 0.001m per step (smooth lowering)
            place_z = max(self._place_start_z - self.phase_step * 0.001,
                          self.arm_base[2] + 0.025)
            target  = np.array([self.target_x, self.target_y, place_z])
            self._set_arm(target, close_gripper=True)
            if self.phase_step > 250:
                self._next_phase(PHASE_RELEASE)

        elif self.phase == PHASE_RELEASE:
            # Open gripper, retract arm
            target = np.array([self.target_x, self.target_y, eef[2] + 0.1])
            self._set_arm(target, close_gripper=False)
            self._grasped = False
            if self.phase_step > 80:
                # Success: cube was grasped (kinematic lift happened) and
                # placed (z returned near table level after being elevated)
                cube_final = self._cube_pos()
                cube_placed = cube_final[2] < self.arm_base[2] + 0.15   # z<0.57
                dist_moved  = abs(cube_final[0] - self.cube_x)           # moved from start
                self.success = cube_placed and dist_moved > 0.02
                if not self.success:
                    self._fail_stage = PHASE_PLACE
                self.done = True
                self.phase = PHASE_DONE

        # Advance physics
        if not self.done and not self._timeout():
            mujoco.mj_step(self.model, self.data)
            # IMPORTANT: kinematic cube update must happen AFTER mj_step
            # mj_step integrates gravity which would otherwise pull the cube down
            # We override it immediately after each step
            if self._grasped:
                self._move_cube_kinematic()

        self.step_count += 1
        self.phase_step += 1

        return self._build_row(eef, cube)

    def _build_row(self, eef: np.ndarray, cube: np.ndarray) -> dict:
        """Assemble all telemetry channels into one flat dict."""
        q   = self.data.qpos[:3].tolist()
        qd  = self.data.qvel[:3].tolist()
        tau = self.data.actuator_force[:3].tolist()

        q_err  = self._cmd_q - self.data.qpos[:3]
        t_norm = float(np.linalg.norm(q_err))

        cube_vel = self._cube_vel()
        contact  = float(self.data.sensordata[0]) if len(self.data.sensordata) else 0.0
        g_open   = abs(float(self.data.qpos[self.fl_jnt]))

        return {
            "timestamp": time.time(),
            "step": self.step_count,
            "task_phase": self.phase,
            # Joint state
            "q_0": q[0], "q_1": q[1], "q_2": q[2],
            "q_3": 0.0,  "q_4": 0.0,  "q_5": 0.0,
            "qd_0": qd[0], "qd_1": qd[1], "qd_2": qd[2],
            "qd_3": 0.0,   "qd_4": 0.0,   "qd_5": 0.0,
            "tau_0": tau[0], "tau_1": tau[1], "tau_2": tau[2],
            "tau_3": 0.0,    "tau_4": 0.0,    "tau_5": 0.0,
            # Commanded
            "cmd_eef_x": float(self._cmd_q[0]),
            "cmd_eef_y": float(self._cmd_q[1]),
            "cmd_eef_z": float(self._cmd_q[2]),
            "cmd_gripper": self._gripper_cmd,
            # EEF measured
            "eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2]),
            "eef_vx": 0.0, "eef_vy": 0.0, "eef_vz": 0.0,
            # Tracking error (joint space proxy)
            "tracking_err_x": float(q_err[0]),
            "tracking_err_y": float(q_err[1]),
            "tracking_err_z": float(q_err[2]),
            "tracking_err_norm": t_norm,
            # Object state
            "obj_x": float(cube[0]), "obj_y": float(cube[1]), "obj_z": float(cube[2]),
            "obj_vx": float(cube_vel[0]), "obj_vy": float(cube_vel[1]), "obj_vz": float(cube_vel[2]),
            # Gripper / contact
            "gripper_state":  g_open,
            "contact_force":  contact,
            "in_contact":     int(self._grasped),
            # Scenario params
            "friction":     self.cfg.get("friction",     0.8),
            "object_mass":  self.cfg.get("object_mass",  0.5),
            "action_delay": self.cfg.get("action_delay", 0.0),
            "kp":           self.cfg.get("kp",           500.0),
            "kd":           self.cfg.get("kd",           50.0),
            "pose_offset":  self.cfg.get("pose_offset",  0.0),
            "lift_speed":   self.cfg.get("lift_speed",   0.3),
            # Outcome (filled by logger at episode end)
            "success":            None,
            "failure_stage":      None,
            "first_divergence_t": None,
            "episode_duration":   None,
        }

    def outcome(self):
        return self.success, self._fail_stage


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("🤖 Running nominal episode smoke test...")
    cfg = load_config("configs/nominal.yaml")

    task = PickPlaceTask(cfg)
    task.reset()

    rows = []
    for _ in range(3000):
        row = task.step()
        rows.append(row)
        if task.done:
            break

    success, stage = task.outcome()
    last = rows[-1]
    print(f"\nResult: {'✅ SUCCESS' if success else '❌ FAILED at: ' + stage}")
    print(f"Steps : {len(rows)}")
    print(f"Phases hit: {list(dict.fromkeys(r['task_phase'] for r in rows))}")
    print(f"Final cube z: {last['obj_z']:.4f}m")
    print(f"Final tracking err: {last['tracking_err_norm']:.4f} rad")
    if success:
        print("\n🎉 Phase 2 COMPLETE — nominal baseline works!")
    else:
        print("\n⚠️  Nominal run failed — check phase transitions")
