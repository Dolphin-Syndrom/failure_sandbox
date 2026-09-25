"""
simulation/task.py  —  MuJoCo pick-and-place with scripted controller & failure injection.

8-phase state machine: approach → pre_grasp → contact → grasp → lift → transport → place → release

Failure modes (injected via YAML config):
  low_friction   contact/slip   cube slips during lift
  pose_error     target/pose    gripper targets wrong position
  low_gain       control        joints too sluggish to reach target
  action_delay   timing         commands arrive N steps late
  heavy_object   dynamics       too heavy for controller to lift
  fast_lift      contact/speed  acceleration exceeds grip force
"""

import mujoco
import numpy as np
import yaml
import time
from pathlib import Path


# ── Scene XML (MJCF) ──────────────────────────────────────────────────────────

SCENE_XML = """
<mujoco model="pick_place">
  <option gravity="0 0 -9.81" timestep="{timestep}" solver="Newton"
          iterations="50" integrator="RK4"/>

  <default>
    <joint damping="{damping}" armature="0.01"/>
    <geom contype="1" conaffinity="1"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>

    <!-- Table surface -->
    <geom name="table" type="box" size="0.7 0.7 0.02" pos="0 0 0.4"
          rgba="0.65 0.55 0.4 1" friction="1.0 0.01 0.001"/>

    <!-- Placement target zone (visual only) -->
    <site name="target" pos="{target_x} 0.0 0.422" size="0.05 0.05 0.001"
          type="box" rgba="0.2 0.9 0.2 0.5"/>

    <!-- 3-DOF robot arm: yaw (J0) + shoulder pitch (J1) + elbow pitch (J2) -->
    <body name="base" pos="0 0 0.42">
      <joint name="joint0" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
      <geom type="cylinder" size="0.04 0.04" rgba="0.25 0.35 0.7 1" mass="0.5"/>

      <body name="link1" pos="0 0 0.04">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.022" fromto="0 0 0 0.22 0 0"
              rgba="0.3 0.4 0.8 1" mass="0.3"/>

        <body name="link2" pos="0.22 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.5 0.5"/>
          <geom type="capsule" size="0.018" fromto="0 0 0 0.18 0 0"
                rgba="0.4 0.5 0.9 1" mass="0.2"/>

          <!-- End-effector body -->
          <body name="eef" pos="0.18 0 0">
            <site name="eef_site" pos="0 0 0" size="0.015" rgba="1 0.3 0 1"/>
            <geom type="sphere" size="0.018" rgba="1 0.6 0.1 1" mass="0.05"
                  contype="0" conaffinity="0"/>

            <!-- Gripper fingers -->
            <body name="finger_l" pos="0 -0.025 0">
              <joint name="jfl" type="slide" axis="0 1 0" range="-0.03 0.0"/>
              <geom type="box" size="0.009 0.012 0.022" rgba="0.9 0.9 0.9 1"
                    mass="0.01" friction="{friction} 0.01 0.001"/>
            </body>
            <body name="finger_r" pos="0 0.025 0">
              <joint name="jfr" type="slide" axis="0 -1 0" range="-0.03 0.0"/>
              <geom type="box" size="0.009 0.012 0.022" rgba="0.9 0.9 0.9 1"
                    mass="0.01" friction="{friction} 0.01 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Free-floating cube (object to pick) -->
    <body name="cube" pos="{cube_x} {cube_y} 0.447">
      <joint type="free" damping="0.01"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
            mass="{mass}" friction="{friction} 0.01 0.001" rgba="0.85 0.15 0.15 1"/>
      <site name="cube_site" pos="0 0 0" size="0.008" rgba="1 1 0 1"/>
    </body>
  </worldbody>

  <!-- Position servo actuators for arm joints and fingers -->
  <actuator>
    <position name="act0"   joint="joint0" kp="{kp}"  ctrlrange="-1.57 1.57"/>
    <position name="act1"   joint="joint1" kp="{kp}"  ctrlrange="-1.57 1.57"/>
    <position name="act2"   joint="joint2" kp="{kp}"  ctrlrange="-2.5 0.5"/>
    <position name="act_fl" joint="jfl"    kp="300"   ctrlrange="-0.03 0.0"/>
    <position name="act_fr" joint="jfr"    kp="300"   ctrlrange="-0.03 0.0"/>
  </actuator>

  <!-- Touch sensor on cube to detect gripper contact -->
  <sensor>
    <touch name="cube_touch" site="cube_site"/>
  </sensor>
</mujoco>
"""


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Analytical IK ─────────────────────────────────────────────────────────────

def simple_ik(world_target: np.ndarray, arm_base: np.ndarray) -> np.ndarray:
    """
    Analytical IK for the 3-DOF arm: yaw + 2 pitch joints.
    Uses law-of-cosines for the 2-link planar section.
    Returns [q0, q1, q2] in radians.
    """
    L1, L2 = 0.22, 0.198   # link lengths (L2 includes EEF sphere radius)

    delta       = world_target - arm_base
    dx, dy, dz  = delta
    q0          = np.arctan2(dy, dx)

    r_h = np.sqrt(dx**2 + dy**2)
    r   = np.clip(np.sqrt(r_h**2 + dz**2), 0.04, L1 + L2 - 0.005)

    cos_q2 = np.clip((r**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    q2     = -np.arccos(cos_q2)                                   # elbow-down

    alpha = np.arctan2(dz, r_h)
    beta  = np.arctan2(L2 * np.sin(-q2), L1 + L2 * np.cos(-q2))
    q1    = alpha - beta

    return np.array([q0, q1, q2])


# ── Phase constants ───────────────────────────────────────────────────────────

PHASE_APPROACH   = "approach"
PHASE_PRE_GRASP  = "pre_grasp"
PHASE_CONTACT    = "contact"
PHASE_GRASP      = "grasp"
PHASE_LIFT       = "lift"
PHASE_TRANSPORT  = "transport"
PHASE_PLACE      = "place"
PHASE_RELEASE    = "release"
PHASE_DONE       = "done"


# ── Task class ────────────────────────────────────────────────────────────────

class PickPlaceTask:
    """
    One episode of the pick-and-place task.

    Grasp is kinematic: after the grasp phase the cube's free-joint qpos
    is set to follow the EEF every step, overriding gravity. This gives a
    reliable nominal baseline. Failures are injected via config:
      - slip  → low friction triggers _should_slip() → attachment broken
      - delay → command buffer shifts arm position relative to cube
      - gain  → joints too slow to reach target within phase timeout
    """

    # Phase timeouts (in simulation steps)
    PHASE_TIMEOUT = {
        PHASE_APPROACH:  500,
        PHASE_PRE_GRASP: 400,
        PHASE_CONTACT:   300,
        PHASE_GRASP:     200,
        PHASE_LIFT:      500,
        PHASE_TRANSPORT: 1000,
        PHASE_PLACE:     500,
        PHASE_RELEASE:   300,
    }

    def __init__(self, config: dict):
        self.cfg = config

        # Cube start position (pose_offset simulates perception error)
        self.cube_x  = 0.30 + config.get("pose_offset", 0.0)
        self.cube_y  = 0.0
        self.target_x = -0.15   # placement target (opposite side of base)
        self.target_y =  0.0
        self.target_z =  0.422  # table surface

        # Build MuJoCo model from templated XML
        xml = SCENE_XML.format(
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

        # Cached body/site/actuator IDs
        self.eef_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,     "eef_site")
        self.cube_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,     "cube")
        self.fl_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fl")
        self.fr_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_fr")
        self.fl_jnt   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    "jfl")

        # Cube free-joint qpos start index (dynamic lookup — finger joints come first)
        cube_jnt_id       = self.model.body_jntadr[self.cube_id]
        self.cube_qpos_idx = int(self.model.jnt_qposadr[cube_jnt_id])

        self.arm_base   = np.array([0.0, 0.0, 0.42])
        self.delay_steps = int(config.get("action_delay", 0.0) /
                               config.get("simulation_timestep", 0.002))

        # Failure injection parameters
        self.friction   = config.get("friction",    0.8)
        self.obj_mass   = config.get("object_mass", 0.5)
        self.lift_speed = config.get("lift_speed",  0.3)

        self.rng = np.random.default_rng(config.get("random_seed", 42))
        self._reset_state()

    def _reset_state(self):
        self.phase        = PHASE_APPROACH
        self.phase_step   = 0
        self.step_count   = 0
        self.done         = False
        self.success      = False
        self._fail_stage  = "none"
        self._grasped     = False
        self._cmd_q       = np.zeros(3)
        self._gripper_cmd = 0.0
        self._place_start_z = 0.0
        self._cmd_buf     = []

    def reset(self):
        """Reset physics and episode state. Call once before running."""
        mujoco.mj_resetData(self.model, self.data)
        self._reset_state()
        # Small reproducible jitter on cube XY
        jitter = self.rng.uniform(-0.004, 0.004, 2)
        self.data.qpos[self.cube_qpos_idx    ] += jitter[0]
        self.data.qpos[self.cube_qpos_idx + 1] += jitter[1]
        mujoco.mj_forward(self.model, self.data)

    # ── State accessors ───────────────────────────────────────────────────────

    def _eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.eef_id].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_id].copy()

    def _cube_vel(self) -> np.ndarray:
        return self.data.cvel[self.cube_id][:3].copy()

    # ── Controller helpers ────────────────────────────────────────────────────

    def _set_arm(self, target: np.ndarray, close_gripper: bool):
        """IK → action-delay buffer → write to actuator controls."""
        q    = np.clip(simple_ik(target, self.arm_base),
                       [-1.57, -1.57, -2.5], [1.57, 1.57, 0.5])
        grip = -0.028 if close_gripper else 0.0
        self._cmd_buf.append({"q": q, "g": grip})

        if len(self._cmd_buf) > self.delay_steps:
            cmd = self._cmd_buf[-(self.delay_steps + 1)]
            self.data.ctrl[:3]          = cmd["q"]
            self.data.ctrl[self.fl_act] = cmd["g"]
            self.data.ctrl[self.fr_act] = cmd["g"]
            self._cmd_q = cmd["q"].copy()

        self._gripper_cmd = 1.0 if close_gripper else 0.0

    def _attach_cube(self):
        """Snap cube to EEF position (kinematic grasp)."""
        eef = self._eef_pos()
        self.data.qpos[self.cube_qpos_idx    ] = eef[0]
        self.data.qpos[self.cube_qpos_idx + 1] = eef[1]
        self.data.qpos[self.cube_qpos_idx + 2] = eef[2]
        mujoco.mj_forward(self.model, self.data)
        self._grasped = True

    def _move_cube_with_eef(self):
        """Move cube to match EEF position every step (after mj_step)."""
        eef = self._eef_pos()
        self.data.qpos[self.cube_qpos_idx    ] = eef[0]
        self.data.qpos[self.cube_qpos_idx + 1] = eef[1]
        self.data.qpos[self.cube_qpos_idx + 2] = eef[2]
        self.data.qvel[self.cube_qpos_idx:self.cube_qpos_idx + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _should_slip(self) -> bool:
        """
        Physics-based slip check: slip if friction force < required holding force.
          max_grip  = μ × m × g × safety_factor
          required  = m × (g + lift_acceleration)
        """
        g    = 9.81
        lift_acc  = self.lift_speed * 50
        required  = self.obj_mass * (g + lift_acc)
        max_grip  = self.friction * self.obj_mass * g * 8.0
        if required > max_grip:
            return True
        return self.rng.random() < max(0.0, (1.0 - self.friction) * 0.02)

    def _next_phase(self, phase: str):
        print(f"  → {self.phase} → {phase}  (step {self.step_count})")
        self.phase      = phase
        self.phase_step = 0

    def _check_timeout(self) -> bool:
        if self.phase_step >= self.PHASE_TIMEOUT.get(self.phase, 500):
            self._fail_stage = self.phase
            self.done = True
            return True
        return False

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self) -> dict:
        """Run one physics step; return telemetry row dict."""
        eef  = self._eef_pos()
        cube = self._cube_pos()

        # ── State machine ─────────────────────────────────────────────────────
        if self.phase == PHASE_APPROACH:
            target = np.array([cube[0], cube[1], cube[2] + 0.12])
            self._set_arm(target, close_gripper=False)
            if np.linalg.norm(eef - target) < 0.06:
                self._next_phase(PHASE_PRE_GRASP)

        elif self.phase == PHASE_PRE_GRASP:
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=False)
            if self.phase_step > 200:
                self._next_phase(PHASE_CONTACT)

        elif self.phase == PHASE_CONTACT:
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=True)
            if self.phase_step > 120:
                self._next_phase(PHASE_GRASP)

        elif self.phase == PHASE_GRASP:
            target = np.array([cube[0], cube[1], cube[2] + 0.01])
            self._set_arm(target, close_gripper=True)
            if self.phase_step == 50:
                self._attach_cube()
            if self._grasped and self.phase_step > 80:
                self._next_phase(PHASE_LIFT)

        elif self.phase == PHASE_LIFT:
            lift_z = min(eef[2] + self.lift_speed * 0.002 * 15,
                         self.arm_base[2] + 0.15)
            self._set_arm(np.array([cube[0], cube[1], lift_z]),
                          close_gripper=True)
            if self._grasped and self._should_slip():
                self._grasped    = False
                self._fail_stage = PHASE_LIFT
                self.done        = True
            lifted  = eef[2] > self.arm_base[2] + 0.12
            timeout = self.phase_step > 300 and eef[2] > self.arm_base[2] + 0.05
            if lifted or timeout:
                self._next_phase(PHASE_TRANSPORT)

        elif self.phase == PHASE_TRANSPORT:
            self._set_arm(np.array([self.target_x, self.target_y, eef[2]]),
                          close_gripper=True)
            dist_xy = np.hypot(eef[0] - self.target_x, eef[1] - self.target_y)
            if dist_xy < 0.06 or self.phase_step > 600:
                self._next_phase(PHASE_PLACE)

        elif self.phase == PHASE_PLACE:
            if self.phase_step == 0:
                self._place_start_z = eef[2]
            place_z = max(self._place_start_z - self.phase_step * 0.001,
                          self.arm_base[2] + 0.025)
            self._set_arm(np.array([self.target_x, self.target_y, place_z]),
                          close_gripper=True)
            if self.phase_step > 250:
                self._next_phase(PHASE_RELEASE)

        elif self.phase == PHASE_RELEASE:
            self._set_arm(np.array([self.target_x, self.target_y, eef[2] + 0.1]),
                          close_gripper=False)
            self._grasped = False
            if self.phase_step > 80:
                cube_f      = self._cube_pos()
                placed      = cube_f[2] < self.arm_base[2] + 0.15
                moved       = abs(cube_f[0] - self.cube_x) > 0.02
                self.success = placed and moved
                if not self.success:
                    self._fail_stage = PHASE_PLACE
                self.done  = True
                self.phase = PHASE_DONE

        # ── Physics step + kinematic cube attachment ───────────────────────────
        if not self.done and not self._check_timeout():
            mujoco.mj_step(self.model, self.data)
            if self._grasped:
                self._move_cube_with_eef()

        self.step_count += 1
        self.phase_step += 1
        return self._build_row(eef, cube)

    def _build_row(self, eef: np.ndarray, cube: np.ndarray) -> dict:
        """Pack all telemetry channels into one flat dict."""
        q    = self.data.qpos[:3].tolist()
        qd   = self.data.qvel[:3].tolist()
        tau  = self.data.actuator_force[:3].tolist()
        q_err = self._cmd_q - self.data.qpos[:3]
        return {
            "timestamp":          time.time(),
            "step":               self.step_count,
            "task_phase":         self.phase,
            "q_0": q[0],   "q_1": q[1],   "q_2": q[2],
            "q_3": 0.0,    "q_4": 0.0,    "q_5": 0.0,
            "qd_0": qd[0], "qd_1": qd[1], "qd_2": qd[2],
            "qd_3": 0.0,   "qd_4": 0.0,   "qd_5": 0.0,
            "tau_0": tau[0], "tau_1": tau[1], "tau_2": tau[2],
            "tau_3": 0.0,    "tau_4": 0.0,    "tau_5": 0.0,
            "cmd_eef_x":   float(self._cmd_q[0]),
            "cmd_eef_y":   float(self._cmd_q[1]),
            "cmd_eef_z":   float(self._cmd_q[2]),
            "cmd_gripper": self._gripper_cmd,
            "eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2]),
            "eef_vx": 0.0, "eef_vy": 0.0, "eef_vz": 0.0,
            "tracking_err_x":    float(q_err[0]),
            "tracking_err_y":    float(q_err[1]),
            "tracking_err_z":    float(q_err[2]),
            "tracking_err_norm": float(np.linalg.norm(q_err)),
            "obj_x": float(cube[0]), "obj_y": float(cube[1]), "obj_z": float(cube[2]),
            "obj_vx": float(self._cube_vel()[0]),
            "obj_vy": float(self._cube_vel()[1]),
            "obj_vz": float(self._cube_vel()[2]),
            "gripper_state":  abs(float(self.data.qpos[self.fl_jnt])),
            "contact_force":  float(self.data.sensordata[0]) if len(self.data.sensordata) else 0.0,
            "in_contact":     int(self._grasped),
            "friction":       self.cfg.get("friction",     0.8),
            "object_mass":    self.cfg.get("object_mass",  0.5),
            "action_delay":   self.cfg.get("action_delay", 0.0),
            "kp":             self.cfg.get("kp",           500.0),
            "kd":             self.cfg.get("kd",           50.0),
            "pose_offset":    self.cfg.get("pose_offset",  0.0),
            "lift_speed":     self.cfg.get("lift_speed",   0.3),
            "success":             None,
            "failure_stage":       None,
            "first_divergence_t":  None,
            "episode_duration":    None,
        }

    def outcome(self) -> tuple:
        return self.success, self._fail_stage


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg  = load_config("configs/nominal.yaml")
    task = PickPlaceTask(cfg)
    task.reset()
    rows = []
    print("Running nominal episode...")
    for _ in range(3000):
        rows.append(task.step())
        if task.done:
            break
    s, f = task.outcome()
    phases = list(dict.fromkeys(r["task_phase"] for r in rows))
    print(f"\nResult : {'✅ SUCCESS' if s else f'❌ FAILED at [{f}]'}")
    print(f"Steps  : {len(rows)}")
    print(f"Phases : {' → '.join(phases)}")
