"""Virtual Microduck lab: the real MJCF + STL duck in native MuJoCo physics.

Two servo models, both BAM M6 (Rhoban) voltage/friction models:
  * "xl330"  - Pollen's original Dynamixel XL330 (bam bundled params), kp_fw=200, as in
               upstream microduck_rl/scripts/infer_policy.py (what every official policy
               is trained against).
  * "hd1910" - Feetech HD-1910 used by the replica: LuwuDynamics 1910_m6.json shipped in
               microduck-replica/software/training, kp_fw=5, vin 7.4-8.0 V, vin_min 7.0 V
               (microduck_hd1910_env_cfg.py).

Policy: Pollen Robotics' official velstand ONNX (trained for XL330).
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
TP = Path(os.environ.get("MDLAB_THIRD_PARTY", HERE / "third_party"))
RL = TP / "microduck_rl"                      # git clone https://github.com/pollen-robotics/microduck_rl
sys.path.insert(0, str(TP / "bam"))            # git clone https://github.com/Rhoban/bam (commit 62bd8ce)
HD1910_JSON = HERE / "params" / "1910_m6.json"  # LuwuDynamics, Apache-2.0 (as shipped in microduck-replica)
POLICY = HERE / "policies" / "velstand.onnx"    # Pollen Robotics, huggingface.co/pollen-robotics/microduck-policies
SCENE = RL / "src/mjlab_microduck/robot/microduck/scene.xml"

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]

# Dark-mode look. Not black: the trunk and servos are near-black plastic, so the
# backdrop is a dark slate blue that keeps them readable.
BG_TOP = [0.16, 0.20, 0.27]
BG_BOTTOM = [0.07, 0.09, 0.12]
FLOOR_A = [0.125, 0.145, 0.185]
FLOOR_B = [0.11, 0.128, 0.165]
FLOOR_LINE = [0.19, 0.22, 0.28]


def official():
    """Load upstream infer_policy.py as a module (it is a script)."""
    src = RL / "scripts" / "infer_policy.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    mod = types.ModuleType("official_infer")
    mod.__file__ = str(src)
    exec(compile(tree, str(src), "exec"), mod.__dict__)
    return mod


OFF = official()


def bam_model(kind: str, vin: float | None = None):
    from bam.model import load_model, load_model_from_dict
    if kind == "xl330":
        m = load_model(motor_name="xl330", model="m6")
        m.actuator.kp = 200.0
        m.actuator.vin = 7.4 if vin is None else vin
        m.actuator.max_current = None
        return m, dict(vin_drop_gain=0.1, vin_min=6.0)
    if kind == "hd1910":
        m = load_model_from_dict(json.loads(HD1910_JSON.read_text()))
        m.actuator.kp = 5.0
        m.actuator.vin = 7.7 if vin is None else vin
        m.actuator.max_current = None
        return m, dict(vin_drop_gain=0.1, vin_min=7.0)
    raise ValueError(kind)


def style_spec(spec: mujoco.MjSpec, w=1920, h=1080, samples=4):
    spec.visual.global_.offwidth = w
    spec.visual.global_.offheight = h
    spec.visual.quality.offsamples = samples
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.32, 0.33, 0.36]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.specular = [0.08, 0.08, 0.08]
    spec.visual.rgba.haze = [*BG_BOTTOM, 1.0]
    spec.visual.map.znear = 0.002
    for t in spec.textures:
        if t.name == "groundplane":
            t.rgb1, t.rgb2, t.markrgb = FLOOR_A, FLOOR_B, FLOOR_LINE
        elif t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1, t.rgb2 = BG_TOP, BG_BOTTOM
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.04
            mat.texrepeat = [8, 8]
    # Key + rim light so dark plastic keeps its edges.
    spec.worldbody.add_light(name="key", pos=[0.6, -0.8, 1.2], dir=[-0.45, 0.6, -0.8],
                             diffuse=[0.55, 0.55, 0.52], specular=[0.15, 0.15, 0.15],
                             castshadow=True)
    spec.worldbody.add_light(name="rim", pos=[-0.8, 0.6, 0.7], dir=[0.7, -0.5, -0.4],
                             diffuse=[0.28, 0.32, 0.40], specular=[0.1, 0.1, 0.12],
                             castshadow=False)
    return spec


def build(kind="xl330", *, vin=None, render=(1920, 1080), extra_mass_per_servo=0.0,
          scene=SCENE, spec_hook=None):
    """Return (model, data, bam_ctrl, policy) for one duck."""
    from bam.mujoco import MujocoController
    bm, ctrl_kw = bam_model(kind, vin)
    force_limit = bm.actuator.vin * bm.kt.value / bm.R.value
    spec = mujoco.MjSpec.from_file(str(scene))
    style_spec(spec, *render)
    if spec_hook:
        spec_hook(spec)
    names = []
    for act in spec.actuators:
        tgt = act.target.name if hasattr(act.target, "name") else str(act.target)
        if tgt.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = [-force_limit, force_limit]
        act.ctrllimited = False
        act.gear = [1, 0, 0, 0, 0, 0]
        names.append(act.name)
        for j in spec.joints:
            if j.name == tgt:
                j.damping = np.zeros((3, 1))
                j.frictionloss = 0
                j.solref_friction = OFF.BAM_STIFF_SOLREF_FRICTION
                j.solimp_friction = OFF.BAM_STIFF_SOLIMP_FRICTION
                break
    if extra_mass_per_servo:
        add_servo_mass(spec, extra_mass_per_servo)
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bm, names, model, data, **ctrl_kw)
    with contextlib.redirect_stdout(io.StringIO()):
        pol = OFF.PolicyInference(model, data, walking_onnx_path=str(POLICY), action_scale=1.0,
                                  bam_ctrl=bam_ctrl, use_projected_gravity=True,
                                  new_cmd_obs=True)
    reset(model, data, bam_ctrl, pol)
    return model, data, bam_ctrl, pol


def add_servo_mass(spec, grams):
    """HD-1910 weighs 21 g vs XL330 18 g: add the difference on every body that
    carries a servo mesh, at that mesh's position."""
    for body in spec.bodies:
        n = sum(1 for g in body.geoms if g.meshname == "xl330")
        if n and body.mass > 0:
            body.mass = body.mass + n * grams / 1000.0


def reset(model, data, bam_ctrl, pol, xy=(0, 0), yaw=0.0, z=0.125):
    mujoco.mj_resetData(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    a = int(model.jnt_qposadr[jid])
    data.qpos[a:a + 7] = [xy[0], xy[1], z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    for i, adr in enumerate(pol.joint_qpos_indices):
        data.qpos[adr] = pol.default_pose[i]
    bam_ctrl.reset(data.qpos)
    act = bam_ctrl.model.actuator
    if type(act).__name__ == "STS3215Actuator":
        # Feetech goal limiter: seed from the measured pose, exactly like the replica's
        # FeetechBamActuator fix (software/training/.../actuator/feetech_bam.py).
        act.q_target_smooth = data.qpos[bam_ctrl.qpos_indexes].copy()
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(model, data)
    pol.vel_cmd[:] = 0
    pol._update_command()


def trunk_state(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    pos = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3, 3)
    up = R[:, 2]
    tilt = math.degrees(math.acos(np.clip(up[2], -1, 1)))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return pos, tilt, yaw


def step_policy(model, data, bam_ctrl, pol, vx=0.0, yaw_rate=0.0, ticks=1, use_policy=True,
                on_substep=None):
    pol.vel_cmd[:] = [vx, 0.0, yaw_rate]
    pol._update_command()
    for _ in range(ticks):
        if use_policy:
            action = pol.infer()
            pol.apply_action(action)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            if on_substep:
                on_substep()


def actuator_torques(model, data):
    return data.actuator_force.copy()
