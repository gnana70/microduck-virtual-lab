"""Physics test battery for the virtual Microduck. Writes results/physics.json."""
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import vlab

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
HD_RATED = 3.7 * 0.0980665   # 3.7 kg.cm at 7.4 V (spec sheet 5-8) -> N.m
HD_STALL = 15 * 0.0980665    # 15 kg.cm at 7.4 V (5-5)
XL_STALL_5V = 0.52           # XL330-M288 at 5 V (Robotis e-manual)


def census():
    m, d, c, p = vlab.build("xl330", render=(320, 240))
    meshes = {}
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
            meshes[name] = meshes.get(name, 0) + 1
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 2 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n]
            R = d.geom_xmat[g].reshape(3, 3)
            pts.append(v @ R.T + d.geom_xpos[g])
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    # Settle on the floor to get the real standing height.
    vlab.step_policy(m, d, c, p, ticks=50, use_policy=False)
    pts2 = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 2 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n]
            R = d.geom_xmat[g].reshape(3, 3)
            pts2.append(v @ R.T + d.geom_xpos[g])
    pts2 = np.vstack(pts2)
    joints = []
    for name in vlab.JOINTS:
        j = m.joint(name)
        joints.append({"joint": name, "range_deg": [round(math.degrees(x), 1) for x in j.range]})
    bodies = []
    for b in range(1, m.nbody):
        bodies.append({"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                       "mass_g": round(float(m.body_mass[b]) * 1000, 1)})
    return {
        "mesh_types": len(meshes), "mesh_instances": int(sum(meshes.values())),
        "servo_meshes": meshes.get("xl330", 0),
        "bearings_22x16x4": meshes.get("seeed_bearing__configuration__22x16x4", 0),
        "bearings_15x10x3": meshes.get("seeed_bearing__configuration_default", 0),
        "total_mass_g": round(float(m.body_mass.sum()) * 1000, 1),
        "standing_height_mm": round(float(pts2[:, 2].max() - pts2[:, 2].min()) * 1000, 1),
        "standing_footprint_mm": [round(float(x) * 1000, 1) for x in (pts2.max(0) - pts2.min(0))[:2]],
        "actuated_joints": int(m.nu), "joints": joints, "bodies": bodies,
        "meshes": meshes,
    }


def hold_test(kind, extra_mass=0.0, seconds=5.0, vin=None):
    """No policy: servos just hold the STAND pose. Gravity load per joint."""
    m, d, c, p = vlab.build(kind, render=(320, 240), extra_mass_per_servo=extra_mass, vin=vin)
    vlab.step_policy(m, d, c, p, ticks=25, use_policy=False)  # settle 0.5 s
    taus = []
    for _ in range(int(seconds * 50)):
        vlab.step_policy(m, d, c, p, ticks=1, use_policy=False)
        taus.append(vlab.actuator_torques(m, d))
    taus = np.array(taus)
    pos, tilt, _ = vlab.trunk_state(m, d)
    q = d.qpos[p.joint_qpos_indices]
    err = np.degrees(np.abs(q - p.default_pose))
    return {"kind": kind, "extra_mass_g_per_servo": extra_mass, "upright": bool(tilt < 20 and pos[2] > 0.08),
            "trunk_z_mm": round(float(pos[2]) * 1000, 1), "tilt_deg": round(tilt, 2),
            "mean_abs_torque": {n: round(float(np.abs(taus[:, i]).mean()), 4) for i, n in enumerate(vlab.JOINTS)},
            "max_pose_error_deg": round(float(err.max()), 2),
            "pose_error_deg": {n: round(float(err[i]), 2) for i, n in enumerate(vlab.JOINTS)}}


def walk_test(kind, vx, yaw_rate=0.0, seconds=10.0, vin=None, yaw0=0.0, extra_mass=0.0, push=None):
    m, d, c, p = vlab.build(kind, render=(320, 240), vin=vin, extra_mass_per_servo=extra_mass)
    vlab.reset(m, d, c, p, yaw=yaw0)
    vlab.step_policy(m, d, c, p, ticks=50)  # 1 s settle with the policy
    pos0, _, yaw_start = vlab.trunk_state(m, d)
    taus, tilts, fell_at = [], [], None
    yaw_prev, yaw_acc = yaw_start, 0.0
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    n = int(seconds * 50)
    for k in range(n):
        if push and push["at_tick"] <= k < push["at_tick"] + push["ticks"]:
            d.xfrc_applied[bid, :3] = push["force"]
        else:
            d.xfrc_applied[bid, :3] = 0
        vlab.step_policy(m, d, c, p, vx, yaw_rate, ticks=1)
        taus.append(vlab.actuator_torques(m, d))
        pos, tilt, yaw = vlab.trunk_state(m, d)
        tilts.append(tilt)
        yaw_acc += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yaw
        if fell_at is None and (tilt > 45 or pos[2] < 0.06):
            fell_at = (k + 1) / 50
    pos1, tilt, yaw_end = vlab.trunk_state(m, d)
    taus = np.array(taus)
    # Displacement in the start heading frame
    dp = pos1[:2] - pos0[:2]
    ch, sh = math.cos(yaw_start), math.sin(yaw_start)
    fwd, lat = dp[0] * ch + dp[1] * sh, -dp[0] * sh + dp[1] * ch
    dyaw = yaw_acc
    return {"kind": kind, "vx_cmd": vx, "yaw_cmd": yaw_rate, "vin": vin, "yaw0": yaw0,
            "seconds": seconds, "fell": fell_at is not None, "fell_at_s": fell_at,
            "forward_m": round(float(fwd), 3), "lateral_m": round(float(lat), 3),
            "heading_change_deg": round(math.degrees(dyaw), 1),
            "fwd_speed_mps": round(float(fwd) / seconds, 3),
            "yaw_rate_meas": round(dyaw / seconds, 3),
            "max_tilt_deg": round(float(max(tilts)), 1),
            "peak_torque": {nm: round(float(np.abs(taus[:, i]).max()), 3) for i, nm in enumerate(vlab.JOINTS)},
            "rms_torque": {nm: round(float(np.sqrt((taus[:, i] ** 2).mean())), 3) for i, nm in enumerate(vlab.JOINTS)}}


def main():
    t0 = time.time()
    res = {"spec": {"hd1910_rated_Nm": round(HD_RATED, 3), "hd1910_stall_Nm": round(HD_STALL, 3),
                    "xl330_stall_5V_Nm": XL_STALL_5V,
                    "hd1910_model_limit_Nm_7.7V": None, "xl330_model_limit_Nm_7.4V": None}}
    for kind in ("xl330", "hd1910"):
        bm, _ = vlab.bam_model(kind)
        res["spec"][f"{kind}_model_limit_Nm_{bm.actuator.vin}V"] = round(bm.actuator.vin * bm.kt.value / bm.R.value, 3)
    res["census"] = census(); print("census", time.time() - t0, flush=True)
    res["hold"] = [hold_test("xl330"), hold_test("hd1910"), hold_test("hd1910", extra_mass=3.0)]
    print("hold", time.time() - t0, flush=True)
    walks = []
    for kind in ("xl330", "hd1910"):
        for vx in (0.15, 0.2, 0.25):
            for yaw0 in (0.0, 0.8, -0.8):
                walks.append(walk_test(kind, vx, yaw0=yaw0))
        walks.append(walk_test(kind, 0.0, yaw_rate=0.9))
        walks.append(walk_test(kind, 0.0, yaw_rate=-0.9))
        walks.append(walk_test(kind, 0.0, seconds=10))
    for vin in (7.0, 7.4, 8.0):
        walks.append(walk_test("hd1910", 0.25, vin=vin))
    walks.append(walk_test("hd1910", 0.25, extra_mass=3.0))
    res["walk"] = walks
    print("walk", time.time() - t0, flush=True)
    pushes = []
    for kind in ("xl330", "hd1910"):
        for f in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0):
            for direction in ("side", "front"):
                force = [0, f, 0] if direction == "side" else [f, 0, 0]
                r = walk_test(kind, 0.0, seconds=4, push={"at_tick": 50, "ticks": 5, "force": force})
                r.update({"push_N": f, "push_dir": direction, "push_s": 0.1})
                pushes.append(r)
    res["push"] = pushes
    print("push", time.time() - t0, flush=True)
    (OUT / "physics.json").write_text(json.dumps(res, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
