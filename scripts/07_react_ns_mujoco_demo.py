import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
XML_PATH = PROJECT_DIR / "models" / "franka_fr3_v2_space" / "scene.xml"


# MODEL NAMES

ARM_JOINT_NAMES = [
    "fr3v2_joint1",
    "fr3v2_joint2",
    "fr3v2_joint3",
    "fr3v2_joint4",
    "fr3v2_joint5",
    "fr3v2_joint6",
    "fr3v2_joint7",
]

EE_SITE_NAME = "ee_site"
TARGET_SITE_NAME = "target_site"


# CONTROLLER


# Options:
#   "clik"
#   "dls"
#   "fixed_rns"
#   "react_ns"
CONTROLLER = "react_ns"

LABEL_MODE = "site"


class Config:

    # Task-space control
    kp = 1.8
    vmax = 0.30

    # Joint velocity safety
    qdot_max = 0.55

    # Constant damping for DLS and Fixed-RNS
    lambda_fixed = 0.04

    # ReAct-NS adaptive damping
    lambda_min = 0.02
    lambda_s = 0.20
    w0 = 0.15

    # Fixed-RNS reaction suppression
    mu_fixed = 8.0

    # ReAct-NS adaptive reaction suppression
    mu_min = 1.0
    mu_max = 80.0
    e0 = 0.25

    # Smoothness regularization
    nu = 0.25

    # Joint-limit avoidance regularization
    eta = 0.0005

    # Numerical safety
    eps = 1e-8

    # Demo duration
    sim_time = 12.0

    # Empirical base-reaction coupling weights
    coupling_weights = np.array([0.60, 0.52, 0.40, 0.32, 0.20, 0.12, 0.08])

    # Heat-map visual settings
    heat_marker_radius_min = 0.025
    heat_marker_radius_max = 0.075

    # Controls how quickly red appears.
    # Increase if everything is red.
    # Decrease if everything is blue.
    heat_normalization = 12.0

    # End-effector trail
    show_ee_trail = True
    ee_trail_stride = 40
    ee_trail_max_points = 150

    # Floating color legend
    show_heat_legend = True

    # Print debug line every N simulation steps
    print_every = 100


# MUJOCO UTILITIES

def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else ""


def configure_viewer_labels(viewer):

    if LABEL_MODE == "none":
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE

    elif LABEL_MODE == "site":
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

    elif LABEL_MODE == "body":
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY

    elif LABEL_MODE == "joint":
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_JOINT

    elif LABEL_MODE == "geom":
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_GEOM

    else:
        print(f"WARNING: Unknown LABEL_MODE={LABEL_MODE}. Using site labels.")
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE


def reset_home(model, data):
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def get_ids(model):
    joint_ids = [model.joint(name).id for name in ARM_JOINT_NAMES]
    dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]
    qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]

    ee_site_id = model.site(EE_SITE_NAME).id
    target_site_id = model.site(TARGET_SITE_NAME).id

    actuator_map = {}
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id in joint_ids:
            actuator_map[joint_id] = act_id

    return joint_ids, dof_ids, qpos_ids, ee_site_id, target_site_id, actuator_map


def print_model_check(model):
    print("\nModel check")
    print("-" * 70)
    print("XML:", XML_PATH)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print("nsite:", model.nsite)

    print("\nJoints:")
    for i in range(model.njnt):
        print(f"  {i:2d}: {mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")

    print("\nSites:")
    for i in range(model.nsite):
        print(f"  {i:2d}: {mj_name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")

    print("-" * 70)


def clamp_norm(x, max_norm):
    norm = np.linalg.norm(x)

    if norm > max_norm:
        return x * (max_norm / (norm + 1e-12))

    return x


def get_end_effector_jacobian(model, data, ee_site_id, dof_ids):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    return jacp[:, dof_ids]


# CONTROL MODEL

def estimate_reaction_matrix(cfg, n):

    A = np.zeros((3, n))
    weights = cfg.coupling_weights.copy()

    if len(weights) != n:
        weights = np.linspace(0.60, 0.08, n)

    A[2, :] = weights
    A[0, :] = 0.15 * weights
    A[1, :] = 0.10 * weights

    return A


def manipulability(J, cfg):
    M = J @ J.T + cfg.eps * np.eye(J.shape[0])
    det = np.linalg.det(M)

    if det < 0:
        det = 0.0

    return float(np.sqrt(det))


def joint_limit_weight_matrix(model, data, qpos_ids):
    n = len(qpos_ids)
    W = np.zeros((n, n))

    for local_i, qpos_id in enumerate(qpos_ids):
        q = data.qpos[qpos_id]

        joint_id = None
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == qpos_id:
                joint_id = j
                break

        if joint_id is None:
            continue

        if not model.jnt_limited[joint_id]:
            continue

        qmin, qmax = model.jnt_range[joint_id]
        dist = min(q - qmin, qmax - q)

        W[local_i, local_i] = 1.0 / (dist * dist + 1e-4)

    return W


def solve_clik(J, xdot_des):
    return np.linalg.pinv(J) @ xdot_des


def solve_dls(J, xdot_des, cfg):
    m = J.shape[0]

    return J.T @ np.linalg.solve(
        J @ J.T + cfg.lambda_fixed**2 * np.eye(m),
        xdot_des,
    )


def solve_fixed_rns(J, A, xdot_des, cfg):
    n = J.shape[1]

    H = (
        J.T @ J
        + cfg.lambda_fixed**2 * np.eye(n)
        + cfg.mu_fixed * (A.T @ A)
    )

    b = J.T @ xdot_des

    return np.linalg.solve(H + cfg.eps * np.eye(n), b)


def solve_react_ns(model, data, J, A, xdot_des, e, qdot_prev, qpos_ids, cfg):
    n = J.shape[1]

    w = manipulability(J, cfg)

    mu = cfg.mu_min + (cfg.mu_max - cfg.mu_min) * np.exp(
        -np.linalg.norm(e) / cfg.e0
    )

    lam = cfg.lambda_min + cfg.lambda_s * np.exp(-w / cfg.w0)

    Wl = joint_limit_weight_matrix(model, data, qpos_ids)

    H = (
        J.T @ J
        + lam**2 * np.eye(n)
        + mu * (A.T @ A)
        + cfg.nu * np.eye(n)
        + cfg.eta * (Wl.T @ Wl)
    )

    b = J.T @ xdot_des + cfg.nu * qdot_prev

    qdot = np.linalg.solve(H + cfg.eps * np.eye(n), b)

    info = {
        "mu": float(mu),
        "lambda": float(lam),
        "manipulability": float(w),
    }

    return qdot, info


def controller_step(controller_name, model, data, J, A, xdot_des, e, qdot_prev, qpos_ids, cfg):
    info = {
        "mu": np.nan,
        "lambda": np.nan,
        "manipulability": manipulability(J, cfg),
    }

    if controller_name == "clik":
        qdot_cmd = solve_clik(J, xdot_des)

    elif controller_name == "dls":
        qdot_cmd = solve_dls(J, xdot_des, cfg)
        info["lambda"] = cfg.lambda_fixed

    elif controller_name == "fixed_rns":
        qdot_cmd = solve_fixed_rns(J, A, xdot_des, cfg)
        info["mu"] = cfg.mu_fixed
        info["lambda"] = cfg.lambda_fixed

    elif controller_name == "react_ns":
        qdot_cmd, info = solve_react_ns(
            model=model,
            data=data,
            J=J,
            A=A,
            xdot_des=xdot_des,
            e=e,
            qdot_prev=qdot_prev,
            qpos_ids=qpos_ids,
            cfg=cfg,
        )

    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    qdot_cmd = np.clip(qdot_cmd, -cfg.qdot_max, cfg.qdot_max)

    return qdot_cmd, info


def apply_velocity_controls(data, joint_ids, actuator_map, qdot_cmd):
    data.ctrl[:] = 0.0

    for local_i, joint_id in enumerate(joint_ids):
        act_id = actuator_map.get(joint_id, None)

        if act_id is not None:
            data.ctrl[act_id] = qdot_cmd[local_i]


# BASE DISTURBANCE METRICS

def base_rotation_angle(data):
    if len(data.qpos) < 7:
        return 0.0

    quat = data.qpos[3:7].copy()
    quat = quat / (np.linalg.norm(quat) + 1e-12)

    qw = np.clip(quat[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(qw))

    return float(angle)


def base_angular_velocity_norm(data):
    if len(data.qvel) < 6:
        return 0.0

    return float(np.linalg.norm(data.qvel[3:6]))


# VISUALIZATION

def heat_color(value):
    """
    Color map:
        0.00 = blue
        0.25 = cyan
        0.50 = green
        0.75 = yellow
        1.00 = red
    """
    v = float(np.clip(value, 0.0, 1.0))

    if v < 0.25:
        local = v / 0.25
        r = 0.0
        g = local
        b = 1.0

    elif v < 0.50:
        local = (v - 0.25) / 0.25
        r = 0.0
        g = 1.0
        b = 1.0 - local

    elif v < 0.75:
        local = (v - 0.50) / 0.25
        r = local
        g = 1.0
        b = 0.0

    else:
        local = (v - 0.75) / 0.25
        r = 1.0
        g = 1.0 - local
        b = 0.0

    return np.array([r, g, b, 0.85], dtype=float)


def compute_joint_reaction_intensity(A, qdot_cmd, info, cfg):
    
    if np.isnan(info["mu"]):
        mu_for_visual = 1.0
    else:
        mu_for_visual = float(info["mu"])

    column_norms = np.linalg.norm(A, axis=0)
    raw = mu_for_visual * column_norms * np.abs(qdot_cmd)

    normalized = raw / (cfg.heat_normalization + 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)

    return raw, normalized


def add_sphere_to_scene(scene, position, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )

    scene.ngeom += 1


def add_capsule_to_scene(scene, p1, p2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.0, 0.0], dtype=float),
        np.zeros(3, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )

    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        p1,
        p2,
    )

    scene.ngeom += 1


def add_heatmap_markers(viewer, data, joint_ids, normalized_heat, cfg):

    scene = viewer.user_scn

    for local_i, joint_id in enumerate(joint_ids):
        if scene.ngeom >= scene.maxgeom:
            break

        position = data.xanchor[joint_id].copy()

        heat = float(normalized_heat[local_i])
        rgba = heat_color(heat)

        radius = (
            cfg.heat_marker_radius_min
            + heat * (cfg.heat_marker_radius_max - cfg.heat_marker_radius_min)
        )

        add_sphere_to_scene(scene, position, radius, rgba)


def add_ee_trail(viewer, trail_points):

    scene = viewer.user_scn

    if len(trail_points) < 2:
        return

    rgba = np.array([0.2, 0.8, 1.0, 0.35], dtype=float)

    for i in range(1, len(trail_points)):
        p1 = trail_points[i - 1]
        p2 = trail_points[i]

        add_capsule_to_scene(scene, p1, p2, 0.006, rgba)


def add_heat_legend(viewer):
 
    scene = viewer.user_scn

    base_pos = np.array([0.15, -0.75, 1.05], dtype=float)
    spacing = 0.09

    heat_values = [0.0, 0.25, 0.50, 0.75, 1.0]

    for i, heat in enumerate(heat_values):
        position = base_pos + np.array([i * spacing, 0.0, 0.0])
        rgba = heat_color(heat)
        radius = 0.028

        add_sphere_to_scene(scene, position, radius, rgba)


def print_heat_status(sim_t, error_norm, base_angle_deg, base_omega, info, normalized_heat):
    mu_str = "nan" if np.isnan(info["mu"]) else f"{info['mu']:.3f}"
    lam_str = "nan" if np.isnan(info["lambda"]) else f"{info['lambda']:.4f}"

    hottest_joint_idx = int(np.argmax(normalized_heat)) + 1
    hottest_value = float(np.max(normalized_heat))

    heat_values = " ".join([f"J{i + 1}:{v:.2f}" for i, v in enumerate(normalized_heat)])

    print(
        f"t={sim_t:6.2f} | "
        f"err={error_norm:7.4f} m | "
        f"base={base_angle_deg:7.3f} deg | "
        f"omega={base_omega:7.4f} | "
        f"mu={mu_str:>8s} | "
        f"lambda={lam_str:>8s} | "
        f"hottest=J{hottest_joint_idx}({hottest_value:.2f}) | "
        f"{heat_values}"
    )


def main():
    cfg = Config()

    if not XML_PATH.exists():
        raise FileNotFoundError(f"Cannot find XML: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    reset_home(model, data)

    print_model_check(model)

    joint_ids, dof_ids, qpos_ids, ee_site_id, target_site_id, actuator_map = get_ids(model)

    if len(actuator_map) < 7:
        print("\nWARNING: Not all arm joints have mapped actuators.")
        print("Mapped actuators:", len(actuator_map))

    qdot_prev = np.zeros(7)
    ee_trail_points = []

    print("\nRunning ReAct-NS MuJoCo reaction heat-map demo")
    print("Controller:", CONTROLLER)
    print("Label mode:", LABEL_MODE)
    print("\nVisual meaning:")
    print("  Colored joint spheres = joint-level reaction contribution")
    print("  Blue/cyan             = low reaction contribution")
    print("  Green/yellow          = medium reaction contribution")
    print("  Red                   = high reaction contribution")
    print("  Cyan curve            = end-effector trajectory trail")
    print("  Floor color spheres   = blue-to-red heat-map legend")
    print("\nNote:")
    print("  These colors are not temperature. They visualize controller reaction intensity.")
    print("-" * 70)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_labels(viewer)

        start_wall_time = time.time()
        step = 0

        while viewer.is_running():
            mujoco.mj_forward(model, data)

            sim_t = data.time

            x_ee = data.site_xpos[ee_site_id].copy()
            x_target = data.site_xpos[target_site_id].copy()

            e = x_target - x_ee
            xdot_des = clamp_norm(cfg.kp * e, cfg.vmax)

            J = get_end_effector_jacobian(model, data, ee_site_id, dof_ids)
            A = estimate_reaction_matrix(cfg, len(dof_ids))

            qdot_cmd, info = controller_step(
                controller_name=CONTROLLER,
                model=model,
                data=data,
                J=J,
                A=A,
                xdot_des=xdot_des,
                e=e,
                qdot_prev=qdot_prev,
                qpos_ids=qpos_ids,
                cfg=cfg,
            )

            raw_heat, normalized_heat = compute_joint_reaction_intensity(
                A=A,
                qdot_cmd=qdot_cmd,
                info=info,
                cfg=cfg,
            )

            apply_velocity_controls(
                data=data,
                joint_ids=joint_ids,
                actuator_map=actuator_map,
                qdot_cmd=qdot_cmd,
            )

            mujoco.mj_step(model, data)

            qdot_prev = qdot_cmd.copy()

            if cfg.show_ee_trail and step % cfg.ee_trail_stride == 0:
                ee_trail_points.append(x_ee.copy())

                if len(ee_trail_points) > cfg.ee_trail_max_points:
                    ee_trail_points.pop(0)

            # Clear previous custom visualization objects.
            viewer.user_scn.ngeom = 0

            # Draw heat-map markers on robot joints.
            add_heatmap_markers(
                viewer=viewer,
                data=data,
                joint_ids=joint_ids,
                normalized_heat=normalized_heat,
                cfg=cfg,
            )

            # Draw end-effector motion trail.
            if cfg.show_ee_trail:
                add_ee_trail(viewer, ee_trail_points)

            # Draw color legend.
            if cfg.show_heat_legend:
                add_heat_legend(viewer)

            if step % cfg.print_every == 0:
                err = np.linalg.norm(e)
                base_angle_deg = np.rad2deg(base_rotation_angle(data))
                base_omega = base_angular_velocity_norm(data)

                print_heat_status(
                    sim_t=sim_t,
                    error_norm=err,
                    base_angle_deg=base_angle_deg,
                    base_omega=base_omega,
                    info=info,
                    normalized_heat=normalized_heat,
                )

            viewer.sync()

            step += 1

            if sim_t >= cfg.sim_time:
                break

            elapsed = time.time() - start_wall_time
            if data.time > elapsed:
                time.sleep(data.time - elapsed)

    print("\nReaction heat-map demo finished.")


if __name__ == "__main__":
    main()