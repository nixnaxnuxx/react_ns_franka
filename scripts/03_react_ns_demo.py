import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = Path("react_ns_franka/models/franka_fr3_v2_space/scene.xml")

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

# Change this to test different controllers:
# "clik", "dls", "fixed_rns", "react_ns"
CONTROLLER = "clik"


class ReActNSConfig:
    """
    Live demo configuration.

    This matches the tuned configuration in 04_run_comparison.py.
    """

    # Task-space proportional control
    kp = 1.8
    vmax = 0.30

    # Joint velocity safety limit
    qdot_max = 0.55

    # Constant damping for DLS and Fixed-RNS
    lambda_fixed = 0.04

    # ReAct-NS adaptive damping:
    # lambda(t) = lambda_min + lambda_s * exp(-manipulability / w0)
    lambda_min = 0.02
    lambda_s = 0.20
    w0 = 0.15

    # Fixed-RNS reaction suppression
    mu_fixed = 8.0

    # ReAct-NS adaptive reaction suppression:
    # mu(t) = mu_min + (mu_max - mu_min) * exp(-||e|| / e0)
    mu_min = 1.0
    mu_max = 80.0
    e0 = 0.25

    # Smoothness regularization
    nu = 0.25

    # Joint-limit avoidance
    eta = 0.0005

    # Numerical regularization
    eps = 1e-8

    # Demo length
    sim_time = 12.0

    # Empirical base-reaction coupling weights
    coupling_weights = np.array([0.60, 0.52, 0.40, 0.32, 0.20, 0.12, 0.08])


def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else ""


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
    print("-" * 60)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print("nsite:", model.nsite)

    print("\nSites:")
    for i in range(model.nsite):
        print(" ", i, mj_name(model, mujoco.mjtObj.mjOBJ_SITE, i))

    print("\nJoints:")
    for i in range(model.njnt):
        print(" ", i, mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

    print("-" * 60)


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


def estimate_reaction_matrix(cfg, n):
    """
    Approximate reaction mapping:
        r_b = A qdot

    This is an empirical matrix for first-stage validation.
    Actual base rotation is still measured from MuJoCo.
    """
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


def main():
    cfg = ReActNSConfig()

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

    print("\nRunning live demo")
    print("Controller:", CONTROLLER)
    print("Close the viewer window to stop early.")
    print("-" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
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

            apply_velocity_controls(data, joint_ids, actuator_map, qdot_cmd)

            mujoco.mj_step(model, data)

            qdot_prev = qdot_cmd.copy()

            if step % 20 == 0:
                err = np.linalg.norm(e)
                base_angle_deg = np.rad2deg(base_rotation_angle(data))
                base_omega = base_angular_velocity_norm(data)
                effort = np.linalg.norm(qdot_cmd) ** 2

                mu_str = "nan" if np.isnan(info["mu"]) else f"{info['mu']:.3f}"
                lam_str = "nan" if np.isnan(info["lambda"]) else f"{info['lambda']:.4f}"

                print(
                    f"t={sim_t:6.2f} | "
                    f"err={err:7.4f} m | "
                    f"base={base_angle_deg:7.3f} deg | "
                    f"omega={base_omega:7.4f} | "
                    f"effort={effort:7.4f} | "
                    f"mu={mu_str:>8s} | "
                    f"lambda={lam_str:>8s}"
                )

            viewer.sync()
            step += 1

            if sim_t >= cfg.sim_time:
                break

            elapsed = time.time() - start_wall_time
            if data.time > elapsed:
                time.sleep(data.time - elapsed)

    print("\nLive demo finished.")


if __name__ == "__main__":
    main()