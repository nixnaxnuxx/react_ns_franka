import csv
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt


XML_PATH = Path("react_ns_franka/models/franka_fr3_v2_space/scene.xml")
RESULTS_DIR = Path("react_ns_franka/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

CONTROLLERS = ["clik", "dls", "fixed_rns", "react_ns"]


class Config:
    """
The updated values are tuned to make ReAct-NS:
    1. Suppress base disturbance more strongly.
    2. Avoid sharp effort spikes.
    3. Show clearer adaptive damping behavior.
    """

    # Task-space controller
    kp = 1.8
    vmax = 0.30

    # Joint velocity safety limit
    qdot_max = 0.55

    # Constant damping used by DLS and Fixed-RNS
    lambda_fixed = 0.04

    # ReAct-NS adaptive damping
    lambda_min = 0.02
    lambda_s = 0.20
    w0 = 0.15

    # Fixed-RNS constant reaction suppression
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

    # Simulation settings
    sim_time = 12.0
    log_every = 5

    coupling_weights = np.array([0.60, 0.52, 0.40, 0.32, 0.20, 0.12, 0.08])


def clamp_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp vector norm to max_norm."""
    norm = np.linalg.norm(x)
    if norm > max_norm:
        return x * max_norm / (norm + 1e-12)
    return x


def get_ids(model):
    """Get MuJoCo IDs for joints, site and actuators"""
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


def reset_home(model, data):
    """Reset  simulation to XML keyframe 'home' if available"""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def get_site_jacobian(model, data, site_id, dof_ids):
    """Return translational end-effector Jacobian for arm joints"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return jacp[:, dof_ids]


def estimate_reaction_matrix(cfg: Config, n: int) -> np.ndarray:

    A = np.zeros((3, n))
    weights = cfg.coupling_weights.copy()

    if len(weights) != n:
        weights = np.linspace(0.60, 0.08, n)

    # Penalize base yaw/spin most strongly.
    A[2, :] = weights

    # Penalize smaller roll/pitch-like disturbance terms.
    A[0, :] = 0.15 * weights
    A[1, :] = 0.10 * weights

    return A


def manipulability(J: np.ndarray, cfg: Config) -> float:
    """
    Yoshikawa-style manipulability measure
    """
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

        # Large value near joint limits.
        W[local_i, local_i] = 1.0 / (dist * dist + 1e-4)

    return W


def solve_clik(J: np.ndarray, xdot_des: np.ndarray) -> np.ndarray:
    """
    Closed-loop inverse kinematics baseline
    """
    return np.linalg.pinv(J) @ xdot_des


def solve_dls(J: np.ndarray, xdot_des: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Damped least-squares baseline
    """
    m = J.shape[0]

    return J.T @ np.linalg.solve(
        J @ J.T + cfg.lambda_fixed**2 * np.eye(m),
        xdot_des,
    )


def solve_fixed_rns(J: np.ndarray, A: np.ndarray, xdot_des: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Fixed reaction-suppression baseline
    """
    n = J.shape[1]

    H = (
        J.T @ J
        + cfg.lambda_fixed**2 * np.eye(n)
        + cfg.mu_fixed * (A.T @ A)
    )

    b = J.T @ xdot_des

    return np.linalg.solve(H + cfg.eps * np.eye(n), b)


def solve_react_ns(
    model,
    data,
    J: np.ndarray,
    A: np.ndarray,
    xdot_des: np.ndarray,
    e: np.ndarray,
    qdot_prev: np.ndarray,
    qpos_ids,
    cfg: Config,
):
    """
    Proposed ReAct-NS controller
    """
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


def controller_step(
    controller_name: str,
    model,
    data,
    J: np.ndarray,
    A: np.ndarray,
    xdot_des: np.ndarray,
    e: np.ndarray,
    qdot_prev: np.ndarray,
    qpos_ids,
    cfg: Config,
):

    info = {
        "mu": np.nan,
        "lambda": np.nan,
        "manipulability": manipulability(J, cfg),
    }

    if controller_name == "clik":
        qdot_cmd = solve_clik(J, xdot_des)
        # CLIK has no damping lambda and no reaction weight mu.
        info["mu"] = np.nan
        info["lambda"] = np.nan

    elif controller_name == "dls":
        qdot_cmd = solve_dls(J, xdot_des, cfg)
        # DLS has constant damping lambda but no reaction weight mu.
        info["mu"] = np.nan
        info["lambda"] = cfg.lambda_fixed

    elif controller_name == "fixed_rns":
        qdot_cmd = solve_fixed_rns(J, A, xdot_des, cfg)
        # Fixed-RNS has constant mu and constant lambda.
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


def base_rotation_angle(data) -> float:
    """
    Approximate base attitude disturbance from the first free-joint quaternion
    """
    if len(data.qpos) < 7:
        return 0.0

    quat = data.qpos[3:7].copy()
    quat = quat / (np.linalg.norm(quat) + 1e-12)

    qw = np.clip(quat[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(qw))

    return float(angle)


def base_angular_velocity_norm(data) -> float:
    """
    For a free joint, data.qvel[3:6] is angular velocity
    """
    if len(data.qvel) < 6:
        return 0.0

    return float(np.linalg.norm(data.qvel[3:6]))


def run_trial(controller_name: str):
    """
    Run one simulation trial for one controller
    """
    cfg = Config()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    reset_home(model, data)

    joint_ids, dof_ids, qpos_ids, ee_site_id, target_site_id, actuator_map = get_ids(model)

    if len(actuator_map) < 7:
        print(f"WARNING: Only {len(actuator_map)} actuators mapped for {controller_name}.")

    qdot_prev = np.zeros(7)
    previous_qdot = np.zeros(7)

    t_log = []
    error_log = []
    base_angle_log = []
    base_omega_log = []
    effort_log = []
    smoothness_log = []
    mu_log = []
    lambda_log = []
    manip_log = []

    n_steps = int(cfg.sim_time / model.opt.timestep)

    for step in range(n_steps):
        mujoco.mj_forward(model, data)

        x_ee = data.site_xpos[ee_site_id].copy()
        x_target = data.site_xpos[target_site_id].copy()

        e = x_target - x_ee

        xdot_des = cfg.kp * e
        xdot_des = clamp_norm(xdot_des, cfg.vmax)

        J = get_site_jacobian(model, data, ee_site_id, dof_ids)
        A = estimate_reaction_matrix(cfg, len(dof_ids))

        qdot_cmd, info = controller_step(
            controller_name=controller_name,
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

        if step % cfg.log_every == 0:
            dt = model.opt.timestep * cfg.log_every

            effort = float(np.linalg.norm(qdot_cmd) ** 2)
            smoothness = float(np.linalg.norm((qdot_cmd - previous_qdot) / dt) ** 2)
            previous_qdot = qdot_cmd.copy()

            t_log.append(float(data.time))
            error_log.append(float(np.linalg.norm(e)))
            base_angle_log.append(base_rotation_angle(data))
            base_omega_log.append(base_angular_velocity_norm(data))
            effort_log.append(effort)
            smoothness_log.append(smoothness)
            mu_log.append(float(info["mu"]) if not np.isnan(info["mu"]) else np.nan)
            lambda_log.append(float(info["lambda"]) if not np.isnan(info["lambda"]) else np.nan)
            manip_log.append(float(info["manipulability"]))

    final_error = error_log[-1]
    max_base_angle = max(base_angle_log)
    base_omega_integral = np.trapz(base_omega_log, t_log)
    effort_integral = np.trapz(effort_log, t_log)
    smoothness_integral = np.trapz(smoothness_log, t_log)

    success = (
        final_error < 0.03
        and np.rad2deg(max_base_angle) < 5.0
    )

    result = {
        "controller": controller_name,
        "final_error_m": float(final_error),
        "max_base_angle_deg": float(np.rad2deg(max_base_angle)),
        "base_omega_integral": float(base_omega_integral),
        "effort_integral": float(effort_integral),
        "smoothness_integral": float(smoothness_integral),
        "success": int(success),
    }

    logs = {
        "t": np.array(t_log),
        "error": np.array(error_log),
        "base_angle_deg": np.rad2deg(np.array(base_angle_log)),
        "base_omega": np.array(base_omega_log),
        "effort": np.array(effort_log),
        "smoothness": np.array(smoothness_log),
        "mu": np.array(mu_log),
        "lambda": np.array(lambda_log),
        "manipulability": np.array(manip_log),
    }

    return result, logs


def save_metrics(results):
    out_path = RESULTS_DIR / "metrics_single_target.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved metrics to: {out_path}")


def save_time_logs(all_logs):
    """Save time-series logs for each controller."""
    for controller, logs in all_logs.items():
        out_path = RESULTS_DIR / f"timeseries_{controller}.csv"

        keys = list(logs.keys())
        n = len(logs["t"])

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)

            for i in range(n):
                row = []
                for key in keys:
                    row.append(logs[key][i])
                writer.writerow(row)

        print(f"Saved time-series log: {out_path}")


def plot_tracking_error(all_logs):
    plt.figure(figsize=(8, 5))

    for controller, logs in all_logs.items():
        plt.plot(logs["t"], logs["error"], label=controller)

    plt.xlabel("Time (s)")
    plt.ylabel("End-effector error (m)")
    plt.title("Tracking Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tracking_error.png", dpi=300)
    plt.close()


def plot_base_rotation(all_logs):
    plt.figure(figsize=(8, 5))

    for controller, logs in all_logs.items():
        plt.plot(logs["t"], logs["base_angle_deg"], label=controller)

    plt.xlabel("Time (s)")
    plt.ylabel("Base rotation angle (deg)")
    plt.title("Base Disturbance Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "base_rotation.png", dpi=300)
    plt.close()


def plot_control_effort(all_logs):
    plt.figure(figsize=(8, 5))

    for controller, logs in all_logs.items():
        plt.plot(logs["t"], logs["effort"], label=controller)

    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity effort")
    plt.title("Control Effort Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "control_effort.png", dpi=300)
    plt.close()


def plot_reaction_weight(all_logs):
    plt.figure(figsize=(8, 5))

    if "fixed_rns" in all_logs:
        plt.plot(
            all_logs["fixed_rns"]["t"],
            all_logs["fixed_rns"]["mu"],
            label="Fixed-RNS constant mu",
            linestyle="--",
        )

    if "react_ns" in all_logs:
        plt.plot(
            all_logs["react_ns"]["t"],
            all_logs["react_ns"]["mu"],
            label="ReAct-NS adaptive mu(t)",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Reaction suppression weight")
    plt.title("Reaction Suppression Weight: Fixed-RNS vs ReAct-NS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reaction_weight_comparison.png", dpi=300)
    plt.close()


def plot_damping(all_logs):
    plt.figure(figsize=(8, 5))

    if "dls" in all_logs:
        plt.plot(
            all_logs["dls"]["t"],
            all_logs["dls"]["lambda"],
            label="DLS constant lambda",
            linestyle=":",
        )

    if "fixed_rns" in all_logs:
        plt.plot(
            all_logs["fixed_rns"]["t"],
            all_logs["fixed_rns"]["lambda"],
            label="Fixed-RNS constant lambda",
            linestyle="--",
        )

    if "react_ns" in all_logs:
        plt.plot(
            all_logs["react_ns"]["t"],
            all_logs["react_ns"]["lambda"],
            label="ReAct-NS adaptive lambda(t)",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Damping")
    plt.title("Damping Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "damping_comparison.png", dpi=300)
    plt.close()


def plot_manipulability(all_logs):
    plt.figure(figsize=(8, 5))

    for controller, logs in all_logs.items():
        plt.plot(logs["t"], logs["manipulability"], label=controller)

    plt.xlabel("Time (s)")
    plt.ylabel("Manipulability")
    plt.title("Manipulability During Motion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "manipulability.png", dpi=300)
    plt.close()


def plot_logs(all_logs):
    """Generate all plots."""
    plot_tracking_error(all_logs)
    plot_base_rotation(all_logs)
    plot_control_effort(all_logs)
    plot_reaction_weight(all_logs)
    plot_damping(all_logs)
    plot_manipulability(all_logs)

    print(f"Saved plots to: {RESULTS_DIR}")


def print_summary(results):
    print("\nSummary:")
    print("-" * 90)
    print(
        f"{'Controller':<12} | "
        f"{'Final Error (m)':>15} | "
        f"{'Max Base (deg)':>15} | "
        f"{'Effort':>12} | "
        f"{'Smoothness':>12} | "
        f"{'Success':>8}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['controller']:<12} | "
            f"{r['final_error_m']:15.4f} | "
            f"{r['max_base_angle_deg']:15.3f} | "
            f"{r['effort_integral']:12.4f} | "
            f"{r['smoothness_integral']:12.4f} | "
            f"{r['success']:8d}"
        )

    print("-" * 90)


def main():
    all_results = []
    all_logs = {}

    for controller in CONTROLLERS:
        print(f"\nRunning controller: {controller}")
        result, logs = run_trial(controller)

        all_results.append(result)
        all_logs[controller] = logs

        print(result)

    save_metrics(all_results)
    save_time_logs(all_logs)
    plot_logs(all_logs)
    print_summary(all_results)

    print("\nComparison complete.")
    print("Open these files:")
    print(" - react_ns_franka/results/metrics_single_target.csv")
    print(" - react_ns_franka/results/tracking_error.png")
    print(" - react_ns_franka/results/base_rotation.png")
    print(" - react_ns_franka/results/control_effort.png")
    print(" - react_ns_franka/results/reaction_weight_comparison.png")
    print(" - react_ns_franka/results/damping_comparison.png")
    print(" - react_ns_franka/results/manipulability.png")


if __name__ == "__main__":
    main()