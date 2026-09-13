import csv
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt


XML_PATH = Path("react_ns_franka/models/franka_fr3_v2_space/scene.xml")
RESULTS_DIR = Path("react_ns_franka/results/mu_sweep")
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


class Config:
    """
    ReAct-NS reaction suppression strength:
        mu_max = [5, 10, 20, 40, 80, 120]
    """

    random_seed = 42

    # Use fewer trials for sweep because each mu_max value requires multiple simulations.
    # For final paper: 20 is acceptable. If too slow, use 10.
    num_trials = 20

    # Sweep values for ReAct-NS adaptive reaction suppression maximum.
    mu_max_values = [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]

    # Task-space controller
    kp = 1.8
    vmax = 0.30

    # Joint velocity safety limit
    qdot_max = 0.55

    # Adaptive damping
    lambda_min = 0.02
    lambda_s = 0.20
    w0 = 0.15

    # Adaptive reaction suppression
    mu_min = 1.0
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

    # Empirical base-reaction coupling weights
    coupling_weights = np.array([0.60, 0.52, 0.40, 0.32, 0.20, 0.12, 0.08])

    # Success thresholds
    precision_error_threshold = 0.25
    base_angle_threshold_deg = 33.0

    # Random target workspace
    target_x_min = 0.45
    target_x_max = 0.80

    target_y_min = -0.30
    target_y_max = 0.35

    target_z_min = 0.55
    target_z_max = 0.90


def clamp_norm(x: np.ndarray, max_norm: float) -> np.ndarray:

    norm = np.linalg.norm(x)
    if norm > max_norm:
        return x * max_norm / (norm + 1e-12)
    return x


def get_ids(model):

    joint_ids = [model.joint(name).id for name in ARM_JOINT_NAMES]
    dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]
    qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]

    ee_site_id = model.site(EE_SITE_NAME).id
    target_site_id = model.site(TARGET_SITE_NAME).id
    target_body_id = model.body("target_body").id

    actuator_map = {}
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id in joint_ids:
            actuator_map[joint_id] = act_id

    return (
        joint_ids,
        dof_ids,
        qpos_ids,
        ee_site_id,
        target_site_id,
        target_body_id,
        actuator_map,
    )


def reset_home(model, data):

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def set_target_position(model, data, target_body_id, target_position):

    model.body_pos[target_body_id] = np.asarray(target_position, dtype=float)
    mujoco.mj_forward(model, data)


def get_site_jacobian(model, data, site_id, dof_ids):

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return jacp[:, dof_ids]


def estimate_reaction_matrix(cfg: Config, n: int) -> np.ndarray:

    A = np.zeros((3, n))
    weights = cfg.coupling_weights.copy()

    if len(weights) != n:
        weights = np.linspace(0.60, 0.08, n)

    A[2, :] = weights
    A[0, :] = 0.15 * weights
    A[1, :] = 0.10 * weights

    return A


def manipulability(J: np.ndarray, cfg: Config) -> float:

    matrix = J @ J.T + cfg.eps * np.eye(J.shape[0])
    det = np.linalg.det(matrix)

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
    mu_max: float,
):

    n = J.shape[1]

    w = manipulability(J, cfg)

    mu = cfg.mu_min + (mu_max - cfg.mu_min) * np.exp(
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


def apply_velocity_controls(data, joint_ids, actuator_map, qdot_cmd):

    data.ctrl[:] = 0.0

    for local_i, joint_id in enumerate(joint_ids):
        act_id = actuator_map.get(joint_id, None)

        if act_id is not None:
            data.ctrl[act_id] = qdot_cmd[local_i]


def base_rotation_angle(data) -> float:

    if len(data.qpos) < 7:
        return 0.0

    quat = data.qpos[3:7].copy()
    quat = quat / (np.linalg.norm(quat) + 1e-12)

    qw = np.clip(quat[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(qw))

    return float(angle)


def base_angular_velocity_norm(data) -> float:

    if len(data.qvel) < 6:
        return 0.0

    return float(np.linalg.norm(data.qvel[3:6]))


def generate_random_targets(cfg: Config):

    rng = np.random.default_rng(cfg.random_seed)

    targets = []
    for _ in range(cfg.num_trials):
        x = rng.uniform(cfg.target_x_min, cfg.target_x_max)
        y = rng.uniform(cfg.target_y_min, cfg.target_y_max)
        z = rng.uniform(cfg.target_z_min, cfg.target_z_max)
        targets.append(np.array([x, y, z], dtype=float))

    return targets


def run_single_trial(mu_max: float, target_position: np.ndarray, trial_id: int):

    cfg = Config()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    reset_home(model, data)

    (
        joint_ids,
        dof_ids,
        qpos_ids,
        ee_site_id,
        target_site_id,
        target_body_id,
        actuator_map,
    ) = get_ids(model)

    set_target_position(model, data, target_body_id, target_position)

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
        xdot_des = clamp_norm(cfg.kp * e, cfg.vmax)

        J = get_site_jacobian(model, data, ee_site_id, dof_ids)
        A = estimate_reaction_matrix(cfg, len(dof_ids))

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
            mu_max=mu_max,
        )

        qdot_cmd = np.clip(qdot_cmd, -cfg.qdot_max, cfg.qdot_max)

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
            mu_log.append(float(info["mu"]))
            lambda_log.append(float(info["lambda"]))
            manip_log.append(float(info["manipulability"]))

    final_error = float(error_log[-1])
    max_base_angle_deg = float(np.rad2deg(max(base_angle_log)))
    base_omega_integral = float(np.trapz(base_omega_log, t_log))
    effort_integral = float(np.trapz(effort_log, t_log))
    smoothness_integral = float(np.trapz(smoothness_log, t_log))
    min_manipulability = float(np.min(manip_log))
    mean_mu = float(np.mean(mu_log))
    final_mu = float(mu_log[-1])
    mean_lambda = float(np.mean(lambda_log))

    precision_success = final_error < cfg.precision_error_threshold

    base_stable_success = (
        final_error < cfg.precision_error_threshold
        and max_base_angle_deg < cfg.base_angle_threshold_deg
    )

    result = {
        "trial_id": trial_id,
        "mu_max": float(mu_max),
        "target_x": float(target_position[0]),
        "target_y": float(target_position[1]),
        "target_z": float(target_position[2]),
        "final_error_m": final_error,
        "max_base_angle_deg": max_base_angle_deg,
        "base_omega_integral": base_omega_integral,
        "effort_integral": effort_integral,
        "smoothness_integral": smoothness_integral,
        "min_manipulability": min_manipulability,
        "mean_mu": mean_mu,
        "final_mu": final_mu,
        "mean_lambda": mean_lambda,
        "precision_success": int(precision_success),
        "base_stable_success": int(base_stable_success),
    }

    return result


def summarize_results(results):

    summary = []

    mu_values = sorted(set(row["mu_max"] for row in results))

    for mu_max in mu_values:
        rows = [r for r in results if r["mu_max"] == mu_max]

        item = {
            "mu_max": float(mu_max),
            "num_trials": len(rows),
            "mean_final_error_m": float(np.mean([r["final_error_m"] for r in rows])),
            "std_final_error_m": float(np.std([r["final_error_m"] for r in rows])),
            "mean_max_base_angle_deg": float(np.mean([r["max_base_angle_deg"] for r in rows])),
            "std_max_base_angle_deg": float(np.std([r["max_base_angle_deg"] for r in rows])),
            "mean_base_omega_integral": float(np.mean([r["base_omega_integral"] for r in rows])),
            "mean_effort_integral": float(np.mean([r["effort_integral"] for r in rows])),
            "mean_smoothness_integral": float(np.mean([r["smoothness_integral"] for r in rows])),
            "mean_min_manipulability": float(np.mean([r["min_manipulability"] for r in rows])),
            "mean_mu": float(np.mean([r["mean_mu"] for r in rows])),
            "mean_final_mu": float(np.mean([r["final_mu"] for r in rows])),
            "mean_lambda": float(np.mean([r["mean_lambda"] for r in rows])),
            "precision_success_rate": float(np.mean([r["precision_success"] for r in rows])),
            "base_stable_success_rate": float(np.mean([r["base_stable_success"] for r in rows])),
        }

        summary.append(item)

    return summary


def save_csv(path: Path, rows):
    
    if not rows:
        print(f"No rows to save for {path}")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def plot_line(summary, key, ylabel, title, filename):
    """Plot a metric versus mu_max."""
    x = [row["mu_max"] for row in summary]
    y = [row[key] for row in summary]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Maximum reaction suppression weight, mu_max")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()


def plot_tradeoff(summary):

    x = [row["mean_final_error_m"] for row in summary]
    y = [row["mean_max_base_angle_deg"] for row in summary]
    labels = [row["mu_max"] for row in summary]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(
            f"{label:g}",
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plt.xlabel("Mean final error (m)")
    plt.ylabel("Mean maximum base rotation (deg)")
    plt.title("Tracking Error vs Base Disturbance Tradeoff")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tradeoff_error_vs_base_rotation.png", dpi=300)
    plt.close()


def plot_success_rates(summary):

    x = [row["mu_max"] for row in summary]
    precision = [row["precision_success_rate"] for row in summary]
    base_stable = [row["base_stable_success_rate"] for row in summary]

    plt.figure(figsize=(8, 5))
    plt.plot(x, precision, marker="o", label="Precision success")
    plt.plot(x, base_stable, marker="s", label="Base-stable success")
    plt.xlabel("Maximum reaction suppression weight, mu_max")
    plt.ylabel("Success rate")
    plt.title("Success Rate vs Reaction Suppression Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "success_rate_vs_mu_max.png", dpi=300)
    plt.close()


def plot_summary(summary):

    plot_line(
        summary,
        key="mean_final_error_m",
        ylabel="Mean final error (m)",
        title="Final Error vs Reaction Suppression Strength",
        filename="final_error_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_max_base_angle_deg",
        ylabel="Mean maximum base rotation (deg)",
        title="Base Disturbance vs Reaction Suppression Strength",
        filename="base_rotation_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_base_omega_integral",
        ylabel="Mean base angular velocity integral",
        title="Base Angular Motion vs Reaction Suppression Strength",
        filename="base_omega_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_effort_integral",
        ylabel="Mean effort integral",
        title="Control Effort vs Reaction Suppression Strength",
        filename="effort_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_smoothness_integral",
        ylabel="Mean smoothness cost",
        title="Smoothness vs Reaction Suppression Strength",
        filename="smoothness_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_min_manipulability",
        ylabel="Mean minimum manipulability",
        title="Retained Manipulability vs Reaction Suppression Strength",
        filename="manipulability_vs_mu_max.png",
    )

    plot_line(
        summary,
        key="mean_mu",
        ylabel="Mean adaptive mu(t)",
        title="Mean Adaptive Reaction Weight vs mu_max",
        filename="mean_mu_vs_mu_max.png",
    )

    plot_tradeoff(summary)
    plot_success_rates(summary)


def print_summary_table(summary):
    print("\nReaction Weight Sweep Summary")
    print("-" * 150)
    print(
        f"{'mu_max':>8} | "
        f"{'Err mean':>9} | "
        f"{'Base deg':>9} | "
        f"{'Omega int':>10} | "
        f"{'Effort':>10} | "
        f"{'Smooth':>10} | "
        f"{'Manip':>9} | "
        f"{'Mean mu':>9} | "
        f"{'Prec SR':>8} | "
        f"{'Base SR':>8}"
    )
    print("-" * 150)

    for row in summary:
        print(
            f"{row['mu_max']:8.1f} | "
            f"{row['mean_final_error_m']:9.4f} | "
            f"{row['mean_max_base_angle_deg']:9.3f} | "
            f"{row['mean_base_omega_integral']:10.4f} | "
            f"{row['mean_effort_integral']:10.4f} | "
            f"{row['mean_smoothness_integral']:10.4f} | "
            f"{row['mean_min_manipulability']:9.4f} | "
            f"{row['mean_mu']:9.3f} | "
            f"{row['precision_success_rate']:8.2f} | "
            f"{row['base_stable_success_rate']:8.2f}"
        )

    print("-" * 150)


def main():
    cfg = Config()

    print("Starting ReAct-NS reaction weight sweep")
    print("XML:", XML_PATH)
    print("Number of random targets:", cfg.num_trials)
    print("mu_max values:", cfg.mu_max_values)
    print("Results directory:", RESULTS_DIR)

    targets = generate_random_targets(cfg)

    all_results = []

    for mu_max in cfg.mu_max_values:
        print(f"\nmu_max = {mu_max}")

        for trial_id, target in enumerate(targets):
            print(
                f"  Trial {trial_id + 1}/{cfg.num_trials} | "
                f"target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})",
                end=" "
            )

            result = run_single_trial(
                mu_max=mu_max,
                target_position=target,
                trial_id=trial_id,
            )

            all_results.append(result)

            print(
                f"err={result['final_error_m']:.4f}, "
                f"base={result['max_base_angle_deg']:.2f} deg, "
                f"stable={result['base_stable_success']}"
            )

    summary = summarize_results(all_results)

    save_csv(RESULTS_DIR / "mu_sweep_all_trials.csv", all_results)
    save_csv(RESULTS_DIR / "mu_sweep_summary.csv", summary)

    plot_summary(summary)

    print_summary_table(summary)

    print("\nReaction weight sweep complete.")
    print("Open these files:")
    print(" - react_ns_franka/results/mu_sweep/mu_sweep_all_trials.csv")
    print(" - react_ns_franka/results/mu_sweep/mu_sweep_summary.csv")
    print(" - react_ns_franka/results/mu_sweep/tradeoff_error_vs_base_rotation.png")
    print(" - react_ns_franka/results/mu_sweep/final_error_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/base_rotation_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/base_omega_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/effort_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/smoothness_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/manipulability_vs_mu_max.png")
    print(" - react_ns_franka/results/mu_sweep/success_rate_vs_mu_max.png")


if __name__ == "__main__":
    main()