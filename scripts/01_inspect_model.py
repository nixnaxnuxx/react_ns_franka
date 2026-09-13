import mujoco
from pathlib import Path


XML_PATH = Path("react_ns_franka/models/franka_fr3_v2_space/scene.xml")


def name_of(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"unnamed_{obj_id}"


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Cannot find XML: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)

    print("\nMODEL LOADED")
    print("XML:", XML_PATH)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print("nsite:", model.nsite)
    print("nbody:", model.nbody)

    print("\nJOINTS")
    for i in range(model.njnt):
        name = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        qpos_adr = model.jnt_qposadr[i]
        dof_adr = model.jnt_dofadr[i]
        print(f"{i:02d} | name={name:30s} | type={jtype} | qposadr={qpos_adr} | dofadr={dof_adr}")

    print("\nACTUATORS")
    for i in range(model.nu):
        name = name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        joint_id = trnid[0]
        joint_name = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "none"
        print(f"{i:02d} | name={name:30s} | joint={joint_name}")

    print("\nSITES")
    for i in range(model.nsite):
        name = name_of(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"{i:02d} | name={name}")

    print("\nBODIES")
    for i in range(model.nbody):
        name = name_of(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:02d} | name={name}")


if __name__ == "__main__":
    main()