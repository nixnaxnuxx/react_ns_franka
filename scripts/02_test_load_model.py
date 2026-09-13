import time
from pathlib import Path

import mujoco
import mujoco.viewer


XML_PATH = Path("react_ns_franka/models/franka_fr3_v2_space/scene.xml")


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Cannot find XML: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    print("Model loaded successfully.")
    print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()