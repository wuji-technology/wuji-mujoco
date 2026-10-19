#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), default="right")
    side = parser.parse_args().side
    root = Path(__file__).resolve().parent
    mjcf_path = root / "wuji_description" / "hand2" / "hand2_2" / "body" / "mjcf" / f"{side}.xml"

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.body(f"{side[0]}_mount").quat = [0, 0, 1, 0]  # Stand the hand upright.
    spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=1, height=6, rgb1=[0.12, 0.14, 0.18])
    model = spec.compile()
    data = mujoco.MjData(model)

    trajectory = np.load(root / "data" / "wave.npy")
    data.ctrl[:] = trajectory[0]
    for _ in range(100):
        mujoco.mj_step(model, data)

    start_time = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 270 if side == "right" else 90
        viewer.cam.elevation = -20
        viewer.cam.distance = max(0.35, 1.5 * model.stat.extent)
        viewer.cam.lookat[:] = model.stat.center
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[1] = 1

        while viewer.is_running():
            frame_idx = int((data.time - start_time) * 30)
            data.ctrl[:] = trajectory[frame_idx % len(trajectory)]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
