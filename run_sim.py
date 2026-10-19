#!/usr/bin/env python3
"""Run the hand wave simulation."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np
from numpy.typing import ArrayLike

ROOT = Path(__file__).resolve().parent
HAND_MODELS = {
    "hand1": Path("models/wuji_hand_description/mjcf"),
    "hand2": Path("models/wuji_description/hand2/hand2_2/body/mjcf"),
}


def _vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(size, result.item())
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or a vector of shape ({size},)")
    return result


class HandSimulation:
    """Load a hand model and apply joint MIT torque control."""

    def __init__(self, hand: str = "hand1", side: str = "right"):
        if hand not in HAND_MODELS:
            raise ValueError(f"Unknown hand {hand!r}; choose from {tuple(HAND_MODELS)}")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.hand, self.side = hand, side
        model_path = ROOT / HAND_MODELS[hand] / f"{side}.xml"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing model: {model_path}\n"
                "Run: git submodule update --init --recursive"
            )
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.modelname = f"wuji-{hand}-{side}"
        if hand == "hand2":
            # Stand hand2 upright by rotating its mount 180 degrees about Y.
            mount = spec.body(f"{side[0]}_mount")
            upright = np.array([0.0, 0.0, 1.0, 0.0])
            rotation = np.empty(4)
            mujoco.mju_mulQuat(rotation, upright, mount.quat)
            mount.quat = rotation
            mount.pos = np.array([-mount.pos[0], mount.pos[1], -mount.pos[2]])
        spec.add_texture(
            name="simulation_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536,
            rgb1=[0.18, 0.22, 0.28], rgb2=[0.08, 0.10, 0.14],
        )
        self.model = model = spec.compile()
        model.vis.headlight.ambient[:] = 0.3
        model.vis.headlight.diffuse[:] = 0.6
        model.vis.headlight.specular[:] = 0.4
        joint_ids = model.actuator_trnid[:, 0].copy()
        if (
            model.nu != 20
            or model.na != 0
            or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
            or np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE)
            or len(np.unique(joint_ids)) != model.nu
            or not np.allclose(model.actuator_gear[:, 0], 1.0)
            or not np.allclose(model.actuator_gear[:, 1:], 0.0)
        ):
            raise ValueError("Expected 20 hinge joints with independent unit-gear actuators")
        self.qpos_ids = model.jnt_qposadr[joint_ids].copy()
        self.dof_ids = model.jnt_dofadr[joint_ids].copy()
        self.default_kp = model.actuator_gainprm[:, 0].copy()
        self.default_kd = -model.actuator_biasprm[:, 2].copy()
        self.joint_ranges = model.jnt_range[joint_ids].copy()
        if not np.all(model.jnt_limited[joint_ids]):
            raise ValueError("Expected finite position limits for every controlled joint")

        self.torque_ranges = np.column_stack((np.full(model.nu, -np.inf),
                                              np.full(model.nu, np.inf)))
        for limited, ranges in (
            (model.actuator_forcelimited, model.actuator_forcerange),
            (model.jnt_actfrclimited[joint_ids], model.jnt_actfrcrange[joint_ids]),
        ):
            mask = np.asarray(limited, dtype=bool)
            self.torque_ranges[mask, 0] = np.maximum(self.torque_ranges[mask, 0], ranges[mask, 0])
            self.torque_ranges[mask, 1] = np.minimum(self.torque_ranges[mask, 1], ranges[mask, 1])
        if (not np.all(np.isfinite(self.torque_ranges))
                or np.any(self.torque_ranges[:, 0] >= self.torque_ranges[:, 1])):
            raise ValueError("Expected valid torque limits for every controlled joint")

        model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[:] = 0.0
        model.actuator_gainprm[:, 0] = 1.0
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[:] = 0.0
        model.actuator_ctrllimited[:] = True
        model.actuator_ctrlrange[:] = self.torque_ranges
        self.data = mujoco.MjData(model)
        self.neutral = np.clip(np.zeros(model.nu), *self.joint_ranges.T)
        self.data.qpos[self.qpos_ids] = self.neutral
        mujoco.mj_forward(model, self.data)
        low, high = self.joint_ranges.T
        travel = np.where(high - self.neutral >= self.neutral - low,
                          high - self.neutral, low - self.neutral)
        self.amplitude = 0.15 * travel

    def step(self, q_des, dq_des=0.0, kp=None, kd=None, tau_ff=0.0) -> np.ndarray:
        """Apply one MIT command and advance one simulation step."""
        n = self.model.nu
        q_des = np.clip(_vector(q_des, n, "q_des"), *self.joint_ranges.T)
        dq_des = _vector(dq_des, n, "dq_des")
        kp = _vector(self.default_kp if kp is None else kp, n, "kp")
        kd = _vector(self.default_kd if kd is None else kd, n, "kd")
        tau_ff = _vector(tau_ff, n, "tau_ff")
        if np.any(kp < 0) or np.any(kd < 0):
            raise ValueError("kp and kd must be nonnegative")
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
        if not np.all(np.isfinite(tau)):
            raise ValueError("MIT torque overflow; reduce command magnitudes")
        self.data.ctrl[:] = np.clip(tau, *self.torque_ranges.T)
        mujoco.mj_step(self.model, self.data)
        if (not np.all(np.isfinite(self.data.qpos))
                or not np.all(np.isfinite(self.data.qvel))
                or any(w.number for w in self.data.warning)):
            raise RuntimeError("MuJoCo reported unstable dynamics; reduce gains or commands")
        return self.data.ctrl.copy()


def configure_camera(sim: HandSimulation, camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = sim.model.stat.center
    camera.distance = max(0.35, 1.5 * sim.model.stat.extent)
    if sim.hand == "hand1":
        camera.azimuth = 180
    else:
        # Face the palm on either hand.
        camera.azimuth = 270 if sim.side == "right" else 90
    camera.elevation = -20


def wave_target(sim: HandSimulation) -> tuple[np.ndarray, np.ndarray]:
    """A four-second wave inside the selected hand's joint limits."""
    omega = 2 * np.pi / 4.0
    phase = omega * sim.data.time
    return (sim.neutral + sim.amplitude * (1 - np.cos(phase)),
            sim.amplitude * omega * np.sin(phase))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=HAND_MODELS, default="hand1")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--headless", action="store_true", help="Run without a window")
    parser.add_argument("--duration", type=float, help="Simulation seconds (headless default: 5)")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")

    try:
        sim = HandSimulation(args.hand, args.side)
        print(f"MuJoCo {mujoco.__version__} | {args.hand} {args.side} | wave")
        duration = args.duration if args.duration is not None else (
            5.0 if args.headless else float("inf")
        )
        context = nullcontext(None)
        if not args.headless:
            from mujoco import viewer as mj_viewer

            context = mj_viewer.launch_passive(sim.model, sim.data)
        with context as viewer:
            if viewer is not None:
                configure_camera(sim, viewer.cam)
                viewer.opt.geomgroup[:] = 0
                viewer.opt.geomgroup[1] = 1
            started = time.monotonic()
            while sim.data.time < duration - 1e-10 and (viewer is None or viewer.is_running()):
                q_des, dq_des = wave_target(sim)
                sim.step(q_des, dq_des)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0, started + sim.data.time - time.monotonic()))
        print(f"Completed {sim.data.time:.3f}s of wave simulation")
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
