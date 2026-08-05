"""Sanity check: confirm env extras['depth'] matches the static image file."""
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def main():
    args = get_args()
    args.rl_device = args.sim_device
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 4
    env_cfg.depth.use_camera = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _, infos = env.reset()
    _, _, _, _, infos, _, _ = env.step(
        torch.zeros(env.num_envs, env.num_actions, device=env.device)
    )

    ref_path = env.cfg.depth.static_image_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    ref = torch.load(ref_path, map_location=env.device).reshape(
        env.cfg.depth.resized[0], env.cfg.depth.resized[1]
    )

    depth = infos["depth"]
    if depth is None:
        print("FAIL: infos['depth'] is None (global_counter may not align with update_interval yet)")
        print("Retrying until depth is emitted...")
        for _ in range(env.cfg.depth.update_interval):
            _, _, _, _, infos, _, _ = env.step(
                torch.zeros(env.num_envs, env.num_actions, device=env.device)
            )
            depth = infos["depth"]
            if depth is not None:
                break

    assert depth is not None, "infos['depth'] never became non-None"
    print("infos['depth'] shape:", tuple(depth.shape), "dtype:", depth.dtype)

    all_match = True
    for i in range(env.num_envs):
        match = torch.allclose(depth[i].float(), ref.float())
        max_diff = float((depth[i].float() - ref.float()).abs().max())
        print(f"  env {i}: match={match}, max_abs_diff={max_diff}")
        all_match = all_match and match

    wm_image = depth.unsqueeze(-1)
    print("wm_obs['image'] shape (after unsqueeze(-1)):", tuple(wm_image.shape))

    if all_match:
        print("PASS: static image is what the policy/world-model receives via infos['depth']")
    else:
        print("FAIL: depth image does not match static file")
        sys.exit(1)


if __name__ == "__main__":
    main()
