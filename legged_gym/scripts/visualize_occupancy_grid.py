"""Visualize the occupancy grid during a play-style simulation.

Saves top-down grid PNGs to logs/occupancy_grid_debug/.

Usage:
    conda activate safety
    python legged_gym/scripts/visualize_occupancy_grid.py \
        --task go1_amp --terrain corridor --headless
"""

import isaacgym

import os
import inspect
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def save_grid_image(grid_np, step, save_dir, cfg):
    """Save the occupancy grid as a top-down PNG with robot marker."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    ax.imshow(grid_np.T, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="auto")

    robot_row = int(cfg.x_backward / cfg.resolution)
    robot_col = int(cfg.y_right / cfg.resolution)
    ax.plot(robot_row, robot_col, "ro", markersize=6, label="robot")

    ax.set_xlabel("x (forward ->)")
    ax.set_ylabel("y (left ->)")
    ax.set_title(f"Occupancy Grid  step={step}")
    ax.legend(loc="upper right", fontsize=8)

    path = os.path.join(save_dir, f"frame_{step:04d}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]

    env_cfg.depth.use_camera = True

    if args.terrain not in ["slope", "stair", "gap", "climb", "crawl", "tilt", "corridor"]:
        args.terrain = "corridor"
    env_cfg.terrain.terrain_proportions = {
        "slope": [0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        "stair": [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        "gap": [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        "climb": [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        "tilt": [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        "crawl": [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
        "corridor": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
    }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.flat_lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _, infos = env.reset()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "WMP"
    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    history_length = 5
    trajectory_history = torch.zeros(
        env.num_envs, history_length,
        env.num_obs - env.privileged_dim - env.height_dim - 3,
        device=env.device,
    )
    obs_without_command = torch.cat((
        obs[:, env.privileged_dim:env.privileged_dim + 6],
        obs[:, env.privileged_dim + 9:-env.height_dim],
    ), dim=1)
    trajectory_history = torch.cat(
        (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1,
    )

    world_model = ppo_runner._world_model.to(env.device)
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_obs = {
        "prop": obs[:, env.privileged_dim:env.privileged_dim + env.cfg.env.prop_dim],
        "is_first": wm_is_first,
        "wm_state": None,
    }
    if env.cfg.depth.use_camera:
        wm_obs["image"] = torch.zeros(
            (env.num_envs,) + env.cfg.depth.resized + (1,), device=world_model.device,
        )
    wm_feature = torch.zeros(env.num_envs, ppo_runner.wm_feature_dim, device=env.device)

    occ_cfg = getattr(env.cfg, 'occupancy_grid', None)
    if occ_cfg is not None and occ_cfg.enabled:
        occ_grid = torch.zeros(env.num_envs, occ_cfg.grid_H, occ_cfg.grid_W, device=env.device)
    else:
        occ_grid = None

    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "occupancy_grid_debug")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving grid images to {save_dir}")

    num_steps = 200
    frame_idx = 0

    for i in range(num_steps):
        if env.global_counter % wm_update_interval == 0:
            wm_future_preds, wm_hidden_state, wm_state = world_model(wm_obs)
            wm_feature = torch.cat(
                [wm_hidden_state, wm_future_preds], dim=-1,
            ).to(world_model.device)
            wm_obs["wm_state"] = wm_state
            wm_is_first[:] = 0

        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach(), wm_feature.detach(), grid=occ_grid)

        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        wm_obs["prop"] = obs[
            :, env.privileged_dim:env.privileged_dim + env.cfg.env.prop_dim
        ].to(world_model.device)
        wm_obs["is_first"] = wm_is_first

        if env.global_counter % wm_update_interval == 0:
            if env.cfg.depth.use_camera:
                wm_obs["image"] = infos["depth"].unsqueeze(-1).to(world_model.device)

        reset_env_ids_np = reset_env_ids.cpu().numpy()
        if len(reset_env_ids_np) > 0:
            wm_is_first[reset_env_ids_np] = 1

        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.cat((
            obs[:, env.privileged_dim:env.privileged_dim + 6],
            obs[:, env.privileged_dim + 9:-env.height_dim],
        ), dim=1)
        trajectory_history = torch.cat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1,
        )

        new_grid = infos.get("occupancy_grid", None)
        if new_grid is not None:
            occ_grid = new_grid.to(env.device)
            if env.global_counter % wm_update_interval == 0:
                grid_np = new_grid[0].cpu().numpy()
                save_grid_image(grid_np, i, save_dir, occ_cfg)
                n_occ = (grid_np > 0.5).sum()
                print(f"Step {i:4d}: {n_occ:5d} occupied cells")
                frame_idx += 1

    print(f"\nDone. Saved {frame_idx} frames to {save_dir}")


if __name__ == "__main__":
    main()
