# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Collision avoidance evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(arr, window=50):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(arr[lo:i + 1])
    return out


def _policy_color(policy_name):
    n = policy_name.lower()
    if 'null' in n or 'safety' in n:
        return '#2563eb'
    if 'lagrangian' in n or 'lag' in n:
        return '#16a34a'
    return '#e05c34'


def _save_collision_plots(out_dir, policy_name, terrain, dt,
                          ca_log, all_env_flat, all_env_stacks,
                          episode_aggregates, episode_reset_steps):
    color  = _policy_color(policy_name)
    dcolor = '#1a3a6b' if color == '#2563eb' else ('#0d5c2a' if color == '#16a34a' else '#8b2a10')
    prefix = f"{policy_name} | {terrain}"

    n_steps = len(ca_log['safety_value'])
    t = np.arange(n_steps) * dt
    ep_idx = list(range(len(episode_aggregates)))

    unique_reset_steps = sorted(set(episode_reset_steps))

    def _ep_lines(ax):
        for rs in unique_reset_steps:
            if rs < n_steps:
                ax.axvline(x=rs * dt, color='#9ca3af', linestyle=':', linewidth=0.5, alpha=0.6)

    # 1. sv timeseries (mean across envs)
    sv_arr = np.array(ca_log['safety_value'])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sv_arr, linewidth=0.8, color=color, label='Mean sv(t)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='sv = 0')
    _ep_lines(ax)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Safety value sv(t)')
    ax.set_title(f'{prefix} — Safety Value Timeseries (Mean across envs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'sv_timeseries.png'), dpi=150); plt.close(fig)

    # 2. sv all-env timeseries with +/- std band
    if all_env_stacks['safety_value']:
        sv_stack = np.array(all_env_stacks['safety_value'])
        sv_mean = sv_stack.mean(axis=1)
        sv_std  = sv_stack.std(axis=1)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t[:len(sv_mean)], sv_mean, linewidth=0.8, color=color, label='Mean sv(t) all envs')
        ax.fill_between(t[:len(sv_mean)], sv_mean - sv_std, sv_mean + sv_std, alpha=0.2, color=color)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
        _ep_lines(ax)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Safety value sv(t)')
        ax.set_title(f'{prefix} — Safety Value (All Envs, Mean \u00b1 1 std)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'sv_allenv_timeseries.png'), dpi=150); plt.close(fig)

    # 3. yaw rate timeseries (mean across envs)
    yr_arr = np.array(ca_log['yaw_rate'])
    yr_roll = _rolling_mean(yr_arr)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, yr_arr, linewidth=0.6, color=color, alpha=0.5, label='Mean yaw rate')
    ax.plot(t, yr_roll, linewidth=1.4, color=dcolor, label='Rolling mean (50)')
    _ep_lines(ax)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Yaw rate (rad/s)')
    ax.set_title(f'{prefix} — Yaw Rate Timeseries (Mean across envs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'yaw_rate_timeseries.png'), dpi=150); plt.close(fig)

    # 4. yaw rate all-env timeseries with +/- std band
    if all_env_stacks['yaw_rate']:
        yr_stack = np.array(all_env_stacks['yaw_rate'])
        yr_mean = yr_stack.mean(axis=1)
        yr_std  = yr_stack.std(axis=1)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t[:len(yr_mean)], yr_mean, linewidth=0.8, color=color, label='Mean yaw rate all envs')
        ax.fill_between(t[:len(yr_mean)], yr_mean - yr_std, yr_mean + yr_std, alpha=0.2, color=color)
        _ep_lines(ax)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Yaw rate (rad/s)')
        ax.set_title(f'{prefix} — Yaw Rate (All Envs, Mean \u00b1 1 std)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'yaw_rate_allenv_timeseries.png'), dpi=150); plt.close(fig)

    # 5. lateral velocity timeseries (mean across envs)
    lv_arr = np.abs(np.array(ca_log['lateral_velocity']))
    lv_roll = _rolling_mean(lv_arr)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, lv_arr, linewidth=0.6, color=color, alpha=0.5, label='|Lateral vel|')
    ax.plot(t, lv_roll, linewidth=1.4, color=dcolor, label='Rolling mean (50)')
    _ep_lines(ax)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('|Lateral velocity| (m/s)')
    ax.set_title(f'{prefix} — Lateral Evasion Velocity (Mean across envs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'lateral_velocity_timeseries.png'), dpi=150); plt.close(fig)

    # 6. vel tracking error + cmd vel secondary axis (mean across envs)
    vte_arr = np.array(ca_log['vel_tracking_error'])
    cmd_arr = np.array(ca_log['cmd_vel_x'])
    vte_roll = _rolling_mean(vte_arr)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.plot(t, vte_arr, linewidth=0.6, color=color, alpha=0.5, label='Tracking error')
    ax1.plot(t, vte_roll, linewidth=1.4, color=dcolor, label='Rolling mean (50)')
    ax2.plot(t, cmd_arr, linewidth=0.8, color='#9ca3af', alpha=0.6, label='cmd vel x')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('|vel error| (m/s)'); ax2.set_ylabel('cmd vel x (m/s)')
    ax1.set_title(f'{prefix} — Velocity Tracking Error (Mean across envs)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'vel_tracking_error_timeseries.png'), dpi=150); plt.close(fig)

    # 7. collision rate timeseries (fraction of envs colliding per step)
    col_rate = np.array(ca_log['collision_event'])
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.fill_between(t, 0, col_rate, color='red', alpha=0.5)
    ax.plot(t, col_rate, linewidth=0.6, color='red', alpha=0.8)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Collision fraction')
    total_col_steps = int((col_rate > 0).sum())
    ax.set_title(f'{prefix} — Collision Rate per Step (steps with any collision: {total_col_steps})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'collision_events_timeseries.png'), dpi=150); plt.close(fig)

    if episode_aggregates:
        ep_sv  = [e['mean_sv'] for e in episode_aggregates]
        ep_yr  = [e['mean_abs_yaw'] for e in episode_aggregates]

        # 8. episode sv bar
        fig, ax = plt.subplots(figsize=(max(6, len(ep_sv) * 0.4 + 2), 4))
        ax.bar(ep_idx, ep_sv, color=color, edgecolor='white', linewidth=0.4)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='sv = 0')
        ax.set_xlabel('Episode'); ax.set_ylabel('Mean sv(t)')
        ax.set_title(f'{prefix} — Mean Safety Value per Episode')
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'episode_sv_bar.png'), dpi=150); plt.close(fig)

        # 9. episode yaw bar
        fig, ax = plt.subplots(figsize=(max(6, len(ep_yr) * 0.4 + 2), 4))
        ax.bar(ep_idx, ep_yr, color=color, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('Episode'); ax.set_ylabel('Mean |yaw rate| (rad/s)')
        ax.set_title(f'{prefix} — Mean Absolute Yaw Rate per Episode')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'episode_yaw_bar.png'), dpi=150); plt.close(fig)

        # 10. sv vs yaw scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ep_sv, ep_yr, color=color, edgecolors='white', linewidths=0.5, s=60, alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Mean sv(t) per episode'); ax.set_ylabel('Mean |yaw rate| (rad/s)')
        ax.set_title(f'{prefix} — Safety Value vs. Evasive Steering')
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'sv_vs_yaw_scatter.png'), dpi=150); plt.close(fig)

    # 11. sv histogram (all envs, all timesteps)
    sv_flat = all_env_flat['safety_value']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sv_flat, bins=50, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(sv_flat.mean(), color=dcolor, linestyle='--', linewidth=1.5, label=f'Mean = {sv_flat.mean():.4f}')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='sv = 0')
    ax.set_xlabel('sv(t)'); ax.set_ylabel('Count')
    ax.set_title(f'{prefix} — Safety Value Distribution (all envs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'sv_histogram.png'), dpi=150); plt.close(fig)

    # 12. yaw rate histogram (all envs, all timesteps)
    abs_yr = np.abs(all_env_flat['yaw_rate'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(abs_yr, bins=50, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(abs_yr.mean(), color=dcolor, linestyle='--', linewidth=1.5, label=f'Mean = {abs_yr.mean():.4f}')
    ax.set_xlabel('|Yaw rate| (rad/s)'); ax.set_ylabel('Count')
    ax.set_title(f'{prefix} — Yaw Rate Distribution (all envs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'yaw_rate_histogram.png'), dpi=150); plt.close(fig)

    print(f'[Plots] All collision avoidance figures saved to: {out_dir}')


def _write_collision_summary(out_dir, policy_name, terrain, dt,
                              all_env_flat, episode_aggregates,
                              num_envs):
    sv_arr  = all_env_flat['safety_value']
    yr_arr  = np.abs(all_env_flat['yaw_rate'])
    lv_arr  = np.abs(all_env_flat['lateral_velocity'])
    fv_arr  = all_env_flat['forward_velocity']
    vte_arr = all_env_flat['vel_tracking_error']
    col_arr = all_env_flat['collision_event'].astype(bool)

    n_samples = len(sv_arr)
    n_steps   = n_samples // max(num_envs, 1)
    n_ep    = len(episode_aggregates)
    ep_col      = [e['collision_count'] for e in episode_aggregates]
    ep_yr_mean  = [e['mean_abs_yaw'] for e in episode_aggregates]
    ep_sv_mean  = [e['mean_sv'] for e in episode_aggregates]
    ep_returns  = [e.get('episode_return', float('nan')) for e in episode_aggregates]

    lines = []
    lines.append('=' * 60)
    lines.append(f'  COLLISION AVOIDANCE EVALUATION — {policy_name}')
    lines.append(f'  Terrain    : {terrain}')
    lines.append(f'  Num envs   : {num_envs}')
    lines.append(f'  Episodes   : {n_ep}')
    lines.append(f'  Sim steps  : {n_steps}')
    lines.append(f'  Total samples (steps x envs): {n_samples}')
    lines.append('=' * 60)
    lines.append('')
    lines.append('── Safety Value (sv) ' + '─' * 39)
    lines.append(f'  Mean sv(t) across all samples : {sv_arr.mean():.5f} \u00b1 {sv_arr.std():.5f}')
    lines.append(f'  Mean sv(t) per episode        : {np.mean(ep_sv_mean):.5f} \u00b1 {np.std(ep_sv_mean):.5f}')
    lines.append(f'  % of samples with sv < 0     : {100*(sv_arr<0).mean():.1f}%   (active barrier violation)')
    lines.append(f'  % of samples with sv < -0.05 : {100*(sv_arr<-0.05).mean():.1f}%   (proxy collision events)')
    lines.append(f'  Min sv(t) seen                : {sv_arr.min():.5f}')
    lines.append('')
    lines.append('── Evasive Steering (yaw) ' + '─' * 34)
    lines.append(f'  Mean |yaw rate|               : {yr_arr.mean():.4f} \u00b1 {yr_arr.std():.4f} rad/s')
    lines.append(f'  Peak |yaw rate|               : {yr_arr.max():.4f} rad/s')
    lines.append(f'  Mean |yaw rate| per episode   : {np.mean(ep_yr_mean):.4f} \u00b1 {np.std(ep_yr_mean):.4f} rad/s')
    lines.append('')
    lines.append('── Lateral Evasion ' + '─' * 40)
    lines.append(f'  Mean |lateral velocity|       : {lv_arr.mean():.4f} \u00b1 {lv_arr.std():.4f} m/s')
    lines.append('')
    lines.append('── Task Performance ' + '─' * 40)
    lines.append(f'  Mean forward velocity         : {fv_arr.mean():.4f} \u00b1 {fv_arr.std():.4f} m/s')
    lines.append(f'  Mean vel tracking error       : {vte_arr.mean():.4f} \u00b1 {vte_arr.std():.4f} m/s')
    ret_valid = [r for r in ep_returns if not np.isnan(r)]
    if ret_valid:
        lines.append(f'  Mean return per episode       : {np.mean(ret_valid):.2f} \u00b1 {np.std(ret_valid):.2f}')
    lines.append('')
    lines.append('── Collision Events ' + '─' * 40)
    lines.append(f'  Total collision samples       : {col_arr.sum()} (env-timesteps)')
    lines.append(f'  Collision rate                : {100*col_arr.mean():.1f}% of all env-timesteps')
    eps_with_col = sum(1 for c in ep_col if c > 0)
    lines.append(f'  Episodes with collision       : {eps_with_col} / {n_ep}  ({100*eps_with_col/max(n_ep,1):.1f}%)')
    lines.append('')
    lines.append('=' * 60)

    text = '\n'.join(lines)
    print(text)
    path = os.path.join(out_dir, 'eval_summary_collision.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'[Summary] Saved to: {path}')


def play(args):
    global POLICY_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.min_difficulty = 0.5
    env_cfg.terrain.max_difficulty = 0.9
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    # env_cfg.terrain.mesh_type = 'plane'
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt', 'corridor', 'scattered']):
        print('terrain should be one of slope, stair, gap, climb, crawl, tilt, corridor, and scattered, set to climb as default')
        args.terrain = 'climb'
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
        'corridor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
        'scattered': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
     }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    env_cfg.depth.use_camera = True
    env_cfg.viewer.lookat_id =  6

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _, infos = env.reset()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'WMP'


    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    frames_record_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'plot_frames_' + str(args.terrain))
    # frames_record_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'video_frames_' + str(args.terrain))
    data_record_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'data_' + str(args.terrain))
    
    robot_index = env_cfg.viewer.lookat_id # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = env.max_episode_length + 1 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                            env.privileged_dim - env.height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)

    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_obs = {
        "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
        "is_first": wm_is_first,
        "wm_state": None,
    }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)), device=world_model.device)

    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    occ_cfg = getattr(env.cfg, 'occupancy_grid', None)
    if occ_cfg is not None and occ_cfg.enabled:
        occ_grid = torch.zeros(env.num_envs, occ_cfg.grid_H, occ_cfg.grid_W, device=env.device)
    else:
        occ_grid = None

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)

    # ── Collision avoidance metric accumulators ───────────────────────────────
    # Collision detection: geometric intersection between every robot rigid
    # body link and each cylindrical obstacle (XY distance < radius + tol,
    # body z below obstacle top).
    CYLINDER_HEIGHT    = 0.3
    BODY_COLLISION_TOL = 0.03

    _has_obs_data = (hasattr(env, 'obstacle_positions') and
                     hasattr(env, 'obstacle_radii') and
                     hasattr(env, 'has_obstacles') and
                     hasattr(env, 'rigid_body_pos'))
    if not _has_obs_data:
        print('[CA eval] env missing obstacle_positions/radii/has_obstacles/rigid_body_pos — '
              'falling back to sv < -0.05 as collision proxy')

    TARGET_EPISODES = 1000

    ca_log = {
        'safety_value': [], 'yaw_rate': [], 'lateral_velocity': [],
        'forward_velocity': [], 'cmd_vel_x': [], 'vel_tracking_error': [],
        'min_obstacle_dist': [], 'collision_event': [],
    }
    _all_env_stacks = {
        'safety_value': [], 'yaw_rate': [], 'lateral_velocity': [],
        'forward_velocity': [], 'cmd_vel_x': [], 'vel_tracking_error': [],
        'min_obstacle_dist': [], 'collision_event': [],
    }
    episode_reset_steps = []
    episode_aggregates  = []
    _ep_data = [{'sv': [], 'yr': [], 'lv': [], 'fv': [], 'vte': [],
                 'col': 0, 'rew': 0.0} for _ in range(env.num_envs)]

    total_completed = 0
    step            = 0

    print(f'[CA eval] Starting {TARGET_EPISODES}-episode run across {env.num_envs} envs...')

    while total_completed < TARGET_EPISODES:
        step += 1
        i = step  # keep variable name consistent with existing logger checks
        if (env.global_counter % wm_update_interval == 0):

            wm_future_preds, wm_hidden_state, wm_state = world_model(wm_obs)
            wm_feature = torch.cat(
                [wm_hidden_state, wm_future_preds], dim=-1).to(world_model.device)
            wm_obs['wm_state'] = wm_state

            wm_is_first[:] = 0

        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach(), wm_feature.detach(), grid=occ_grid)

        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # ── Per-timestep collision avoidance collection (all envs) ─────────
        _sv_all  = env.safety_values.cpu().numpy() if hasattr(env, 'safety_values') else np.zeros(env.num_envs)
        _yr_all  = env.base_ang_vel[:, 2].cpu().numpy()
        _lv_all  = env.base_lin_vel[:, 1].cpu().numpy()
        _fv_all  = env.base_lin_vel[:, 0].cpu().numpy()
        _cmd_all = env.commands[:, 0].cpu().numpy()
        _vte_all = np.abs(_cmd_all - _fv_all)

        _col_all = np.zeros(env.num_envs, dtype=bool)
        _od_all  = np.full(env.num_envs, np.nan)
        if _has_obs_data:
            try:
                _body_pos  = env.rigid_body_pos
                _obs_pos   = env.obstacle_positions
                _obs_rad   = env.obstacle_radii
                _obs_mask_t = env.obstacle_mask if hasattr(env, 'obstacle_mask') else torch.ones(_obs_pos.shape[:2], dtype=torch.bool, device=env.device)
                _has_obs_t  = env.has_obstacles

                _body_xy  = _body_pos[:, :, :2].unsqueeze(2)
                _obs_xy   = _obs_pos[:, :, :2].unsqueeze(1)
                _xy_dist  = torch.norm(_body_xy - _obs_xy, dim=-1)

                _obs_top_z = _obs_pos[:, :, 2] + CYLINDER_HEIGHT
                _z_ok = _body_pos[:, :, 2].unsqueeze(2) <= (_obs_top_z.unsqueeze(1) + BODY_COLLISION_TOL)

                _touching = (
                    (_xy_dist < _obs_rad.unsqueeze(1).unsqueeze(2) + BODY_COLLISION_TOL)
                    & _z_ok
                    & _obs_mask_t.unsqueeze(1)
                    & _has_obs_t.unsqueeze(1).unsqueeze(2)
                )
                _col_all = _touching.any(dim=1).any(dim=1).cpu().numpy()

                _bot_pos = env.root_states[:, :3]
                _delta   = _obs_pos - _bot_pos.unsqueeze(1)
                _dist    = torch.norm(_delta, dim=-1)
                _h       = _dist - 2.0 * _obs_rad.unsqueeze(-1) - 0.35
                _h       = _h.masked_fill(~_obs_mask_t, 1e6)
                _h       = _h.masked_fill(~_has_obs_t.unsqueeze(1), 1e6)
                _od_all  = _h.min(dim=1).values.cpu().numpy()
            except Exception:
                _col_all = (_sv_all < -0.05)
        else:
            _col_all = (_sv_all < -0.05)

        ca_log['safety_value'].append(float(_sv_all.mean()))
        ca_log['yaw_rate'].append(float(_yr_all.mean()))
        ca_log['lateral_velocity'].append(float(_lv_all.mean()))
        ca_log['forward_velocity'].append(float(_fv_all.mean()))
        ca_log['cmd_vel_x'].append(float(_cmd_all.mean()))
        ca_log['vel_tracking_error'].append(float(_vte_all.mean()))
        ca_log['min_obstacle_dist'].append(float(np.nanmean(_od_all)))
        ca_log['collision_event'].append(float(_col_all.mean()))

        _all_env_stacks['safety_value'].append(_sv_all.copy())
        _all_env_stacks['yaw_rate'].append(_yr_all.copy())
        _all_env_stacks['lateral_velocity'].append(_lv_all.copy())
        _all_env_stacks['forward_velocity'].append(_fv_all.copy())
        _all_env_stacks['cmd_vel_x'].append(_cmd_all.copy())
        _all_env_stacks['vel_tracking_error'].append(_vte_all.copy())
        _all_env_stacks['min_obstacle_dist'].append(_od_all.copy())
        _all_env_stacks['collision_event'].append(_col_all.copy())

        _rews_np = rews.cpu().numpy()
        for _e in range(env.num_envs):
            _ep_data[_e]['sv'].append(float(_sv_all[_e]))
            _ep_data[_e]['yr'].append(abs(float(_yr_all[_e])))
            _ep_data[_e]['lv'].append(abs(float(_lv_all[_e])))
            _ep_data[_e]['fv'].append(float(_fv_all[_e]))
            _ep_data[_e]['vte'].append(float(_vte_all[_e]))
            if _col_all[_e]:
                _ep_data[_e]['col'] += 1
            _ep_data[_e]['rew'] += float(_rews_np[_e])

        _dones_np = dones.cpu().numpy()
        for _e in range(env.num_envs):
            if _dones_np[_e]:
                ep = _ep_data[_e]
                if ep['sv']:
                    episode_reset_steps.append(step)
                    episode_aggregates.append({
                        'mean_sv':          float(np.mean(ep['sv'])),
                        'mean_abs_yaw':     float(np.mean(ep['yr'])),
                        'peak_abs_yaw':     float(np.max(ep['yr']))  if ep['yr'] else 0.0,
                        'mean_abs_lat_vel': float(np.mean(ep['lv'])),
                        'mean_fwd_vel':     float(np.mean(ep['fv'])),
                        'mean_vte':         float(np.mean(ep['vte'])),
                        'collision_count':  ep['col'],
                        'episode_length':   len(ep['sv']),
                        'episode_return':   ep['rew'],
                    })
                _ep_data[_e] = {'sv': [], 'yr': [], 'lv': [], 'fv': [], 'vte': [],
                                'col': 0, 'rew': 0.0}
                total_completed += 1

        # Progress logging every 50 completed episodes
        if total_completed > 0 and total_completed % 50 == 0:
            _prev = getattr(play, '_last_logged_ep', -1)
            if total_completed != _prev:
                print(f'  [CA eval] Episodes completed: {total_completed:>5d} / {TARGET_EPISODES}'
                      f'  |  Step: {step:>7d}')
                play._last_logged_ep = total_completed

        if occ_grid is not None:
            new_grid = infos.get("occupancy_grid", None)
            if new_grid is not None:
                occ_grid = new_grid.to(env.device)

        wm_obs["prop"] = obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim].to(world_model.device)
        wm_obs["is_first"] = wm_is_first

        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                wm_obs["image"] = infos["depth"].unsqueeze(-1).to(world_model.device)


        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        # update world model input
        reset_env_ids = reset_env_ids.cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_is_first[reset_env_ids] = 1

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:-env.height_dim]),
                                           dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                record_filename = os.path.join(frames_record_path, f"{img_idx}.png")

                os.makedirs(frames_record_path, exist_ok=True)

                env.gym.write_viewer_image_to_file(env.viewer, record_filename)
                img_idx += 1 
        
        if MOVE_CAMERA:
            lootat = env.root_states[env_cfg.viewer.lookat_id, :3]
            # camara_position = lootat.detach().cpu().numpy() + np.array([-1, -2, 0.5]) # Isometric
            # camara_position = lootat.detach().cpu().numpy() + np.array([0, -2, 0.5] )# Sideways
            # camara_position = lootat.detach().cpu().numpy() + np.array([1.5, 0, 0.5]) # Frontal
            camara_position = lootat.detach().cpu().numpy() + np.array([0.05, 0.05, 5.5]) # Top-down
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

            os.makedirs(data_record_path, exist_ok=True)  
            data_log_path = os.path.join(data_record_path, 'log.pkl')
            logger.save(data_log_path)

    print(f'\n[CA eval] Finished. {total_completed} episodes completed in {step} total steps.')
    print('total reward:', total_reward)

    # ── Post-loop: collision avoidance plots and summary ──────────────────────
    # Flush any incomplete final episodes from all envs
    for _e in range(env.num_envs):
        ep = _ep_data[_e]
        if ep['sv']:
            episode_aggregates.append({
                'mean_sv':          float(np.mean(ep['sv'])),
                'mean_abs_yaw':     float(np.mean(ep['yr'])),
                'peak_abs_yaw':     float(np.max(ep['yr'])) if ep['yr'] else 0.0,
                'mean_abs_lat_vel': float(np.mean(ep['lv'])),
                'mean_fwd_vel':     float(np.mean(ep['fv'])),
                'mean_vte':         float(np.mean(ep['vte'])),
                'collision_count':  ep['col'],
                'episode_length':   len(ep['sv']),
                'episode_return':   ep['rew'],
            })

    all_env_flat = {}
    for _key in _all_env_stacks:
        all_env_flat[_key] = np.array(_all_env_stacks[_key]).ravel()

    ca_out_dir = os.path.join(
        data_record_path,
        f'eval_collision_{args.terrain}_{POLICY_NAME}'
    )
    os.makedirs(ca_out_dir, exist_ok=True)

    if len(ca_log['safety_value']) > 0:
        _save_collision_plots(
            out_dir=ca_out_dir,
            policy_name=POLICY_NAME,
            terrain=args.terrain,
            dt=env.dt,
            ca_log=ca_log,
            all_env_flat=all_env_flat,
            all_env_stacks=_all_env_stacks,
            episode_aggregates=episode_aggregates,
            episode_reset_steps=episode_reset_steps,
        )
        _write_collision_summary(
            out_dir=ca_out_dir,
            policy_name=POLICY_NAME,
            terrain=args.terrain,
            dt=env.dt,
            all_env_flat=all_env_flat,
            episode_aggregates=episode_aggregates,
            num_envs=env.num_envs,
        )
    else:
        print('[CA eval] No data collected — skipping plots.')

if __name__ == '__main__':
    EXPORT_POLICY = True
    # RECORD_FRAMES = False
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    # Set POLICY_NAME to identify this run in outputs.
    # Options: "baseline", "ppo_lagrangian", "null_space_safety" (or any string)
    POLICY_NAME = 'null_space_safety'
    args = get_args()
    args.rl_device = args.sim_device
    play(args)