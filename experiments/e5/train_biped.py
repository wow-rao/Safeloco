"""Train the E5 biped fall-safety policies, from scratch.

docs/bipedal_dynamic_safety.md: three arms over the same task (Cassie velocity
tracking on 70% flat / 30% stairs with training-time pushes), differing only
in the *mechanism* that carries the fall-safety signal:

pi_nom   task-only PPO.  `algorithm.safety_coef = 0` skips the safety
         surrogate and the damped null-space projection entirely, and
         `safety_value` is absent from `rewards.scales` so no shaping term is
         registered.  The reachability critic is still *fitted* on the
         rollouts (`train_safety_critic_when_off = True`; a parameter-disjoint
         head, zero policy influence) so fall-prediction calibration can be
         evaluated on this arm too.  This is the "prove the need" policy: it
         keeps the standard termination penalty every locomotion pipeline has,
         and still falls -- that gap is what E5 is about.

pi_rs    reward shaping.  pi_nom plus `rewards.scales.safety_value = w`, which
         adds w * sv to the reward, where sv is the fall margin from
         `Cassie._compute_safety_value` (clamped at 0.5, negative when
         tilting/low/fallen).

pi_ours  the paper's method: `safety_coef > 0` turns on the reachability
         critic surrogate and the damped null-space projection, on the same
         fall margin.

Deltas shared by all arms (recorded to policy_config.json): 2048 envs, terrain
proportions [0,0,.15,.15,0,0,0,0,0,.7], training pushes raised to 1.5 m/s,
forward command capped at 1.2 m/s.

Usage
-----
    python experiments/e5/train_biped.py --policy pi_nom --task cassie \
        --headless --max_iterations 10000

    # margin ablation (docs Sec. "capturability-informed variant"):
    python experiments/e5/train_biped.py --policy pi_ours --task cassie \
        --headless --margin_mode fail_set_dcm

    # nominal-height probe (no training): settle under zero actions and
    # print h_rel / margin percentiles to verify fall_safety.h_nom
    python experiments/e5/train_biped.py --policy pi_nom --task cassie \
        --headless --num_envs 64 --probe_steps 60
"""

import argparse
import inspect
import json
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402

E5_TERRAIN_PROPORTIONS = [0.0, 0.0, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]
E5_MAX_PUSH_VEL = 1.5
E5_CMD_VX_MAX = 1.2
E5_NUM_ENVS = 2048


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy", type=str, default="pi_nom",
                   choices=["pi_nom", "pi_rs", "pi_ours"])
    p.add_argument("--margin_mode", type=str, default="fail_set",
                   choices=["fail_set", "fail_set_dcm"],
                   help="fall_safety.mode; fail_set_dcm adds the "
                        "capture-point term (ablation B)")
    p.add_argument("--rs_weight", type=float, default=1.0,
                   help="pi_rs only: weight on the safety_value reward term")
    p.add_argument("--ours_safety_coef", type=float, default=0.1,
                   help="pi_ours only: algorithm.safety_coef (go1 ships 0.1)")
    p.add_argument("--d_safe", type=float, default=None,
                   help="override algorithm.d_safe (config: -0.05)")
    p.add_argument("--d_danger", type=float, default=None,
                   help="override algorithm.d_danger (config: -0.35)")
    p.add_argument("--alpha", type=float, default=None,
                   help="override algorithm.safety_return_alpha (config: 0.9)")
    p.add_argument("--probe_steps", type=int, default=0,
                   help="> 0: build the env, hold the default pose this many "
                        "steps under zero actions, print h_rel / margin "
                        "percentiles for the h_nom check, and exit")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

import torch  # noqa: E402


def apply_shared_config(env_cfg, train_cfg):
    """Deltas identical across arms; returned for the record."""
    env_cfg.env.num_envs = E5_NUM_ENVS
    env_cfg.terrain.terrain_proportions = list(E5_TERRAIN_PROPORTIONS)
    env_cfg.domain_rand.max_push_vel_xy = E5_MAX_PUSH_VEL
    env_cfg.commands.ranges.lin_vel_x = [0.0, E5_CMD_VX_MAX]
    env_cfg.commands.ranges.flat_lin_vel_x = [0.0, E5_CMD_VX_MAX]
    env_cfg.fall_safety.mode = ARGS.margin_mode
    if ARGS.d_safe is not None:
        train_cfg.algorithm.d_safe = ARGS.d_safe
    if ARGS.d_danger is not None:
        train_cfg.algorithm.d_danger = ARGS.d_danger
    if ARGS.alpha is not None:
        train_cfg.algorithm.safety_return_alpha = ARGS.alpha
    return {
        "num_envs": E5_NUM_ENVS,
        "terrain_proportions": list(E5_TERRAIN_PROPORTIONS),
        "max_push_vel_xy": E5_MAX_PUSH_VEL,
        "lin_vel_x": [0.0, E5_CMD_VX_MAX],
        "fall_safety.mode": ARGS.margin_mode,
        "d_safe": train_cfg.algorithm.d_safe,
        "d_danger": train_cfg.algorithm.d_danger,
        "safety_return_alpha": train_cfg.algorithm.safety_return_alpha,
    }


def apply_policy_config(env_cfg, train_cfg, policy):
    """Per-arm deltas, following experiments/e1/train_policy.py exactly.

    Never touch safety_coef_min/max here: plain PPO (cassie's algorithm
    class) does not accept those kwargs -- they are AMPPPO-only.
    """
    delta = {"policy": policy}

    if policy in ("pi_nom", "pi_rs"):
        train_cfg.algorithm.safety_coef = 0.0
        # Fit the reachability critic diagnostically (parameter-disjoint
        # head, no effect on the policy update) so calibration runs on
        # every arm, not just pi_ours.
        train_cfg.algorithm.train_safety_critic_when_off = True
        delta["safety_coef"] = 0.0
        delta["train_safety_critic_when_off"] = True
    else:
        train_cfg.algorithm.safety_coef = ARGS.ours_safety_coef
        train_cfg.algorithm.train_safety_critic_when_off = False
        delta["safety_coef"] = ARGS.ours_safety_coef

    if policy == "pi_rs":
        env_cfg.rewards.scales.safety_value = ARGS.rs_weight
        delta["rewards.scales.safety_value"] = ARGS.rs_weight
    else:
        # Make sure a stale config edit cannot leak a shaping term.
        if hasattr(env_cfg.rewards.scales, "safety_value"):
            env_cfg.rewards.scales.safety_value = 0.0
            delta["rewards.scales.safety_value"] = 0.0

    return delta


def probe(env, n_steps):
    """Settle under zero actions and report the standing geometry.

    The h_rel median after settling is the number fall_safety.h_nom should
    match (within ~0.1 m); the margins should be comfortably positive with
    safety_values pinned at the 0.5 clamp.
    """
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(n_steps):
        env.step(zeros)
    ground = env.measured_heights.mean(dim=1) \
        if torch.is_tensor(env.measured_heights) else env.env_origins[:, 2]
    h_rel = env.root_states[:, 2] - ground
    def pct(t, q):
        return torch.quantile(t.float(), q).item()
    print("=" * 70)
    print("E5 probe after {} zero-action steps ({} envs)".format(
        n_steps, env.num_envs))
    for name, t in (("h_rel [m]", h_rel),
                    ("safety_value (sv)", env.safety_values),
                    ("fall_margin_min", env.fall_margin_min),
                    ("proj_grav_z", env.projected_gravity[:, 2])):
        print("  {:<20} p5={:+.3f}  p50={:+.3f}  p95={:+.3f}".format(
            name, pct(t, 0.05), pct(t, 0.50), pct(t, 0.95)))
    print("  fall_safety.h_nom  = {:.3f} (config) -- set it to ~p50(h_rel) "
          "if they differ by > 0.1".format(env.cfg.fall_safety.h_nom))
    print("=" * 70)


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    assert train_cfg.runner.algorithm_class_name == 'PPO', \
        "E5 assumes plain PPO (cassie); got {}".format(
            train_cfg.runner.algorithm_class_name)

    run_name = ARGS.policy
    if ARGS.margin_mode == "fail_set_dcm":
        run_name += "_dcm"
    train_cfg.runner.run_name = run_name
    train_cfg.runner.save_interval = 500

    shared = apply_shared_config(env_cfg, train_cfg)
    delta = apply_policy_config(env_cfg, train_cfg, ARGS.policy)

    env, env_cfg = task_registry.make_env(name=args.task, args=args,
                                          env_cfg=env_cfg)

    if ARGS.probe_steps > 0:
        probe(env, ARGS.probe_steps)
        return

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                           train_cfg.runner.experiment_name,
                           train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)
    record = dict(shared, **delta)
    record.update(run_name=train_cfg.runner.run_name,
                  max_iterations=train_cfg.runner.max_iterations,
                  num_envs=env_cfg.env.num_envs,
                  seed=train_cfg.seed,
                  use_amp=train_cfg.runner.use_amp,
                  reward_scales=class_to_dict(env_cfg.rewards.scales))
    with open(os.path.join(log_dir, "policy_config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)

    print("=" * 70)
    print("training {}  ->  {}".format(run_name, log_dir))
    for k, v in sorted(dict(shared, **delta).items()):
        if k != "reward_scales":
            print("  {} = {}".format(k, v))
    print("  iterations = {}, envs = {}".format(
        train_cfg.runner.max_iterations, env_cfg.env.num_envs))
    print("=" * 70)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
