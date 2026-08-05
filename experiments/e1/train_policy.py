"""Train the base policies E1 needs, from scratch.

experiment_protocol_phased.md §1: "pi_nom (task-only PPO -- exists, Table 14;
the substrate for all deployment-time filters)".  With no surviving runs, it
has to be retrained before anything else in Phase 1 can start.

Only `--policy` and `--rs_weight` are new flags; everything else
(`--task`, `--headless`, `--run_name`, `--max_iterations`, `--num_envs`,
`--seed`, `--sim_device`, `--wm_device`) is `legged_gym`'s usual `get_args`.

What each policy is, in terms of the shipped config
---------------------------------------------------
pi_nom   task-only PPO.  `algorithm.safety_coef = 0`, which makes
         `AMPPPO.update` skip the entire safety branch
         (`amp_ppo.py:350`) -- no safety surrogate, no safety critic
         update, no damped null-space projection.  The reward is
         untouched: `safety_value` is not in `rewards.scales`, so
         `_reward_safety_value` is never registered.  **This is the only
         policy E1 strictly requires.**

pi_rs    reward shaping.  Same as pi_nom plus
         `rewards.scales.safety_value = w`, which adds `w * sv` to the
         reward, where `sv = 0.3*h_dot + h` clamped at 0.5 and negative on
         approach.  Optional for E1 (Q-hat substrate breadth and one
         optional sweep point); required for E2, where it is the
         Reward-only row.

pi_ours  the paper's method: `safety_coef > 0` turns on the reachability
         critic and the damped null-space projection.  **Not needed for
         E1** -- the verdict rule quotes the paper's own numbers for it.
         Provided so the same script covers the later Pareto plots.

Both the world model and the policy are trained by this one run
(`WMPRunner.learn` calls `train_world_model` once enough episodes have
accumulated) and both are written into the same checkpoint, so there is no
separate world-model training step.

Usage
-----
    python experiments/e1/train_policy.py --policy pi_nom --task go1_amp \
        --headless --max_iterations 4000
"""

import argparse
import inspect
import json
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy", type=str, default="pi_nom",
                   choices=["pi_nom", "pi_rs", "pi_ours"])
    p.add_argument("--rs_weight", type=float, default=1.0,
                   help="pi_rs only: weight on the safety_value reward term. "
                        "Document whatever you use -- returns are not "
                        "comparable across differing reward configs.")
    p.add_argument("--ours_safety_coef", type=float, default=0.1,
                   help="pi_ours only: algorithm.safety_coef")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

import torch  # noqa: E402,F401


def apply_policy_config(env_cfg, train_cfg, policy):
    """Returns the deltas applied, for the record written to the log dir."""
    delta = {"policy": policy}

    if policy in ("pi_nom", "pi_rs"):
        # Disable the safety branch entirely.  min == max == 0 also pins the
        # adaptive coefficient, so nothing can reintroduce it mid-run.
        train_cfg.algorithm.safety_coef = 0.0
        train_cfg.algorithm.safety_coef_min = 0.0
        train_cfg.algorithm.safety_coef_max = 0.0
        delta["safety_coef"] = 0.0
    else:
        train_cfg.algorithm.safety_coef = ARGS.ours_safety_coef
        train_cfg.algorithm.safety_coef_min = ARGS.ours_safety_coef
        train_cfg.algorithm.safety_coef_max = ARGS.ours_safety_coef
        delta["safety_coef"] = ARGS.ours_safety_coef

    if policy == "pi_rs":
        env_cfg.rewards.scales.safety_value = ARGS.rs_weight
        delta["rewards.scales.safety_value"] = ARGS.rs_weight
    else:
        # `safety_value` is absent from the shipped scales; make sure a stale
        # edit to the config file cannot leak a shaping term into pi_nom.
        if hasattr(env_cfg.rewards.scales, "safety_value"):
            env_cfg.rewards.scales.safety_value = 0.0
            delta["rewards.scales.safety_value"] = 0.0

    return delta


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # train.py's defaults, then the policy delta, then anything from get_args
    # (--run_name / --max_iterations override these inside make_wmp_runner).
    train_cfg.runner.run_name = ARGS.policy
    train_cfg.runner.save_interval = 500
    train_cfg.runner.max_iterations = 2000

    delta = apply_policy_config(env_cfg, train_cfg, ARGS.policy)

    env, env_cfg = task_registry.make_env(name=args.task, args=args,
                                          env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                           train_cfg.runner.experiment_name,
                           train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)
    record = dict(delta,
                  run_name=train_cfg.runner.run_name,
                  max_iterations=train_cfg.runner.max_iterations,
                  num_envs=env_cfg.env.num_envs,
                  seed=train_cfg.seed,
                  terrain_proportions=list(env_cfg.terrain.terrain_proportions),
                  use_amp=train_cfg.runner.use_amp,
                  amp_reward_coef=train_cfg.runner.amp_reward_coef,
                  reward_scales=class_to_dict(env_cfg.rewards.scales))
    with open(os.path.join(log_dir, "policy_config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)

    print("=" * 70)
    print("training {}  ->  {}".format(ARGS.policy, log_dir))
    for k, v in delta.items():
        print("  {} = {}".format(k, v))
    print("  iterations = {}, envs = {}".format(
        train_cfg.runner.max_iterations, env_cfg.env.num_envs))
    if train_cfg.runner.use_amp:
        print("  NOTE: AMP is on (amp_reward_coef = {}). Returns are only "
              "comparable across policies trained with the same reward "
              "config -- state this in App. F.".format(
                  train_cfg.runner.amp_reward_coef))
    print("=" * 70)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
