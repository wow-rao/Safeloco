"""Train the step-trap policies: the gradient-projection method on a terrain
whose only exit is over a low sill.

What gets trained
-----------------
Two run directories, both on a mixed roughflat + step-trap terrain and both
leaving `pi_nom` untouched (every completed study is measured against it, and
a policy trained on different terrain is a different robot):

  pi_trap_ours   --policy ours   safety_coef pinned ON (default 0.1).  The
                 paper's method: reachability critic + damped null-space
                 projection, fed by the volumetric step-over l-value that
                 step-trap cells publish (legged_gym/utils/box_sdf.py).  The
                 l-value keeps the space above the sill safe, so the safety
                 gradient steers the gait toward higher swings rather than
                 forbidding the crossing.

  pi_trap_nom    --policy nom    safety_coef pinned 0.  Task-only ablation:
                 same terrain exposure, no safety machinery.  Separates
                 "learned to cross because it saw the terrain" from "crosses
                 cleanly because the l-value shaped how".

Config traps this script handles for you (the same ones train_yaw_policy.py
documents at length):

* `safety_coef` min/max are pinned alongside the value, or the adaptive
  schedule reintroduces the coefficient mid-run.
* Terrain proportions put zero mass in slots 0-8, so `roughflat_start_idx`
  is 0: the `flat_*` command ranges govern every env and heading control
  covers none -- identical command interface to corridor training.
* The step-trap slot is index 12, so the proportions list must have 13
  entries; `_reward_cheat` then covers the trap envs and keeps training
  pointed at the gate instead of learning to circle the pen.

Usage:
    python experiments/etrap/train_trap_policy.py --policy ours \
        --task go1_amp --headless --max_iterations 2000 --seed 1
    python experiments/etrap/train_trap_policy.py --policy nom \
        --task go1_amp --headless --max_iterations 2000 --seed 1
"""

import argparse
import json
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy", type=str, default="ours",
                   choices=["ours", "nom"],
                   help="ours = gradient projection on (safety_coef 0.1); "
                        "nom = task-only ablation (safety_coef 0)")
    p.add_argument("--run_name", type=str, default=None,
                   help="defaults to pi_trap_<policy>")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_iterations", type=int, default=2000,
                   help="matches pi_nom/pi_rs so returns stay comparable")
    p.add_argument("--safety_coef", type=float, default=0.1,
                   help="used only with --policy ours; pinned min==max")
    p.add_argument("--trap_fraction", type=float, default=0.8,
                   help="fraction of envs on step-trap cells; the rest are "
                        "rough flat (gait quality does not collapse into "
                        "the pen-only distribution)")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402


def apply_trap_config(env_cfg, train_cfg):
    """Terrain proportions + safety coefficient; returns what changed."""
    frac = max(0.0, min(1.0, float(ARGS.trap_fraction)))
    props = [0.0] * 9 + [1.0 - frac, 0.0, 0.0, frac]   # slot 12 = step trap
    before = {
        "terrain_proportions": list(env_cfg.terrain.terrain_proportions),
        "safety_coef": float(getattr(train_cfg.algorithm, "safety_coef", 0.0)),
    }
    env_cfg.terrain.terrain_proportions = props

    coef = float(ARGS.safety_coef) if ARGS.policy == "ours" else 0.0
    train_cfg.algorithm.safety_coef = coef
    train_cfg.algorithm.safety_coef_min = coef
    train_cfg.algorithm.safety_coef_max = coef

    after = {"terrain_proportions": props, "safety_coef": coef}
    return {"before": before, "after": after}


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    run_name = ARGS.run_name or "pi_trap_{}".format(ARGS.policy)
    train_cfg.seed = ARGS.seed
    train_cfg.runner.run_name = run_name
    train_cfg.runner.save_interval = 500
    train_cfg.runner.max_iterations = ARGS.max_iterations

    delta = apply_trap_config(env_cfg, train_cfg)

    props = list(env_cfg.terrain.terrain_proportions)
    governed_by_flat_branch = (sum(props[:9]) == 0)
    trap_covered_by_cheat = len(props) >= 13 and props[12] > 0

    env, env_cfg = task_registry.make_env(name=args.task, args=args,
                                          env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                           train_cfg.runner.experiment_name,
                           train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)
    record = dict(
        run_name=train_cfg.runner.run_name,
        policy=ARGS.policy,
        seed=ARGS.seed,
        max_iterations=train_cfg.runner.max_iterations,
        num_envs=env_cfg.env.num_envs,
        trap_config=delta,
        terrain_proportions=props,
        yaw_governed_by_flat_branch=governed_by_flat_branch,
        trap_envs_covered_by_cheat_penalty=trap_covered_by_cheat,
        heading_command=bool(env_cfg.commands.heading_command),
        safety_coef=float(train_cfg.algorithm.safety_coef),
        lvalue=class_to_dict(env_cfg.lvalue) if hasattr(env_cfg, "lvalue")
        else None,
        step_trap=class_to_dict(env_cfg.terrain.step_trap)
        if hasattr(env_cfg.terrain, "step_trap") else None,
        reward_scales=class_to_dict(env_cfg.rewards.scales),
        purpose=("step-trap policy: exits a closed pen by stepping over a "
                 "low sill; the volumetric l-value keeps the space above the "
                 "sill safe so the safety gradient shapes the swing instead "
                 "of forbidding the crossing"),
        git_commit=EC.git_commit(),
    )
    with open(os.path.join(log_dir, "trap_config.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    print("=" * 70)
    print("training {} ({})  ->  {}".format(
        run_name, ARGS.policy, log_dir))
    print("  terrain_proportions = {}".format(props))
    print("  safety_coef         = {} (min==max, pinned)".format(
        train_cfg.algorithm.safety_coef))
    if float(train_cfg.algorithm.safety_coef) > 0:
        print("  gradient projection ON: trap envs feed the volumetric "
              "l-value (box_sdf) into the reachability critic")
    else:
        print("  task-only ablation: same terrain, no safety machinery")
    if not trap_covered_by_cheat:
        print("  WARNING: proportions carry no slot-12 mass; nothing about "
              "this run is the step trap. Check --trap_fraction.")
    print("=" * 70)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)

    print("\ndone. Evaluate with, e.g.:")
    print("  python experiments/etrap/run_etrap.py --task {} --headless \\\n"
          "      --load_run {} --policy_tag {} --controller none".format(
              args.task, run_name, run_name))


if __name__ == "__main__":
    main()
