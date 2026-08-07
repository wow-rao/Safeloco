"""Train a locomotion policy that can actually follow a yaw command.

Why this exists
---------------
The command-filtering study reached its sharpest claim -- *no configuration is
simultaneously sound, full-speed, and better than the reference* -- on a
policy that physically cannot turn.  Measured yaw tracking gain: 0.015.  The
filter's fast configurations command a yaw rate the robot ignores, so the
barrier certifies a twist that is never executed and its guarantee is void.

That is a real finding about *this* policy.  It is not a finding about
command filtering, and a reviewer would say so in one sentence.  A fragility
claim tested only on a platform that cannot exercise the mechanism is not
evidence.  So: train a policy that can turn, and re-run the study against it.

Note the framing this serves.  It is expected and unsurprising that a
command-space filter does well at collision avoidance -- it operates on a far
higher-level interface than a joint-space method.  The claim being defended is
not "we avoid collisions better", it is about soundness and about what a
command filter can express at all.  That claim needs the strongest available
opponent, which is why this makes the baseline stronger on purpose.

The three config fields that matter, none of them the obvious one
-----------------------------------------------------------------
**1. `flat_ang_vel_yaw`, not `ang_vel_yaw`.**  Training runs on
`terrain_proportions = [0]*9 + [0.2, 0.8]`, so `sum(proportions[:9]) == 0` and
therefore `roughflat_start_idx == 0`.  Trace `_resample_commands` with that:

  * `heading_command = True`, so the `else` branch that would sample
    `ang_vel_yaw` never runs.  That field is dead config during training.
  * `flat_env_ids = env_ids[env_ids >= 0]` selects **every** env, so all of
    them are re-sampled from `flat_ang_vel_yaw`.
  * heading control in `_post_physics_step_callback` covers
    `[:roughflat_start_idx]` = nobody, so nothing overwrites the result.

So `flat_ang_vel_yaw = [-0.01, 0.01]` is the range that governed, and it
explains the calibration exactly: the policy saturates at ~0.02 rad/s because
that is precisely what it saw.  Widening `ang_vel_yaw` alone changes nothing.
Both are set here, the second for consistency and to stop the next reader
hitting the same trap.

**2. `tracking_ang_vel` was 0.0.**  The yaw-tracking reward is switched off in
the shipped config, so even with a wide command range there is no gradient
pushing the policy to follow it.  Widening the range without this would
produce a policy that ignores yaw exactly as before, having burned the GPU
time to find out.

**3. `safety_coef` defaults to 0.1, i.e. ON.**  This one is not about yaw at
all, and it is the easiest to miss because nothing in this script's purpose
points at it.  `train_policy.py` zeroes it for pi_nom and pi_rs; a script that
simply does not mention it inherits 0.1 and trains the yaw-capable analogue of
**pi_ours**.  Layering a command filter on a policy that already carries the
paper's safety mechanism measures nothing about command filtering.  `min` and
`max` are pinned alongside it, or the adaptive schedule brings it back
mid-run.

This writes to its own run directory and does not touch `pi_nom`.  Every
completed study shares that baseline; a policy with different command ranges
is a different robot, and silently replacing it would invalidate rows that
are already analysed.

Usage:
    python experiments/e4y/train_yaw_policy.py --task go1_amp --headless \
        --max_iterations 2000 --seed 1

Then -- before spending anything on a sweep -- check it worked:
    python experiments/e4y/calibrate_commands.py --task go1_amp --headless \
        --load_run pi_nom_yaw --out logs/e4y/calibration_yaw.json

The gate is the forward tracking gain (must be ~1, i.e. it still walks) and
the yaw tracking gain (must be well above 0.015, or the retrain failed).
"""

import argparse
import json
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402

#: The range that actually governs yaw during training -- see the module
#: docstring.  Both fields are set to this; only the `flat_` one has effect.
YAW_RANGE = [-1.0, 1.0]

#: Shipped config has this at 0.0, i.e. no reward for tracking yaw at all.
#: 0.5 against `tracking_lin_vel = 1.5` is the usual legged_gym ratio.
TRACKING_ANG_VEL = 0.5


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_name", type=str, default="pi_nom_yaw",
                   help="own run directory; pi_nom is left untouched because "
                        "every completed study is measured against it")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_iterations", type=int, default=2000,
                   help="matches pi_nom/pi_rs so returns stay comparable")
    p.add_argument("--yaw_range", type=float, nargs=2, default=YAW_RANGE)
    p.add_argument("--tracking_ang_vel", type=float, default=TRACKING_ANG_VEL)
    p.add_argument("--safety_coef", type=float, default=0.0,
                   help="0 = task-only, matching pi_nom. The shipped config "
                        "defaults this to 0.1, so leaving it alone trains the "
                        "analogue of pi_ours instead and the comparison is "
                        "meaningless. Set non-zero only on purpose.")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402


def apply_yaw_config(env_cfg, train_cfg):
    """Widen the yaw command range, switch its reward on, and zero safety_coef.

    Returns what changed, so the run directory records it and a later reader
    does not have to diff configs to find out what this policy is.

    **safety_coef must be zeroed explicitly.**  The shipped config defaults it
    to 0.1, and `train_policy.py` zeroes it for pi_nom/pi_rs precisely because
    the default is *on*.  Inheriting the default here would train the
    yaw-capable analogue of pi_ours -- a policy that already carries the
    paper's safety mechanism -- and layering a command filter on that measures
    nothing about command filtering.  min and max are pinned too, or the
    adaptive schedule reintroduces the coefficient mid-run.
    """
    rng = env_cfg.commands.ranges
    before = {
        "ang_vel_yaw": list(getattr(rng, "ang_vel_yaw", [])),
        "flat_ang_vel_yaw": list(getattr(rng, "flat_ang_vel_yaw", [])),
        "tracking_ang_vel": float(env_cfg.rewards.scales.tracking_ang_vel),
        "safety_coef": float(getattr(train_cfg.algorithm, "safety_coef", 0.0)),
    }
    lo, hi = float(ARGS.yaw_range[0]), float(ARGS.yaw_range[1])
    rng.ang_vel_yaw = [lo, hi]           # dead config here, set for clarity
    rng.flat_ang_vel_yaw = [lo, hi]      # the one that actually governs
    env_cfg.rewards.scales.tracking_ang_vel = float(ARGS.tracking_ang_vel)

    train_cfg.algorithm.safety_coef = ARGS.safety_coef
    train_cfg.algorithm.safety_coef_min = ARGS.safety_coef
    train_cfg.algorithm.safety_coef_max = ARGS.safety_coef

    after = {
        "ang_vel_yaw": [lo, hi],
        "flat_ang_vel_yaw": [lo, hi],
        "tracking_ang_vel": float(ARGS.tracking_ang_vel),
        "safety_coef": float(ARGS.safety_coef),
    }
    return {"before": before, "after": after}


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg.seed = ARGS.seed
    train_cfg.runner.run_name = ARGS.run_name
    train_cfg.runner.save_interval = 500
    train_cfg.runner.max_iterations = ARGS.max_iterations

    delta = apply_yaw_config(env_cfg, train_cfg)

    # Which envs the flat resample branch covers decides whether any of the
    # above has an effect at all.  Record it rather than trusting the trace.
    props = list(env_cfg.terrain.terrain_proportions)
    governed_by_flat_branch = (sum(props[:9]) == 0)

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
        seed=ARGS.seed,
        max_iterations=train_cfg.runner.max_iterations,
        num_envs=env_cfg.env.num_envs,
        yaw_config=delta,
        terrain_proportions=props,
        yaw_governed_by_flat_branch=governed_by_flat_branch,
        heading_command=bool(env_cfg.commands.heading_command),
        safety_coef=float(train_cfg.algorithm.safety_coef),
        reward_scales=class_to_dict(env_cfg.rewards.scales),
        purpose=("yaw-capable baseline for the command-filtering study; "
                 "pi_nom cannot turn (measured tracking gain 0.015) so the "
                 "soundness claim could not be tested on it"),
        git_commit=EC.git_commit(),
    )
    with open(os.path.join(log_dir, "yaw_config.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    print("=" * 70)
    print("training a yaw-capable policy  ->  {}".format(log_dir))
    for k in ("ang_vel_yaw", "flat_ang_vel_yaw", "tracking_ang_vel",
              "safety_coef"):
        print("  {:<18} {} -> {}".format(k, delta["before"][k],
                                         delta["after"][k]))
    if governed_by_flat_branch:
        print("  yaw is resampled from flat_ang_vel_yaw for ALL envs "
              "(roughflat_start_idx = 0), which is the field that matters")
    else:
        print("  WARNING: sum(terrain_proportions[:9]) != 0, so some envs are "
              "governed by ang_vel_yaw and some by flat_ang_vel_yaw, and "
              "heading control overwrites yaw for the first block. Both "
              "ranges are set, but check _resample_commands before trusting "
              "this run.")
    if float(train_cfg.algorithm.safety_coef) != 0.0:
        print("  *** safety_coef is {}, NOT 0. This is the pi_ours-style "
              "policy, not a task-only baseline. Intended? ***"
              .format(train_cfg.algorithm.safety_coef))
    else:
        print("  safety_coef 0.0 -- task-only, matching pi_nom "
              "(config default is 0.1 and must be overridden)")
    print("=" * 70)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)

    print("\ndone. Before sweeping anything, verify the retrain worked:")
    print("  python experiments/e4y/calibrate_commands.py --task {} "
          "--headless \\\n      --load_run {} --out logs/e4y/calibration_yaw.json"
          .format(args.task, ARGS.run_name))
    print("  forward tracking gain ~1.0  -> it still walks")
    print("  yaw tracking gain >> 0.015  -> it can now turn")


if __name__ == "__main__":
    main()
