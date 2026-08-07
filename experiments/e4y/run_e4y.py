"""E4-Y -- command-space filtering **with steering authority**.

The completed command-space study filtered forward speed only.  With the
barrier written on the base centre, yaw does not appear in h_dot, so the
filter reduced to a brake: its only answer to "too close" was "stop", it had
no legal option on 79-100% of the steps it engaged, and the robot parked
rather than steered.  That is not the mechanism OCR/ABS-style filters use --
they filter a twist (v, omega) on a unicycle -- so until yaw is in the
filtered set, that row is not fairly tested.

E4-Y puts the barrier on a **look-ahead point** a fixed distance ahead of the
base, which is the standard remedy: both v and omega then enter the CBF
condition at first order, and the filter can genuinely steer.  The projection
is weighted so that yaw is the *cheaper* correction -- when both steering and
braking would satisfy the constraint, the filter prefers to steer.  That is a
deliberate choice and the whole point of the variant.

Run `calibrate_commands.py` first.  It measures what the command interface can
actually deliver and decides how this study should be framed:

  * usable yaw tracking -> a fair test of the steering mechanism;
  * degenerate yaw tracking -> a fair test of retrofitting onto *this* policy
    and a weak test of the mechanism, which must be stated rather than left
    for a reader to discover.

The yaw-disabled arm runs the same code with `omega_range = (0, 0)`, so the
control is the same filter with one axis removed rather than a different
implementation.

Usage:
    python experiments/e4y/run_e4y.py --load_run pi_nom --alpha 2.0 \
        --task go1_amp --headless
    python experiments/e4y/run_e4y.py --load_run pi_nom --alpha 2.0 \
        --no_yaw --task go1_amp --headless
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
    p.add_argument("--filter", type=str, default="unicycle",
                   choices=["unicycle", "none"])
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--yaw", dest="yaw", action="store_true", default=True,
                   help="allow the filter to steer (the point of this study)")
    p.add_argument("--no_yaw", dest="yaw", action="store_false",
                   help="control arm: same filter, yaw axis removed, which "
                        "reduces it to the braking-only filter")
    p.add_argument("--omega_limit", type=float, default=None,
                   help="max |yaw rate| the filter may command. Defaults to "
                        "the trained range, which is what the policy can "
                        "actually track.")
    p.add_argument("--lookahead", type=float, default=0.35,
                   help="distance ahead of the base at which the barrier is "
                        "written. Larger gives yaw more leverage but guards a "
                        "point further from the body that is actually hit.")
    p.add_argument("--w_v", type=float, default=4.0,
                   help="cost of changing forward speed")
    p.add_argument("--w_omega", type=float, default=1.0,
                   help="cost of changing yaw rate; below w_v so the filter "
                        "prefers steering to braking")
    p.add_argument("--barrier", type=str, default="geometric",
                   choices=["geometric", "sv"])
    p.add_argument("--body_extent", type=float, default=None)
    p.add_argument("--max_passes", type=int, default=12)
    p.add_argument("--policy_tag", type=str, default="pi_nom")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--eval_envs", type=int, default=250)
    p.add_argument("--eval_terrain", type=str, default="corridor")
    p.add_argument("--eval_seed_file", type=str, default=None)
    p.add_argument("--dr", type=str, default="off")
    p.add_argument("--out_dir", type=str, default="logs/e4y/sweep")
    p.add_argument("--run_id", type=str, default=None)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402
from safeloco_eval.filters import build_filter  # noqa: E402
from safeloco_eval.command_filters import (  # noqa: E402
    build_command_filter, BODY_COLLISION_TOL)


def default_run_id():
    """Run name carries alpha, look-ahead and the arm.

    The look-ahead is in the name because it is not a tuning knob here, it is
    the geometry that decides whether yaw exists as a control input at all:
    L = 0 puts the barrier back on the base centre, where the yaw term
    vanishes identically and the filter *is* the braking-only filter of the
    completed study.  Two runs that differ in L are different experiments and
    must not share a filename.
    """
    if ARGS.filter == "none":
        return "{}_unfiltered".format(ARGS.policy_tag)
    return "{}_uni_a{:g}_L{:g}_{}".format(
        ARGS.policy_tag, ARGS.alpha, ARGS.lookahead,
        "yaw" if ARGS.yaw else "noyaw")


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()
    os.makedirs(ARGS.out_dir, exist_ok=True)

    # Trained ranges, read before the eval overrides collapse them.
    tcfg, _ = task_registry.get_cfgs(name=args.task)
    rng = tcfg.commands.ranges
    train_vx = tuple(getattr(rng, "lin_vel_x", [0.0, 0.8]))
    train_w = tuple(getattr(rng, "ang_vel_yaw", [0.0, 0.0]))

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.eval_terrain, ARGS.eval_envs,
                                dr_mode=ARGS.dr)

    if ARGS.yaw:
        lim = ARGS.omega_limit
        omega_range = (-abs(lim), abs(lim)) if lim is not None else train_w
    else:
        omega_range = (0.0, 0.0)

    body_margin = None
    if ARGS.body_extent is not None:
        body_margin = BODY_COLLISION_TOL + ARGS.body_extent

    cmd_filt = build_command_filter(
        ARGS.filter, alpha=ARGS.alpha, vx_range=train_vx,
        omega_range=omega_range, max_passes=ARGS.max_passes,
        barrier=ARGS.barrier, body_margin=body_margin,
        lookahead=ARGS.lookahead, w_v=ARGS.w_v, w_omega=ARGS.w_omega)

    if ARGS.filter != "none":
        span = abs(omega_range[1] - omega_range[0])
        print("[E4-Y] yaw {} | yaw range {} (trained {}) | lookahead {:.2f} m "
              "| weights v={:g} omega={:g}".format(
                  "ENABLED" if ARGS.yaw else "disabled (control arm)",
                  omega_range, train_w, ARGS.lookahead, ARGS.w_v, ARGS.w_omega))
        if ARGS.yaw and span < 1e-6:
            print("[E4-Y] WARNING: the trained yaw range is degenerate, so the "
                  "filter has no steering authority to use. Pass --omega_limit "
                  "to widen it, and note that commands outside the trained "
                  "range are not tracked -- see calibrate_commands.py.")

    act_filt = build_filter("none", qsafe=None, epsilon_rad=0.0,
                            action_scale=float(env_cfg.control.action_scale),
                            clip_actions=float(env_cfg.normalization.clip_actions),
                            device=env.device)

    variant = "{}_{}".format(cmd_filt.name, "yaw" if ARGS.yaw else "noyaw")
    if ARGS.filter != "none" and ARGS.lookahead == 0.0:
        # L = 0 collapses the yaw term to zero identically, so this row is the
        # completed braking-only study re-run under the corrected projection
        # rather than a second control arm.  Label it as such.
        variant += "_L0_braking_replica"
        if ARGS.yaw:
            print("[E4-Y] NOTE: --lookahead 0 removes the yaw term from the "
                  "constraint, so --yaw has no effect. This row is the "
                  "braking-only filter whatever the flag says.")
    meta = {"run_id": run_id, "method": "E4Y", "variant": variant,
            "epsilon_or_alpha": (ARGS.alpha if ARGS.filter != "none" else ""),
            "terrain": ARGS.eval_terrain}

    collector = EC.EvalCollector(env, runner, policy, act_filt, qsafe=None,
                                 run_meta=meta, cmd_filt=cmd_filt)
    seed_list, _ = S.load(ARGS.eval_seed_file)
    records = collector.run(ARGS.n_episodes, seed_list)

    csv_path = os.path.join(ARGS.out_dir, run_id + ".csv")
    EC.write_records(csv_path, records)
    EC.write_manifest(
        os.path.join(ARGS.out_dir, run_id + ".manifest.json"),
        run_id=run_id, method="E4Y", variant=variant,
        alpha=(ARGS.alpha if ARGS.filter != "none" else None),
        yaw_enabled=ARGS.yaw, omega_range=list(omega_range),
        trained_yaw_range=list(train_w), trained_vx_range=list(train_vx),
        lookahead=ARGS.lookahead, w_v=ARGS.w_v, w_omega=ARGS.w_omega,
        barrier=ARGS.barrier,
        barrier_required_clearance_at_r0p3=(
            cmd_filt.required_clearance(0.3)
            if hasattr(cmd_filt, "required_clearance") else None),
        policy_tag=ARGS.policy_tag,
        # The reward function the evaluated policy was trained under.  A
        # yaw-capable baseline adds a tracking_ang_vel term, so its `return`
        # is on a different scale from pi_nom's and from Table 2's -- the
        # analyser reads this and refuses to score the return axis across a
        # mismatch rather than comparing two different reward functions.
        eval_reward_scales={
            k: float(v) for k, v in
            vars(type(env_cfg.rewards.scales)).items()
            if not k.startswith("_") and isinstance(v, (int, float))},
        tracking_ang_vel=float(getattr(env_cfg.rewards.scales,
                                       "tracking_ang_vel", 0.0)),
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        terrain=ARGS.eval_terrain, dr_mode=ARGS.dr,
        collision_metric=("geometric_link_cylinder"
                          if collector.collision_geometry else "sv_proxy"),
        n_episodes=len(records), eval_envs=ARGS.eval_envs,
        terrain_seed=EC.TERRAIN_SEED, cmd_lin_vel_x=EC.EVAL_CMD_LIN_VEL_X,
        control_dt=float(env.dt),
    )

    from safeloco_eval import metrics as M
    from safeloco_eval import stats as ST
    ck, cn = M.collision_rate(records)
    ak, an = M.activation_rate(records)
    ik, inn = M.infeasibility_rate(records)
    steer, brake = M.steer_brake_split(records)
    print("[E4-Y] {}  collision={:.2f}%  fwd={:.3f} m/s  dist={:.2f} m  "
          "contact/m={:.1f}  activation={:.1f}%  infeasible={:.1f}%  "
          "steer share={:.0f}%".format(
              run_id, 100.0 * ck / max(cn, 1), M.mean_fwd_vel(records),
              M.distance_travelled(records), M.contact_per_metre(records),
              100.0 * ak / max(an, 1), 100.0 * ik / max(inn, 1),
              100.0 * (steer if steer == steer else 0.0)))
    print("  wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
