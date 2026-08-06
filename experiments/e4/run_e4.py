"""E4 -- command-space filtering, the steel-man baseline.

analysis_protocol.md §3.4.  Evaluation-only (§2): one policy, N = 1,000
episodes on the fixed seed list, Wilson CIs.  Sweeps the CBF class-K gain
alpha; `--alpha inf` (or `--filter none`) is the unfiltered row.

This is the experiment that should *work*.  Command-space filtering is the
form these methods actually take, and clearance has relative degree one with
respect to a base-velocity command, so the constraint is a linear inequality
with a closed-form projection and no learned critic anywhere.  E4 exists to
give that its best shot; §3.4's gate then reads the result on the joint
frontier rather than on collision rate alone.

Usage:
    python experiments/e4/run_e4.py --load_run pi_nom --filter none \
        --task go1_amp --headless
    python experiments/e4/run_e4.py --load_run pi_nom --alpha 1.0 \
        --task go1_amp --headless
"""

import argparse
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--filter", type=str, default="cbf",
                   choices=["cbf", "none"],
                   help="cbf = first-order CBF on the commanded planar "
                        "velocity; none = the unfiltered reference row.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="class-K gain in h_dot >= -alpha*h. SMALL alpha is "
                        "MORE conservative (permits only slow approach); "
                        "alpha -> inf recovers the unfiltered policy.")
    p.add_argument("--max_passes", type=int, default=12,
                   help="cyclic-projection passes for the multi-obstacle case")
    p.add_argument("--clamp_commands", action="store_true", default=True,
                   help="clamp the filtered command to the ranges the policy "
                        "was trained on; off-distribution commands are "
                        "tracked badly and would flatter the filter.")
    p.add_argument("--no_clamp_commands", dest="clamp_commands",
                   action="store_false")
    p.add_argument("--policy_tag", type=str, default="pi_nom")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--eval_envs", type=int, default=250)
    p.add_argument("--eval_terrain", type=str, default="corridor")
    p.add_argument("--eval_seed_file", type=str, default=None)
    p.add_argument("--dr", type=str, default="off")
    p.add_argument("--out_dir", type=str, default="logs/e4/sweep")
    p.add_argument("--run_id", type=str, default=None)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402
from safeloco_eval.filters import build_filter  # noqa: E402
from safeloco_eval.command_filters import build_command_filter  # noqa: E402


def default_run_id():
    if ARGS.filter == "none":
        return "{}_unfiltered".format(ARGS.policy_tag)
    return "{}_cbf_alpha{:g}".format(ARGS.policy_tag, ARGS.alpha)


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()
    os.makedirs(ARGS.out_dir, exist_ok=True)

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.eval_terrain, ARGS.eval_envs,
                                dr_mode=ARGS.dr)

    rng = env_cfg.commands.ranges
    vx_range = tuple(getattr(rng, "lin_vel_x", [-1.0, 1.0])) \
        if ARGS.clamp_commands else None
    vy_range = tuple(getattr(rng, "lin_vel_y", [-1.0, 1.0])) \
        if ARGS.clamp_commands else None
    # The eval config pins lin_vel_y to [0, 0], which would forbid the
    # sidestepping that is the whole reason command space can work.  Widen the
    # lateral clamp to the training range instead of the eval one.
    if ARGS.clamp_commands and vy_range is not None and vy_range[0] == vy_range[1]:
        vy_range = (-0.6, 0.6)

    cmd_filt = build_command_filter(
        ARGS.filter, alpha=ARGS.alpha, vx_range=vx_range, vy_range=vy_range,
        max_passes=ARGS.max_passes)

    # E4 never touches the action; the action filter is the identity so the
    # unfiltered action path is bit-identical to E1's reference row.
    act_filt = build_filter("none", qsafe=None, epsilon_rad=0.0,
                            action_scale=float(env_cfg.control.action_scale),
                            clip_actions=float(env_cfg.normalization.clip_actions),
                            device=env.device)

    variant = cmd_filt.name
    meta = {"run_id": run_id, "method": "E4", "variant": variant,
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
        run_id=run_id, method="E4", variant=variant,
        alpha=(ARGS.alpha if ARGS.filter != "none" else None),
        policy_tag=ARGS.policy_tag,
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        max_passes=ARGS.max_passes,
        clamp_commands=ARGS.clamp_commands,
        vx_range=list(vx_range) if vx_range else None,
        vy_range=list(vy_range) if vy_range else None,
        barrier="h = dist - 2*r_obs - 0.35 (legged_robot.py), relative degree 1 "
                "wrt commanded base velocity",
        yaw_filtered=False,
        terrain=ARGS.eval_terrain, dr_mode=ARGS.dr,
        collision_metric=("geometric_link_cylinder"
                          if collector.collision_geometry else "sv_proxy"),
        n_episodes=len(records), eval_envs=ARGS.eval_envs,
        terrain_seed=EC.TERRAIN_SEED, cmd_lin_vel_x=EC.EVAL_CMD_LIN_VEL_X,
        action_scale=float(env_cfg.control.action_scale),
        clip_actions=float(env_cfg.normalization.clip_actions),
        control_dt=float(env.dt),
    )

    from safeloco_eval import metrics as M
    from safeloco_eval import stats as ST
    ck, cn = M.collision_rate(records)
    fk, fn = M.fall_rate(records)
    ak, an = M.activation_rate(records)
    ik, _in = M.infeasibility_rate(records)
    print("[E4] {}  episodes={}  collision={:.2f}%  fall={:.2f}%  "
          "return={:.2f}  vel_err={:.3f}  lat_vel={:.3f}  activation={:.2f}%  "
          "infeasible={:.2f}%".format(
              run_id, len(records),
              100.0 * ck / max(cn, 1), 100.0 * fk / max(fn, 1),
              ST.mean(M.per_episode(records, "return")),
              ST.mean(M.per_episode(records, "vel_err")),
              ST.mean(M.per_episode(records, "mean_lat_vel")),
              100.0 * ak / max(an, 1), 100.0 * ik / max(_in, 1)))
    print("  wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
