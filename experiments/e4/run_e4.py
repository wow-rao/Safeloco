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
    p.add_argument("--barrier", type=str, default="geometric",
                   choices=["geometric", "sv"],
                   help="geometric = matches the collision metric (contact "
                        "when a link is within r_obs + tol of the axis in XY). "
                        "sv = legged_robot.py's reward-shaping expression, "
                        "h = dist_3d - 2*r - 0.35, whose safe set is EMPTY in "
                        "this corridor (0.95 m demanded vs 0.8 m available).")
    p.add_argument("--body_extent", type=float, default=None,
                   help="geometric barrier only: how far links reach beyond "
                        "the base the barrier is written on. Default 0.20 m.")
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

    # Snapshot the *training* command ranges before build_env_and_policy runs
    # apply_eval_overrides, which collapses them to the fixed eval command
    # (lin_vel_x -> [0.6, 0.6]).  Clamping to those collapsed ranges would pin
    # the filtered command back to nominal every step -- the filter could not
    # even brake, and E4 would be a straw man rather than a steel-man.
    from legged_gym.utils import task_registry  # noqa: E402
    _train_cfg_probe, _ = task_registry.get_cfgs(name=args.task)
    _rng = _train_cfg_probe.commands.ranges
    train_vx = tuple(getattr(_rng, "lin_vel_x", [0.0, 0.8]))
    train_vy = tuple(getattr(_rng, "lin_vel_y", [0.0, 0.0]))

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.eval_terrain, ARGS.eval_envs,
                                dr_mode=ARGS.dr)

    vx_range = train_vx if ARGS.clamp_commands else None
    vy_range = train_vy if ARGS.clamp_commands else None

    # This policy's trained lateral range is [0, 0]: it was never taught to
    # sidestep, so a lateral command is off-distribution and would not be
    # tracked.  The CBF's premise is that commanded velocity IS the realised
    # velocity to first order; commanding motion the policy cannot produce
    # breaks that premise and would make the filter look safe on paper while
    # the robot ignored it.  So the filter's authority here is essentially
    # braking, and that is reported rather than engineered around.
    if ARGS.clamp_commands and vy_range is not None and vy_range[0] == vy_range[1]:
        print("[E4] NOTE: trained lateral command range is [{:g}, {:g}] -- the "
              "filter can only modulate forward speed, not sidestep."
              .format(*vy_range))

    body_margin = None
    if ARGS.body_extent is not None:
        from safeloco_eval.command_filters import BODY_COLLISION_TOL
        body_margin = BODY_COLLISION_TOL + ARGS.body_extent
    cmd_filt = build_command_filter(
        ARGS.filter, alpha=ARGS.alpha, vx_range=vx_range, vy_range=vy_range,
        max_passes=ARGS.max_passes, barrier=ARGS.barrier,
        body_margin=body_margin)

    if ARGS.filter != "none":
        req = cmd_filt.required_clearance(0.3)
        print("[E4] barrier '{}': demands {:.2f} m centre-to-centre clearance "
              "(corridor minimum lateral clearance is 0.8 m per App. G)"
              .format(ARGS.barrier, req))

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
        train_vx_range=list(train_vx), train_vy_range=list(train_vy),
        lateral_authority=(train_vy[1] > train_vy[0]),
        vx_range=list(vx_range) if vx_range else None,
        vy_range=list(vy_range) if vy_range else None,
        barrier=ARGS.barrier,
        barrier_radius_mult=getattr(cmd_filt, "radius_mult", None),
        barrier_body_margin=getattr(cmd_filt, "body_margin", None),
        barrier_uses_xy=getattr(cmd_filt, "use_xy", None),
        barrier_required_clearance_at_r0p3=(
            cmd_filt.required_clearance(0.3)
            if hasattr(cmd_filt, "required_clearance") else None),
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
