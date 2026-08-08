"""E5 -- bipedal fall safety under directed pushes.  One sweep cell.

Evaluation only: the arm's checkpoint is frozen, a PushSchedule injects
directed base-velocity impulses on a fixed step schedule, and the collector
records falls, margins, and the reachability critic's value per step.

Registered predictions live in experiments/e5/README.md; analyze_e5.py
applies them mechanically.  This script only produces the numbers.

Usage (one cell)
----------------
    python experiments/e5/run_e5.py --task cassie --headless \
        --load_run pi_nom --arm pi_nom --cmd_vx 0.8 --push_mag 1.0 \
        --push_dirs all --eval_terrain flat --out_dir logs/e5/sweep

Run the whole grid with experiments/e5/sweep_e5.sh.
"""

import argparse
import inspect
import json
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip, add_common_eval_args  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    add_common_eval_args(p)
    p.add_argument("--arm", type=str, default="pi_nom",
                   help="row label: pi_nom / pi_rs / pi_ours / pi_ours_dcm")
    p.add_argument("--cmd_vx", type=float, default=0.8,
                   help="pinned forward command [m/s]")
    p.add_argument("--push_mag", type=float, default=0.0,
                   help="impulse magnitude [m/s]; 0 disables pushes")
    p.add_argument("--push_dirs", type=str, default="all",
                   choices=["all", "cardinal", "random"])
    p.add_argument("--push_mode", type=str, default="add",
                   choices=["add", "set"],
                   help="add: shove on top of current velocity (default); "
                        "set: overwrite like training-time _push_robots")
    p.add_argument("--push_p0", type=int, default=200,
                   help="first push step (200 = 4 s in, gait settled)")
    p.add_argument("--push_dp", type=int, default=150,
                   help="steps between pushes (150 = 3 s)")
    p.add_argument("--push_seed", type=int, default=1234)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args  # noqa: E402

import torch  # noqa: E402,F401

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import eval_biped as EB  # noqa: E402
from safeloco_eval import metrics as M  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402


def default_run_id():
    tag = "{}_vx{:g}_m{:g}".format(ARGS.arm, ARGS.cmd_vx, ARGS.push_mag)
    if ARGS.push_mag > 0:
        tag += "_" + ARGS.push_dirs
        if ARGS.push_mode != "add":
            tag += "_set"
    if ARGS.eval_terrain != "flat":
        tag += "_" + ARGS.eval_terrain
    return tag


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EB.build_biped_env_and_policy(
            args, ARGS.eval_terrain, ARGS.eval_envs, ARGS.cmd_vx,
            dr_mode=ARGS.dr, load_run=(args.load_run or None),
            checkpoint=(args.checkpoint if args.checkpoint not in (None, -1)
                        else None))

    push = EB.PushSchedule(ARGS.push_mag, mode=ARGS.push_mode,
                           dirs=ARGS.push_dirs, p0=ARGS.push_p0,
                           dp=ARGS.push_dp, seed=ARGS.push_seed)

    meta = {"run_id": run_id, "method": "E5", "variant": ARGS.arm,
            "epsilon_or_alpha": ARGS.push_mag, "terrain": ARGS.eval_terrain,
            "cmd_vx": ARGS.cmd_vx, "push_mag": ARGS.push_mag,
            "push_mode": ARGS.push_mode}

    collector = EB.BipedEvalCollector(env, runner, policy, push,
                                      run_meta=meta)
    seed_list, _ = S.load(ARGS.eval_seed_file)
    records = collector.run(ARGS.n_episodes, seed_list)

    # The checkpoint's own training record is the truth about how the
    # safety branch was configured -- the eval-time train_cfg reflects the
    # shipped config defaults, not the trained run.
    train_record = {}
    tr_path = os.path.join(os.path.dirname(resume_path), "policy_config.json")
    if os.path.exists(tr_path):
        try:
            with open(tr_path) as f:
                train_record = json.load(f)
        except (OSError, ValueError):
            train_record = {}

    def _trained_field(key, fallback):
        return train_record[key] if key in train_record else fallback

    trained_safety_coef = float(_trained_field(
        "safety_coef", train_cfg.algorithm.safety_coef) or 0)
    trained_critic_when_off = bool(_trained_field(
        "train_safety_critic_when_off",
        getattr(train_cfg.algorithm, "train_safety_critic_when_off", False)))

    csv_path = os.path.join(ARGS.out_dir, run_id + ".csv")
    EC.write_records(csv_path, records)
    collector.save_calibration(os.path.join(ARGS.out_dir, run_id + ".npz"))
    EC.write_manifest(
        os.path.join(ARGS.out_dir, run_id + ".manifest.json"),
        run_id=run_id, method="E5", arm=ARGS.arm,
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        terrain=ARGS.eval_terrain, dr_mode=ARGS.dr,
        cmd_vx=ARGS.cmd_vx, push_mag=ARGS.push_mag,
        push_dirs=ARGS.push_dirs, push_mode=ARGS.push_mode,
        push_p0=ARGS.push_p0, push_dp=ARGS.push_dp,
        push_seed=ARGS.push_seed,
        fall_safety_mode=env_cfg.fall_safety.mode,
        h_nom=env_cfg.fall_safety.h_nom,
        safety_coef=trained_safety_coef,
        train_safety_critic_when_off=trained_critic_when_off,
        safety_return_alpha=_trained_field(
            "safety_return_alpha",
            getattr(train_cfg.algorithm, "safety_return_alpha", 0.7)),
        vsafe_trained=(trained_safety_coef > 0 or trained_critic_when_off),
        n_episodes=len(records), eval_envs=ARGS.eval_envs,
        terrain_seed=EC.TERRAIN_SEED,
        control_dt=float(env.dt),
        push_fall_window_s=M.PUSH_FALL_WINDOW_S,
    )

    # One-line console summary so a sweep is watchable without the analyser.
    k_fall, n_eps = M.fall_rate(records)
    k1000, kk = M.falls_per_1000_steps(records)
    pushed_k, pushed_n = M.push_attributed_fall_rate(records)
    print("[e5] {}  falls {}/{} eps ({:.1f}%)  {:.2f}/1k steps  "
          "push-attributed {}/{}  mean margin {:+.3f}".format(
              run_id, k_fall, n_eps, 100.0 * k_fall / max(1, n_eps),
              k1000 / max(kk, 1e-9), pushed_k, pushed_n,
              sum(float(r["mean_fall_margin"]) for r in records)
              / max(1, len(records))))


if __name__ == "__main__":
    main()
