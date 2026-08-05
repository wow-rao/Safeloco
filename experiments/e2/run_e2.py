"""E2 evaluation -- one trained checkpoint, with and without the test filter.

analysis_protocol.md §3.2 turns on the *internalization gap* (§1):

    gap = metric(with test filter) - metric(without)

so every E2 checkpoint is evaluated twice under identical seeds, and the two
rows differ only in whether the filter is attached at deployment.  Positive
gap = worse without the filter = the policy was leaning on it.

Emits the shared per-episode schema (§6), so E1 and E2 rows concatenate.

Collision is the geometric link-vs-cylinder test, matching E1 rather than
§1's `min_cbf_h < -0.05`.  On identical E1 runs the two conventions give
11.83% and 49.25%; only the geometric one is commensurate with the
reward-shaping band (~12-14%) and the 4.4% reference that §3.2's verdict rule
compares against.  The margin threshold is retained as the secondary
proximity column.

Usage:
    # with the test filter
    python experiments/e2/run_e2.py --load_run e2_filter_only_s1 \
        --variant_tag filter_only --seed_tag 1 --test_filter on \
        --qsafe_ckpt logs/e1/qsafe.pt --epsilon 0.2 --task go1_amp --headless

    # same checkpoint, filter removed
    python experiments/e2/run_e2.py --load_run e2_filter_only_s1 \
        --variant_tag filter_only --seed_tag 1 --test_filter off \
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
    p.add_argument("--test_filter", type=str, default="off",
                   choices=["on", "off"],
                   help="attach the filter at deployment or not. The pair of "
                        "runs is what the internalization gap is computed from.")
    p.add_argument("--variant_tag", type=str, required=True,
                   choices=["reward_only", "filter_only", "dual"],
                   help="which E2 training variant this checkpoint came from.")
    p.add_argument("--seed_tag", type=int, required=True,
                   help="training seed of the checkpoint; kept in the record "
                        "so the analyser can group by seed without parsing "
                        "run names.")
    p.add_argument("--filter", type=str, default="A", choices=["A", "B"])
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=-0.25)
    p.add_argument("--always_on", action="store_true")
    p.add_argument("--k_samples", type=int, default=64)
    p.add_argument("--m_steps", type=int, default=3)
    p.add_argument("--grad_normalize", type=str, default="inf")
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--qsafe_ckpt", type=str, default=None)
    p.add_argument("--filter_seed", type=int, default=1234)
    p.add_argument("--trigger_source", type=str, default="vhat")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--eval_envs", type=int, default=250)
    p.add_argument("--eval_terrain", type=str, default="corridor")
    p.add_argument("--eval_seed_file", type=str, default=None)
    p.add_argument("--dr", type=str, default="off")
    p.add_argument("--out_dir", type=str, default="logs/e2/eval")
    p.add_argument("--run_id", type=str, default=None)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402
from safeloco_eval.filters import build_filter  # noqa: E402
from safeloco_eval.qsafe import SafetyActionValue  # noqa: E402


def default_run_id():
    return "e2_{}_s{}_tf{}".format(ARGS.variant_tag, ARGS.seed_tag,
                                   ARGS.test_filter)


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()
    os.makedirs(ARGS.out_dir, exist_ok=True)

    filter_on = ARGS.test_filter == "on"
    if filter_on and not ARGS.qsafe_ckpt:
        raise SystemExit("--qsafe_ckpt is required when --test_filter on")

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.eval_terrain, ARGS.eval_envs,
                                dr_mode=ARGS.dr)

    qsafe, qsafe_meta = None, {}
    if ARGS.qsafe_ckpt:
        qsafe, qsafe_meta = SafetyActionValue.load(ARGS.qsafe_ckpt,
                                                   device=env.device)
        for prm in qsafe.parameters():
            prm.requires_grad_(False)

    filt = build_filter(
        variant=(ARGS.filter if filter_on else "none"),
        qsafe=(qsafe if filter_on else None),
        epsilon_rad=ARGS.epsilon,
        action_scale=float(env_cfg.control.action_scale),
        clip_actions=float(env_cfg.normalization.clip_actions),
        tau=ARGS.tau, delta=ARGS.delta, always_on=ARGS.always_on,
        k_samples=ARGS.k_samples, m_steps=ARGS.m_steps,
        grad_normalize=ARGS.grad_normalize, device=env.device,
        seed=ARGS.filter_seed)

    # `variant` carries the training variant and the deployment condition, so
    # a concatenated E1+E2 table stays unambiguous about which row is which.
    variant = "{}_tf{}".format(ARGS.variant_tag, ARGS.test_filter)
    meta = {"run_id": run_id, "method": "E2", "variant": variant,
            "epsilon_or_alpha": (ARGS.epsilon if filter_on else ""),
            "seed": ARGS.seed_tag,
            "terrain": ARGS.eval_terrain}

    collector = EC.EvalCollector(
        env, runner, policy, filt,
        qsafe=(qsafe if filter_on else None),
        trigger_source=ARGS.trigger_source, run_meta=meta)
    seed_list, _ = S.load(ARGS.eval_seed_file)
    records = collector.run(ARGS.n_episodes, seed_list)

    csv_path = os.path.join(ARGS.out_dir, run_id + ".csv")
    EC.write_records(csv_path, records)
    EC.write_manifest(
        os.path.join(ARGS.out_dir, run_id + ".manifest.json"),
        run_id=run_id, method="E2", variant=variant,
        train_variant=ARGS.variant_tag, train_seed=ARGS.seed_tag,
        test_filter=ARGS.test_filter,
        epsilon_rad=(ARGS.epsilon if filter_on else None),
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        qsafe_ckpt=ARGS.qsafe_ckpt,
        qsafe_sha=(EC.sha256_file(ARGS.qsafe_ckpt) if ARGS.qsafe_ckpt else None),
        qsafe_battery=qsafe_meta.get("report", {}).get("battery"),
        delta=(float(filt.delta) if hasattr(filt, "delta") else None),
        tau=ARGS.tau, always_on=ARGS.always_on,
        trigger_source=ARGS.trigger_source,
        k_samples=ARGS.k_samples, m_steps=ARGS.m_steps,
        grad_normalize=ARGS.grad_normalize, filter_seed=ARGS.filter_seed,
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
    print("[E2] {}  episodes={}  collision={:.2f}%  fall={:.2f}%  "
          "return={:.2f}  activation={:.2f}%".format(
              run_id, len(records),
              100.0 * ck / max(cn, 1), 100.0 * fk / max(fn, 1),
              ST.mean(M.per_episode(records, "return")),
              100.0 * ak / max(an, 1)))
    print("  wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
