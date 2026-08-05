"""E1 -- deployment-time joint-space projection, collision.  One sweep point.

experiment_protocol_phased.md §3.  Evaluation only: pi_nom is frozen, the
filter sits between `act_inference(...)` and `env.step(...)`, and nothing is
trained.

Registered predictions (fixed before running, reported verbatim either way):
  (i)   fall rate rises monotonically with eps, clearly above unfiltered by
        eps ~ 0.2;
  (ii)  at small eps, collision ~ unfiltered and >> 4.4% (ours), with a high
        target-miss rate;
  (iii) mean |lateral vel| stays near pi_nom's -- no emergent sidestepping;
  (iv)  no eps lands on the Pareto front (33.9 return / 4.4% / 0.085 vel err).

Falsification: some eps with fall ~ unfiltered AND collision <= ~6% AND return
>= ~30.  `analyze_e1.py` applies the rule mechanically; this script only
produces the numbers.

Usage (one point)
-----------------
    python experiments/e1/run_e1.py --task go1_amp --headless \
        --load_run <PI_NOM_RUN> --qsafe_ckpt logs/e1/qsafe.pt \
        --filter A --epsilon 0.1 --out_dir logs/e1/sweep

Run the whole sweep with experiments/e1/sweep_e1.sh.
"""

import argparse
import inspect
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip, add_common_eval_args  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    add_common_eval_args(p)
    p.add_argument("--filter", type=str, default="A",
                   choices=["none", "A", "B"],
                   help="A = sampling projection (primary), B = gradient "
                        "projection, none = unfiltered pi_nom reference")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="trust-region radius in RADIANS of joint command; "
                        "converted internally by action_scale")
    p.add_argument("--k_samples", type=int, default=64)
    p.add_argument("--m_steps", type=int, default=3,
                   help="variant B gradient steps (eta = epsilon / m)")
    p.add_argument("--grad_normalize", type=str, default="inf",
                   choices=["inf", "none"])
    p.add_argument("--tau", type=float, default=-0.25,
                   help="trigger threshold. Primary = d_safe = -0.25 "
                        "(Table 4); secondary = 0.0. NOTE: "
                        "cone_constraint.py currently hardcodes d_safe=-0.4 "
                        "inside the function body -- addendum §2, top item.")
    p.add_argument("--always_on", action="store_true",
                   help="ablation: no trigger, filter every step")
    p.add_argument("--trigger_source", type=str, default="vhat",
                   choices=["vhat", "qpol", "sv_oracle"])
    p.add_argument("--delta", type=float, default=None,
                   help="override the 0.05*IQR margin stored in the ckpt")
    p.add_argument("--qsafe_ckpt", type=str, default=None)
    p.add_argument("--qsafe_ensemble", type=str, nargs="*", default=None,
                   help="extra member checkpoints -> min-aggregated control")
    p.add_argument("--filter_seed", type=int, default=1234)
    p.add_argument("--policy_tag", type=str, default="pi_nom")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args  # noqa: E402

import torch  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402
from safeloco_eval.filters import build_filter  # noqa: E402
from safeloco_eval.qsafe import SafetyActionValue, QSafeEnsemble  # noqa: E402


def default_run_id():
    if ARGS.filter == "none":
        return "{}_unfiltered".format(ARGS.policy_tag)
    tag = "{}_{}_eps{:.3f}".format(ARGS.policy_tag, ARGS.filter, ARGS.epsilon)
    if ARGS.always_on:
        tag += "_alwayson"
    if abs(ARGS.tau + 0.25) > 1e-9:
        tag += "_tau{:g}".format(ARGS.tau)
    if ARGS.trigger_source != "vhat":
        tag += "_" + ARGS.trigger_source
    if ARGS.qsafe_ensemble:
        tag += "_ens{}".format(len(ARGS.qsafe_ensemble) + 1)
    if ARGS.filter == "B" and ARGS.grad_normalize != "inf":
        tag += "_rawgrad"
    return tag


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.eval_terrain, ARGS.eval_envs,
                                dr_mode=ARGS.dr)

    qsafe, qsafe_meta = None, {}
    if ARGS.qsafe_ckpt:
        qsafe, qsafe_meta = SafetyActionValue.load(ARGS.qsafe_ckpt,
                                                   device=env.device)
        if ARGS.qsafe_ensemble:
            members = [qsafe] + [
                SafetyActionValue.load(p, device=env.device)[0]
                for p in ARGS.qsafe_ensemble]
            qsafe = QSafeEnsemble(members).to(env.device)
        for p in qsafe.parameters():
            p.requires_grad_(False)
    elif ARGS.filter != "none":
        raise SystemExit("--qsafe_ckpt is required for filter variants A/B")

    filt = build_filter(
        variant=ARGS.filter, qsafe=qsafe, epsilon_rad=ARGS.epsilon,
        action_scale=float(env_cfg.control.action_scale),
        clip_actions=float(env_cfg.normalization.clip_actions),
        tau=ARGS.tau, delta=ARGS.delta, always_on=ARGS.always_on,
        k_samples=ARGS.k_samples, m_steps=ARGS.m_steps,
        grad_normalize=ARGS.grad_normalize, device=env.device,
        seed=ARGS.filter_seed)

    variant = filt.name + ("_alwayson" if ARGS.always_on else "")
    meta = {"run_id": run_id, "method": "E1", "variant": variant,
            "epsilon_or_alpha": (ARGS.epsilon if ARGS.filter != "none" else ""),
            "terrain": ARGS.eval_terrain}

    collector = EC.EvalCollector(env, runner, policy, filt, qsafe=qsafe,
                                 trigger_source=ARGS.trigger_source,
                                 run_meta=meta)
    seed_list, _ = S.load(ARGS.eval_seed_file)
    records = collector.run(ARGS.n_episodes, seed_list)

    csv_path = os.path.join(ARGS.out_dir, run_id + ".csv")
    EC.write_records(csv_path, records)
    EC.write_manifest(
        os.path.join(ARGS.out_dir, run_id + ".manifest.json"),
        run_id=run_id, method="E1", variant=variant,
        epsilon_rad=(ARGS.epsilon if ARGS.filter != "none" else None),
        policy_tag=ARGS.policy_tag,
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        qsafe_ckpt=ARGS.qsafe_ckpt,
        qsafe_sha=(EC.sha256_file(ARGS.qsafe_ckpt) if ARGS.qsafe_ckpt else None),
        qsafe_battery=qsafe_meta.get("report", {}).get("battery"),
        qsafe_verdict=qsafe_meta.get("report", {}).get("verdict"),
        delta=(float(filt.delta) if hasattr(filt, "delta") else None),
        tau=ARGS.tau, always_on=ARGS.always_on,
        trigger_source=ARGS.trigger_source,
        k_samples=ARGS.k_samples, m_steps=ARGS.m_steps,
        grad_normalize=ARGS.grad_normalize, filter_seed=ARGS.filter_seed,
        terrain=ARGS.eval_terrain, dr_mode=ARGS.dr,
        n_episodes=len(records), eval_envs=ARGS.eval_envs,
        terrain_seed=EC.TERRAIN_SEED, cmd_lin_vel_x=EC.EVAL_CMD_LIN_VEL_X,
        action_scale=float(env_cfg.control.action_scale),
        clip_actions=float(env_cfg.normalization.clip_actions),
        control_dt=float(env.dt),
    )

    # A one-line console summary so a sweep is watchable without the analyser.
    from safeloco_eval import metrics as M
    ck, cn = M.collision_rate(records)
    fk, fn = M.fall_rate(records)
    ak, an = M.activation_rate(records)
    mk, mn = M.target_miss_rate(records)
    print("\n[E1] {}  episodes={}  collision={:.2f}%  fall={:.2f}%  "
          "return={:.2f}  activation={:.2f}%  target-miss={:.2f}%".format(
              run_id, len(records), 100.0 * ck / max(cn, 1),
              100.0 * fk / max(fn, 1),
              sum(float(r["return"]) for r in records) / max(len(records), 1),
              100.0 * ak / max(an, 1), 100.0 * mk / max(mn, 1)))
    print("[E1] wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
