"""E2 -- training-time filtering (the CBF-RL mechanism), collision.

analysis_protocol.md §3.2.  Three trained variants, >= 3 seeds each, every
checkpoint later evaluated both with and without the test filter so the
internalization gap (§1) can be computed:

  reward_only   pi_rs -- reward shaping alone, no filter anywhere.  This is
                the protocol's "Reward-only" row and the band the
                reduces-to-shaping test is measured against (~12-14%).
  filter_only   the E1 projection filter applied during training, no shaping.
                The CBF-RL nav-ablation setup: if the policy only looks safe
                while the filter is attached, it never internalized anything.
  dual          both.

**What the PPO update sees.**  The rollout executes the *filtered* action but
the storage keeps the action the policy *sampled*, so the update is credited
with returns produced by actions it did not take.  That mismatch is the
mechanism under test, not an implementation slip -- see wmp_runner.__init__.
`--store_executed` flips it for a contrast run.

**Which filter.**  The same object E1 deployed, projecting against the same
validated Q_hat checkpoint (rho = 0.847, AUROC = 0.881).  Reusing it is what
makes E1 and E2 comparable; it also means Q_hat was fit on pi_nom rollouts and
drifts off-distribution as training moves the policy, which is a real
limitation of any learned-barrier CBF-RL transplant and is reported as one.

**Choosing epsilon.**  Not registered anywhere.  `--calibrate N` runs N short
iterations and reports how hard the filter actually binds (fraction filtered,
KL, return trend) so the choice is made from measurements rather than by hand;
see experiments/e2/README.md.  A filter that never binds makes filter_only a
copy of pi_nom and the experiment vacuous; one that binds too hard collapses
training and says nothing about internalization.

Usage:
    python experiments/e2/train_e2.py --variant filter_only --seed 1 \
        --epsilon 0.2 --task go1_amp --headless

    # binding calibration only, no full run
    python experiments/e2/train_e2.py --variant filter_only --seed 1 \
        --epsilon 0.2 --calibrate 150 --task go1_amp --headless
"""

import argparse
import json
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402

VARIANTS = ("reward_only", "filter_only", "dual")


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--variant", type=str, default="filter_only",
                   choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=1,
                   help="training seed. Protocol §2 wants >= 3 per variant "
                        "and forbids pooling episodes across them.")
    p.add_argument("--rs_weight", type=float, default=1.0,
                   help="reward_only/dual: weight on the safety_value reward "
                        "term. Must match across the two or their returns are "
                        "not comparable.")
    p.add_argument("--filter", type=str, default="A", choices=["A", "B"],
                   help="A = sampling projection, B = gradient. E1 found no "
                        "qualitative difference at matched epsilon.")
    p.add_argument("--epsilon", type=float, default=0.2,
                   help="training-time trust region, radians. See --calibrate.")
    p.add_argument("--tau", type=float, default=-0.25,
                   help="V_hat trigger threshold; matches E1's primary config.")
    p.add_argument("--always_on", action="store_true",
                   help="filter every step instead of triggering on V_hat.")
    p.add_argument("--k_samples", type=int, default=64)
    p.add_argument("--m_steps", type=int, default=3)
    p.add_argument("--delta", type=float, default=None,
                   help="acceptance margin; defaults to the value recorded in "
                        "the Q_hat report (0.05 * IQR).")
    p.add_argument("--qsafe_ckpt", type=str, default="logs/e1/qsafe.pt")
    p.add_argument("--qsafe_report", type=str, default="logs/e1/qsafe.report.json")
    p.add_argument("--filter_seed", type=int, default=1234)
    p.add_argument("--max_iterations", type=int, default=2000,
                   help="matches pi_nom/pi_rs (train_policy.py), so returns "
                        "are comparable against the reference rows.")
    p.add_argument("--calibrate", type=int, default=0,
                   help="run only this many iterations and write a binding "
                        "report instead of a full training run.")
    p.add_argument("--store_executed", action="store_true",
                   help="contrast run: update the policy toward the filtered "
                        "action rather than the sampled one.")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

import torch  # noqa: E402

from safeloco_eval.filters import build_filter  # noqa: E402
from safeloco_eval.qsafe import SafetyActionValue  # noqa: E402
from safeloco_eval import eval_common as EC  # noqa: E402


def uses_filter(variant):
    return variant in ("filter_only", "dual")


def uses_shaping(variant):
    return variant in ("reward_only", "dual")


def apply_variant_config(env_cfg, train_cfg, variant):
    """Config deltas for one E2 variant, returned for the run record.

    The safety branch (pi_ours' reachability critic and null-space
    projection) is off in every E2 variant -- E2 is about the CBF-RL
    mechanism, not about our method.
    """
    delta = {"variant": variant}

    train_cfg.algorithm.safety_coef = 0.0
    train_cfg.algorithm.safety_coef_min = 0.0
    train_cfg.algorithm.safety_coef_max = 0.0
    delta["safety_coef"] = 0.0

    if uses_shaping(variant):
        env_cfg.rewards.scales.safety_value = ARGS.rs_weight
        delta["rewards.scales.safety_value"] = ARGS.rs_weight
    elif hasattr(env_cfg.rewards.scales, "safety_value"):
        # Never let a stale config edit leak shaping into filter_only.
        env_cfg.rewards.scales.safety_value = 0.0
        delta["rewards.scales.safety_value"] = 0.0

    delta["train_filter"] = uses_filter(variant)
    return delta


def resolve_delta():
    """Acceptance margin: the recorded 0.05 * IQR unless overridden."""
    if ARGS.delta is not None:
        return ARGS.delta, "cli"
    path = os.path.join(ROOT, ARGS.qsafe_report) \
        if not os.path.isabs(ARGS.qsafe_report) else ARGS.qsafe_report
    with open(path) as fh:
        return float(json.load(fh)["delta"]), ARGS.qsafe_report


def attach_train_filter(runner, env_cfg, device):
    """Load Q_hat and hang the filter off the runner. Returns the record."""
    ckpt = ARGS.qsafe_ckpt if os.path.isabs(ARGS.qsafe_ckpt) \
        else os.path.join(ROOT, ARGS.qsafe_ckpt)
    qsafe, _meta = SafetyActionValue.load(ckpt, device=device)
    for prm in qsafe.parameters():
        prm.requires_grad_(False)
    delta, delta_src = resolve_delta()

    filt = build_filter(
        ARGS.filter, qsafe,
        epsilon_rad=ARGS.epsilon,
        action_scale=env_cfg.control.action_scale,
        clip_actions=env_cfg.normalization.clip_actions,
        delta=delta,
        tau=ARGS.tau,
        always_on=ARGS.always_on,
        k_samples=ARGS.k_samples,
        m_steps=ARGS.m_steps,
        device=str(device),
        seed=ARGS.filter_seed)

    runner.train_filter = filt
    runner.train_qsafe = qsafe
    runner.train_store_executed = ARGS.store_executed

    return {
        "filter_variant": ARGS.filter,
        "epsilon_rad": ARGS.epsilon,
        "tau": None if ARGS.always_on else ARGS.tau,
        "always_on": ARGS.always_on,
        "delta": delta,
        "delta_source": delta_src,
        "k_samples": ARGS.k_samples,
        "m_steps": ARGS.m_steps,
        "filter_seed": ARGS.filter_seed,
        "store_executed": ARGS.store_executed,
        "qsafe_ckpt": ARGS.qsafe_ckpt,
        "qsafe_sha": EC.sha256_file(ckpt),
    }


def run_name():
    base = "e2_{}_s{}".format(ARGS.variant, ARGS.seed)
    if ARGS.calibrate:
        # Calibration sweeps epsilon, so the run name has to carry it or every
        # point writes into the same directory and appends to the same
        # train_diag.csv -- which silently turns the per-epsilon report into a
        # summary of the concatenation.
        return "{}_cal_eps{:g}".format(base, ARGS.epsilon)
    return base


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg.seed = ARGS.seed
    train_cfg.runner.run_name = run_name()
    train_cfg.runner.save_interval = 500
    train_cfg.runner.max_iterations = ARGS.calibrate or ARGS.max_iterations

    delta = apply_variant_config(env_cfg, train_cfg, ARGS.variant)

    env, env_cfg = task_registry.make_env(name=args.task, args=args,
                                          env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                           train_cfg.runner.experiment_name,
                           train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)

    filter_record = None
    if uses_filter(ARGS.variant):
        filter_record = attach_train_filter(runner, env_cfg, runner.device)

    # Start each run's diagnostics clean.  Appending across runs is how the
    # first calibration sweep produced a report covering all four epsilon
    # values at once; re-running any single config would do the same.
    runner.e2_diag_path = os.path.join(log_dir, "train_diag.csv")
    if os.path.exists(runner.e2_diag_path):
        os.replace(runner.e2_diag_path, runner.e2_diag_path + ".prev")

    record = dict(delta,
                  run_name=train_cfg.runner.run_name,
                  seed=ARGS.seed,
                  calibrate=ARGS.calibrate,
                  max_iterations=train_cfg.runner.max_iterations,
                  num_envs=env_cfg.env.num_envs,
                  filter=filter_record,
                  action_scale=env_cfg.control.action_scale,
                  clip_actions=env_cfg.normalization.clip_actions,
                  use_amp=train_cfg.runner.use_amp,
                  reward_scales=class_to_dict(env_cfg.rewards.scales),
                  git_commit=EC.git_commit())
    with open(os.path.join(log_dir, "e2_config.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    print("=" * 70)
    print("E2 {}  seed {}  ->  {}".format(ARGS.variant, ARGS.seed, log_dir))
    for k, v in delta.items():
        print("  {} = {}".format(k, v))
    if filter_record:
        print("  training filter: {} eps={} tau={} delta={:.6f}".format(
            filter_record["filter_variant"], filter_record["epsilon_rad"],
            filter_record["tau"], filter_record["delta"]))
    if ARGS.calibrate:
        print("  CALIBRATION RUN -- {} iterations, no checkpoint expected"
              .format(ARGS.calibrate))
    print("=" * 70)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)

    if ARGS.calibrate:
        report_calibration(runner.e2_diag_path, log_dir)


def report_calibration(diag_path, log_dir):
    """Summarise how hard the filter bound during a short run.

    Reports the three things that decide whether an epsilon is usable: does
    the filter engage at all, how far does it move the action in units of the
    policy's own noise, and did return trend down while it did so.
    """
    import csv
    import statistics as st

    if not os.path.exists(diag_path):
        print("no diagnostics written -- nothing to calibrate on")
        return

    with open(diag_path) as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        print("diagnostics file is empty")
        return

    # Defensive: if a file ever holds more than one run (the iteration counter
    # restarts), summarise only the last one rather than averaging across
    # runs that had different settings.
    starts = [i for i, r in enumerate(rows)
              if i == 0 or _int(r.get("iteration")) <= _int(rows[i - 1].get("iteration"))]
    if len(starts) > 1:
        print("WARNING: {} runs found in {}; reporting only the last {} rows. "
              "Earlier segments had different settings and were not averaged in."
              .format(len(starts), diag_path, len(rows) - starts[-1]))
        rows = rows[starts[-1]:]

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, TypeError, ValueError):
                pass
        return out

    frac, kl = col("frac_filtered"), col("mean_kl_exec_sampled")
    ret = [x for x in col("mean_return") if x == x]
    half = max(len(ret) // 2, 1)

    summary = {
        "iterations": len(rows),
        "epsilon_rad": ARGS.epsilon,
        "variant": ARGS.variant,
        "frac_filtered_mean": st.mean(frac) if frac else None,
        "kl_exec_sampled_mean": st.mean(kl) if kl else None,
        "return_first_half": st.mean(ret[:half]) if ret else None,
        "return_second_half": st.mean(ret[half:]) if ret[half:] else None,
    }
    if summary["return_first_half"] is not None and summary["return_second_half"] is not None:
        summary["return_trend"] = summary["return_second_half"] - summary["return_first_half"]

    out = os.path.join(log_dir, "calibration.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print("CALIBRATION  eps = {}  variant = {}".format(ARGS.epsilon, ARGS.variant))
    print("  fraction of steps filtered : {:.3f}".format(summary["frac_filtered_mean"] or 0.0))
    print("  mean KL(executed||sampled) : {:.4f}".format(summary["kl_exec_sampled_mean"] or 0.0))
    print("  return  first half -> second half : {} -> {}".format(
        _fmt(summary["return_first_half"]), _fmt(summary["return_second_half"])))
    print("\n  A usable epsilon binds (fraction filtered well above 0) without")
    print("  return trending down over the calibration window.  Compare across")
    print("  epsilon values before committing to the full 9-run matrix.")
    print("  wrote {}".format(out))
    print("=" * 70)


def _fmt(x):
    return "n/a" if x is None else "{:.2f}".format(x)


def _int(x, default=-1):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
