"""E2 analysis -- applies analysis_protocol.md §3.2 mechanically.

Ingests the per-episode CSVs written by run_e2.py (one per variant x seed x
deployment condition) plus the per-iteration train_diag.csv from each training
run, and emits the R-table, the verdict, and the §7 rebuttal sentence.

Two things this file is careful about, both from §2:

  * **Never pool episodes across seeds.**  Every cell is summarised as the
    mean +/- std *across per-seed means*.  Pooling would understate
    uncertainty by roughly sqrt(episodes per seed) and is the single easiest
    way to manufacture a verdict that is not there.
  * **Seed intervals at n = 3 are wide.**  The registered rule asks for
    non-overlapping seed-level intervals; with three seeds a 95% t-interval
    is mean +/- 4.30 * SEM.  That is reported honestly rather than swapped
    for something narrower, and the paired per-seed analysis is reported
    alongside as a *supporting, unregistered* diagnostic -- it is far more
    powerful, because with/without share a checkpoint and an eval seed list.

Run:
    python experiments/e2/analyze_e2.py --dir logs/e2/eval \
        --train_dir logs/go1_viploco_warp_v5 --out logs/e2/R2a
"""

import argparse
import csv
import json
import os
import sys

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M      # noqa: E402
from safeloco_eval import stats as ST       # noqa: E402

VARIANTS = ("reward_only", "filter_only", "dual")
CONDITIONS = ("on", "off")

# Two-sided 95% t quantiles; n = 3 seeds is the protocol minimum, so the
# small-df end is the part that matters.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093,
       29: 2.045}


def t_quantile(df):
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    return T95[min(keys, key=lambda k: abs(k - df))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e2/eval",
                   help="directory of run_e2.py per-episode CSVs")
    p.add_argument("--train_dir", type=str, default=None,
                   help="parent directory of the E2 training runs, for the "
                        "channel diagnostics (train_diag.csv per run)")
    p.add_argument("--out", type=str, default="logs/e2/R2a")
    p.add_argument("--interval", type=str, default="t95",
                   choices=["t95", "std"],
                   help="what 'seed-level interval' means in the §3.2 verdict. "
                        "t95 = mean +/- t*SEM across seeds (default, "
                        "conservative); std = mean +/- 1 std across seeds, "
                        "the quantity §2 asks to report.")
    return p.parse_args()


# ----------------------------------------------------------------- loading --

def load_cells(directory):
    """-> {(variant, test_filter): {seed: [records]}}"""
    cells = {}
    if not os.path.isdir(directory):
        raise SystemExit("no such directory: {}".format(directory))
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(directory, name)) as fh:
            recs = list(csv.DictReader(fh))
        if not recs:
            continue
        variant_field = recs[0].get("variant", "")
        # run_e2 writes "<train_variant>_tf<on|off>"
        if "_tf" not in variant_field:
            continue
        train_variant, cond = variant_field.rsplit("_tf", 1)
        if train_variant not in VARIANTS or cond not in CONDITIONS:
            continue
        seed = recs[0].get("seed", "")
        cells.setdefault((train_variant, cond), {}).setdefault(seed, []).extend(recs)
    return cells


def per_seed_rates(by_seed, fn):
    """Per-seed value of a (k, n) rate metric, in seed order."""
    return [fn(recs)[0] / max(fn(recs)[1], 1)
            for _seed, recs in sorted(by_seed.items())]


def per_seed_means(by_seed, field):
    return [ST.mean(M.per_episode(recs, field))
            for _seed, recs in sorted(by_seed.items())]


def summarise(values, mode):
    """(mean, lo, hi, std, n) across seeds."""
    n = len(values)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan, 0
    m = ST.mean(values)
    if n == 1:
        return m, m, m, 0.0, 1
    s = ST.std(values)
    if mode == "std":
        return m, m - s, m + s, s, n
    half = t_quantile(n - 1) * (s / (n ** 0.5))
    return m, m - half, m + half, s, n


def cell_summary(by_seed, mode):
    coll = per_seed_rates(by_seed, M.collision_rate)
    fall = per_seed_rates(by_seed, M.fall_rate)
    return {
        "n_seeds": len(by_seed),
        "seeds": sorted(by_seed.keys()),
        "collision_per_seed": coll,
        "fall_per_seed": fall,
        "collision": summarise(coll, mode),
        "fall": summarise(fall, mode),
        "ret": summarise(per_seed_means(by_seed, "return"), mode),
        "vel_err": summarise(per_seed_means(by_seed, "vel_err"), mode),
        "n_episodes": sum(len(r) for r in by_seed.values()),
    }


# ----------------------------------------------------------------- verdict --

def paired_gap(cells, variant):
    """Per-seed (without - with) collision differences, matched by seed.

    Supporting diagnostic only -- not the registered rule.  Each seed
    contributes one checkpoint evaluated twice on the same seed list, so the
    difference is paired and far tighter than comparing two cells.
    """
    off = cells.get((variant, "off"), {})
    on = cells.get((variant, "on"), {})
    shared = sorted(set(off) & set(on))
    diffs = []
    for s in shared:
        k_off, n_off = M.collision_rate(off[s])
        k_on, n_on = M.collision_rate(on[s])
        diffs.append(k_off / max(n_off, 1) - k_on / max(n_on, 1))
    return shared, diffs


def apply_verdict(cells, summaries, mode):
    checks, detail = {}, {}

    # 1. Internalization failure: filter_only is materially worse without the
    #    test filter than with it, seed-level intervals not overlapping.
    fo_off = summaries.get(("filter_only", "off"))
    fo_on = summaries.get(("filter_only", "on"))
    if fo_off and fo_on and fo_off["n_seeds"] and fo_on["n_seeds"]:
        m_off, lo_off, hi_off = fo_off["collision"][:3]
        m_on, lo_on, hi_on = fo_on["collision"][:3]
        checks["internalization_failure"] = bool(
            m_off > m_on and not ST.ci_overlap((lo_off, hi_off), (lo_on, hi_on)))
        detail["internalization"] = {
            "filter_only_without": fmt_pct(m_off, lo_off, hi_off),
            "filter_only_with": fmt_pct(m_on, lo_on, hi_on),
            "gap_pp": 100.0 * (m_off - m_on),
        }
    else:
        checks["internalization_failure"] = None
        detail["internalization"] = "filter_only rows missing"

    # 2. Reduces to shaping: dual-without lands in the shaping band rather
    #    than near ours.
    du_off = summaries.get(("dual", "off"))
    if du_off and du_off["n_seeds"]:
        m = du_off["collision"][0]
        checks["reduces_to_shaping"] = bool(M.in_shaping_band(m))
        detail["reduces_to_shaping"] = {
            "dual_without": fmt_pct(*du_off["collision"][:3]),
            "shaping_band": "{:.0f}-{:.0f}%".format(
                100 * M.SHAPING_BAND_LO, 100 * M.SHAPING_BAND_HI),
            "ours_reference": "{:.1f}%".format(100 * M.OURS_COLLISION_RATE),
        }
        ro_off = summaries.get(("reward_only", "off"))
        if ro_off and ro_off["n_seeds"]:
            detail["reduces_to_shaping"]["measured_reward_only"] = \
                fmt_pct(*ro_off["collision"][:3])
    else:
        checks["reduces_to_shaping"] = None
        detail["reduces_to_shaping"] = "dual rows missing"

    holds = any(v is True for v in checks.values())
    return {"checks": checks, "detail": detail, "holds": holds,
            "interval_mode": mode}


def channel_attribution(train_dir):
    """§3.2's channel question, from the per-iteration training diagnostics."""
    if not train_dir or not os.path.isdir(train_dir):
        return None
    out = {}
    for name in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, name, "train_diag.csv")
        if not name.startswith("e2_") or not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        def col(key):
            vals = []
            for r in rows:
                try:
                    v = float(r[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if v == v:
                    vals.append(v)
            return vals

        wm = col("wm_model_loss")
        out[name] = {
            "iterations": len(rows),
            "wm_model_loss": ST.mean(wm) if wm else None,
            "mean_kl": ST.mean(col("mean_kl_exec_sampled")) or 0.0,
            "frac_filtered": ST.mean(col("frac_filtered")) or 0.0,
            "final_return": (col("mean_return") or [float("nan")])[-1],
        }
    return out or None


def summarise_channels(chan):
    """Turn the per-run diagnostics into the sentence §3.2 asks for."""
    if not chan:
        return None
    def avg(pred, key):
        vals = [v[key] for k, v in chan.items() if pred(k) and v[key] is not None]
        return ST.mean(vals) if vals else None

    filtered = lambda k: ("filter_only" in k) or ("dual" in k)          # noqa: E731
    unfiltered = lambda k: "reward_only" in k                            # noqa: E731

    wm_f, wm_u = avg(filtered, "wm_model_loss"), avg(unfiltered, "wm_model_loss")
    kl_f = avg(filtered, "mean_kl")

    verdict = "insufficient diagnostics"
    if wm_f is not None and wm_u is not None:
        wm_elevated = wm_f > 1.10 * wm_u
        kl_elevated = (kl_f or 0.0) > 0.01
        if wm_elevated and kl_elevated:
            verdict = "both channels: world-model loss and PPO ratio elevated"
        elif wm_elevated:
            verdict = ("world-model channel: filtered actions corrupted the "
                       "behaviour-cloning target")
        elif kl_elevated:
            verdict = "PPO channel alone: model loss flat, executed/sampled KL elevated"
        else:
            verdict = ("neither channel: model loss flat and KL small -- "
                       "training was stable, report as a clean negative")
    return {"wm_loss_filtered": wm_f, "wm_loss_unfiltered": wm_u,
            "kl_filtered": kl_f, "verdict": verdict}


# ------------------------------------------------------------------ output --

def fmt_pct(m, lo, hi):
    return "{:.1f} [{:.1f}, {:.1f}]".format(100 * m, 100 * lo, 100 * hi)


ROW_ORDER = [(v, c) for v in VARIANTS for c in ("on", "off")]
LABEL = {"on": "with test filter", "off": "without"}


def markdown_table(summaries):
    head = ("| Variant | Test filter | Collision % ↓ | Fall % ↓ | Return ↑ | "
            "Vel. err ↓ | Seeds |")
    out = [head, "|---|---|---|---|---|---|---|"]
    for key in ROW_ORDER:
        s = summaries.get(key)
        if not s or not s["n_seeds"]:
            continue
        v, cond = key
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            v, LABEL[cond],
            fmt_pct(*s["collision"][:3]), fmt_pct(*s["fall"][:3]),
            "{:.2f} ± {:.2f}".format(s["ret"][0], s["ret"][3]),
            "{:.3f} ± {:.3f}".format(s["vel_err"][0], s["vel_err"][3]),
            s["n_seeds"]))
    return "\n".join(out)


def per_seed_appendix(summaries):
    """§2 wants the individual seed values shown, not just the summary."""
    out = ["| Variant | Test filter | Collision % per seed | Fall % per seed |",
           "|---|---|---|---|"]
    for key in ROW_ORDER:
        s = summaries.get(key)
        if not s or not s["n_seeds"]:
            continue
        v, cond = key
        out.append("| {} | {} | {} | {} |".format(
            v, LABEL[cond],
            ", ".join("{:.1f}".format(100 * x) for x in s["collision_per_seed"]),
            ", ".join("{:.1f}".format(100 * x) for x in s["fall_per_seed"])))
    return "\n".join(out)


S7_HOLDS = (
    "Transplanting the CBF-RL mechanism to joint-space collision avoidance "
    "either fails to internalize (Filter-only degrades from {with_f} with the "
    "filter to {without_f} without) or reduces to reward shaping (Dual reaches "
    "{dual}, within the shaping band), while {channel} channel diagnostics "
    "show {channel_detail}.")

S7_FALSIFIED = (
    "The mechanism can internalize collision constraints given a good learned "
    "barrier; differentiation then rests on joint limits, first-order task "
    "neutrality, the absence of per-constraint barrier engineering, and "
    "freedom from reward-weight tuning.")


def rebuttal_sentence(verdict, summaries, channels):
    if not verdict["holds"]:
        return S7_FALSIFIED
    fo_on = summaries.get(("filter_only", "on"))
    fo_off = summaries.get(("filter_only", "off"))
    du_off = summaries.get(("dual", "off"))
    ch = (channels or {}).get("verdict", "no channel diagnostics available")
    return S7_HOLDS.format(
        with_f=(fmt_pct(*fo_on["collision"][:3]) + "%") if fo_on else "n/a",
        without_f=(fmt_pct(*fo_off["collision"][:3]) + "%") if fo_off else "n/a",
        dual=(fmt_pct(*du_off["collision"][:3]) + "%") if du_off else "n/a",
        channel=("world-model and PPO" if "both" in ch else
                 "world-model" if "world-model" in ch else "PPO"),
        channel_detail=ch)


def main():
    args = parse_args()
    cells = load_cells(args.dir)
    if not cells:
        raise SystemExit("no E2 CSVs found in {}".format(args.dir))

    summaries = {k: cell_summary(v, args.interval) for k, v in cells.items()}
    verdict = apply_verdict(cells, summaries, args.interval)
    channels = summarise_channels(channel_attribution(args.train_dir))

    print(markdown_table(summaries))
    print("\n--- per-seed values (§2) ---")
    print(per_seed_appendix(summaries))

    print("\n--- paired per-seed gap (supporting, not registered) ---")
    paired = {}
    for v in VARIANTS:
        seeds, diffs = paired_gap(cells, v)
        if not diffs:
            continue
        m, lo, hi, sd, n = summarise(diffs, args.interval)
        paired[v] = {"seeds": seeds, "diffs_pp": [100 * d for d in diffs],
                     "mean_pp": 100 * m, "lo_pp": 100 * lo, "hi_pp": 100 * hi}
        print("  {:<12} without - with = {:+.2f} pp [{:+.2f}, {:+.2f}]  "
              "(n={} seeds)".format(v, 100 * m, 100 * lo, 100 * hi, n))
    print("  positive = worse without the filter = did not internalize")

    if channels:
        print("\n--- channel attribution (§3.2) ---")
        print("  world-model loss  filtered {} vs unfiltered {}".format(
            _f(channels["wm_loss_filtered"]), _f(channels["wm_loss_unfiltered"])))
        print("  KL(executed||sampled), filtered runs: {}".format(
            _f(channels["kl_filtered"])))
        print("  -> {}".format(channels["verdict"]))

    print("\n--- verdict (§3.2, intervals = {}) ---".format(args.interval))
    for k, v in verdict["checks"].items():
        print("  {}: {}".format(
            k, "HELD" if v is True else ("FAILED" if v is False else "N/A")))
    print("  primary claim: {}".format(
        "SUPPORTED" if verdict["holds"] else "FALSIFIED"))

    sentence = rebuttal_sentence(verdict, summaries, channels)
    print("\n--- §7 sentence ---")
    print(sentence)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("# Table R2a -- E2 training-time filtering (collision)\n\n")
        fh.write(markdown_table(summaries) + "\n\n")
        fh.write("## Per-seed values (§2)\n\n" + per_seed_appendix(summaries) + "\n\n")
        fh.write("## Verdict (§3.2)\n\n```\n")
        for k, v in verdict["checks"].items():
            fh.write("{}: {}\n".format(
                k, "HELD" if v is True else ("FAILED" if v is False else "N/A")))
        fh.write("```\n\n## §7 sentence\n\n" + sentence + "\n")

    blob = {"summaries": {"{}|{}".format(*k): _jsonable(v)
                          for k, v in summaries.items()},
            "verdict": verdict, "paired_gap": paired,
            "channels": channels, "sentence": sentence,
            "interval_mode": args.interval}
    with open(args.out + ".json", "w") as fh:
        json.dump(blob, fh, indent=2, default=str)
    print("\nwrote {}.md / .json".format(args.out))


def _f(x):
    return "n/a" if x is None else "{:.4f}".format(x)


def _jsonable(s):
    out = dict(s)
    for k in ("collision", "fall", "ret", "vel_err"):
        m, lo, hi, sd, n = out[k]
        out[k] = {"mean": m, "lo": lo, "hi": hi, "std": sd, "n_seeds": n}
    return out


if __name__ == "__main__":
    main()
