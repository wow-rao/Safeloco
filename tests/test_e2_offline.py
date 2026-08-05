"""Offline tests for the E2 statistics, verdict rule and analysis pipeline.

Standard library only -- no IsaacGym, no torch, no GPU -- so the part of E2
that turns numbers into a *claim* can be checked before any training run burns
GPU time.  What it covers:

  * the §1 internalization gap and its sign convention
  * the shaping-band test
  * across-seed summarisation, including the thing §2 forbids: pooling
    episodes across seeds
  * the §3.2 verdict rule on four synthetic worlds -- internalization
    failure, reduces-to-shaping, both, and neither (falsified)
  * paired per-seed gaps
  * channel attribution on synthetic training diagnostics
  * end-to-end analyze_e2.py on synthetic CSVs

Run:  python tests/test_e2_offline.py
"""

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M     # noqa: E402
from safeloco_eval import stats as ST      # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "experiments", "e2"))
import analyze_e2 as A                     # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ------------------------------------------------------------- synthesising --

def make_records(n, coll_frac, fall_p, ret_mu, seed, variant, cond,
                 steps=1000, rng=None):
    """n episodes whose pooled collision rate is ~coll_frac."""
    rng = rng or random.Random(hash((seed, variant, cond)) & 0xffff)
    recs = []
    for i in range(n):
        r = M.blank_record()
        r.update(run_id="e2_%s_s%s_tf%s" % (variant, seed, cond),
                 method="E2", variant="%s_tf%s" % (variant, cond),
                 epsilon_or_alpha="", seed=seed, episode_idx=i,
                 terrain="corridor", n_steps=steps,
                 n_collision_steps=int(round(coll_frac * steps)),
                 fell=1 if rng.random() < fall_p else 0,
                 term_timeout=1)
        r["collided"] = 1 if r["n_collision_steps"] > 0 else 0
        r["return"] = ret_mu + rng.gauss(0, 0.5)
        r["vel_err"] = 0.15
        recs.append(r)
    return recs


def write_cell(directory, variant, cond, seed, recs):
    path = os.path.join(directory, "e2_%s_s%s_tf%s.csv" % (variant, seed, cond))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=M.EPISODE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    return path


def build_world(directory, spec, seeds=(1, 2, 3), n=200):
    """spec: {(variant, cond): (collision_fraction, fall_p, return)}"""
    os.makedirs(directory, exist_ok=True)
    for (variant, cond), (cf, fp, ret) in spec.items():
        for si, s in enumerate(seeds):
            # small per-seed jitter so across-seed std is non-degenerate
            jitter = (si - 1) * 0.004
            write_cell(directory, variant, cond, s,
                       make_records(n, max(cf + jitter, 0.0), fp, ret, s,
                                    variant, cond))


# --------------------------------------------------------------- unit tests --

def test_internalization_gap_sign():
    with_f = make_records(50, 0.05, 0.02, 30.0, 1, "filter_only", "on")
    without = make_records(50, 0.13, 0.02, 30.0, 1, "filter_only", "off")
    gap = M.internalization_gap(without, with_f)
    check("gap is positive when the policy is worse without the filter",
          gap > 0, "got %.4f" % gap)
    check("gap magnitude matches the rate difference",
          close(gap, 0.13 - 0.05, tol=1e-6), "got %.6f" % gap)
    check("gap is zero when both conditions match",
          close(M.internalization_gap(with_f, with_f), 0.0, 1e-12))


def test_shaping_band():
    check("13.9% is inside the shaping band", M.in_shaping_band(0.139))
    check("4.4% is outside the shaping band", not M.in_shaping_band(0.044))
    check("band edges are inclusive",
          M.in_shaping_band(0.12) and M.in_shaping_band(0.14))
    check("11.9% is below the band", not M.in_shaping_band(0.119))


def test_summarise_across_seeds():
    vals = [0.10, 0.12, 0.14]
    m, lo, hi, sd, n = A.summarise(vals, "std")
    check("mean across seeds", close(m, 0.12, 1e-12), "got %r" % m)
    check("std across seeds is the sample std",
          close(sd, ST.std(vals), 1e-12))
    check("std interval is mean +/- 1 std",
          close(lo, m - sd, 1e-12) and close(hi, m + sd, 1e-12))

    mt, lot, hit, sdt, nt = A.summarise(vals, "t95")
    half = 4.303 * (sd / (3 ** 0.5))
    check("t95 interval at n=3 uses t=4.303",
          close(lot, m - half, 1e-9) and close(hit, m + half, 1e-9))
    check("t95 interval is wider than the std interval", hit - lot > hi - lo)
    check("single seed degenerates without crashing",
          A.summarise([0.11], "t95")[1] == 0.11)
    check("empty cell yields n=0", A.summarise([], "t95")[4] == 0)


def test_no_pooling_across_seeds():
    """§2: the cell summary must be built from per-seed means, not pooled.

    Seeds with very different episode counts expose the difference: the
    pooled rate is dominated by the big seed, the per-seed mean is not.
    """
    by_seed = {
        "1": make_records(900, 0.20, 0.0, 30.0, 1, "filter_only", "off"),
        "2": make_records(50, 0.02, 0.0, 30.0, 2, "filter_only", "off"),
        "3": make_records(50, 0.02, 0.0, 30.0, 3, "filter_only", "off"),
    }
    s = A.cell_summary(by_seed, "std")
    per_seed_mean = (0.20 + 0.02 + 0.02) / 3.0
    pooled = (900 * 0.20 + 50 * 0.02 + 50 * 0.02) / 1000.0
    check("cell mean is the mean of per-seed rates",
          close(s["collision"][0], per_seed_mean, 1e-6),
          "got %.6f, per-seed %.6f, pooled would be %.6f"
          % (s["collision"][0], per_seed_mean, pooled))
    check("cell mean is NOT the pooled rate",
          not close(s["collision"][0], pooled, 1e-3))


def test_paired_gap():
    cells = {
        ("filter_only", "on"): {
            "1": make_records(100, 0.05, 0.0, 30.0, 1, "filter_only", "on"),
            "2": make_records(100, 0.06, 0.0, 30.0, 2, "filter_only", "on")},
        ("filter_only", "off"): {
            "1": make_records(100, 0.15, 0.0, 30.0, 1, "filter_only", "off"),
            "2": make_records(100, 0.16, 0.0, 30.0, 2, "filter_only", "off")},
    }
    seeds, diffs = A.paired_gap(cells, "filter_only")
    check("paired gap matches seeds present in both conditions",
          seeds == ["1", "2"], "got %r" % seeds)
    check("each paired difference is without - with",
          all(close(d, 0.10, 1e-6) for d in diffs), "got %r" % diffs)
    check("missing condition yields no pairs",
          A.paired_gap({("dual", "on"): {"1": []}}, "dual")[1] == [])


# ------------------------------------------------------------ verdict tests --

WORLDS = {
    # filter_only collapses without the filter -> internalization failure.
    # dual sits well below the shaping band -> reduces_to_shaping is False.
    "internalization_only": {
        ("reward_only", "on"): (0.130, 0.03, 30.0),
        ("reward_only", "off"): (0.130, 0.03, 30.0),
        ("filter_only", "on"): (0.040, 0.03, 30.0),
        ("filter_only", "off"): (0.190, 0.05, 29.0),
        ("dual", "on"): (0.040, 0.03, 30.0),
        ("dual", "off"): (0.050, 0.03, 30.0),
    },
    # filter_only holds up without the filter, dual lands in the band.
    "shaping_only": {
        ("reward_only", "on"): (0.130, 0.03, 30.0),
        ("reward_only", "off"): (0.130, 0.03, 30.0),
        ("filter_only", "on"): (0.050, 0.03, 30.0),
        ("filter_only", "off"): (0.052, 0.03, 30.0),
        ("dual", "on"): (0.050, 0.03, 30.0),
        ("dual", "off"): (0.130, 0.03, 30.0),
    },
    "both": {
        ("reward_only", "on"): (0.130, 0.03, 30.0),
        ("reward_only", "off"): (0.130, 0.03, 30.0),
        ("filter_only", "on"): (0.040, 0.03, 30.0),
        ("filter_only", "off"): (0.200, 0.06, 28.5),
        ("dual", "on"): (0.045, 0.03, 30.0),
        ("dual", "off"): (0.132, 0.03, 30.0),
    },
    # everything internalizes and nothing reduces to shaping -> falsified.
    "falsified": {
        ("reward_only", "on"): (0.130, 0.03, 30.0),
        ("reward_only", "off"): (0.130, 0.03, 30.0),
        ("filter_only", "on"): (0.045, 0.03, 30.0),
        ("filter_only", "off"): (0.047, 0.03, 30.0),
        ("dual", "on"): (0.043, 0.03, 30.0),
        ("dual", "off"): (0.046, 0.03, 30.0),
    },
}

EXPECTED = {
    "internalization_only": (True, False, True),
    "shaping_only": (False, True, True),
    "both": (True, True, True),
    "falsified": (False, False, False),
}


def test_verdict_worlds(tmp):
    for world, spec in WORLDS.items():
        d = os.path.join(tmp, "world_" + world)
        build_world(d, spec)
        cells = A.load_cells(d)
        summaries = {k: A.cell_summary(v, "std") for k, v in cells.items()}
        v = A.apply_verdict(cells, summaries, "std")
        want_int, want_shape, want_holds = EXPECTED[world]
        got = (v["checks"]["internalization_failure"],
               v["checks"]["reduces_to_shaping"], v["holds"])
        check("world '%s' -> internalization=%s" % (world, want_int),
              got[0] is want_int, "got %r" % (got[0],))
        check("world '%s' -> reduces_to_shaping=%s" % (world, want_shape),
              got[1] is want_shape, "got %r" % (got[1],))
        check("world '%s' -> holds=%s" % (world, want_holds),
              got[2] is want_holds, "got %r" % (got[2],))


def test_verdict_missing_rows(tmp):
    d = os.path.join(tmp, "world_partial")
    build_world(d, {("reward_only", "on"): (0.13, 0.03, 30.0),
                    ("reward_only", "off"): (0.13, 0.03, 30.0)})
    cells = A.load_cells(d)
    summaries = {k: A.cell_summary(v, "std") for k, v in cells.items()}
    v = A.apply_verdict(cells, summaries, "std")
    check("absent filter_only reports N/A rather than a verdict",
          v["checks"]["internalization_failure"] is None)
    check("absent dual reports N/A rather than a verdict",
          v["checks"]["reduces_to_shaping"] is None)
    check("no check true -> does not hold", v["holds"] is False)


def test_falsified_sentence(tmp):
    d = os.path.join(tmp, "world_falsified")
    cells = A.load_cells(d)
    summaries = {k: A.cell_summary(v, "std") for k, v in cells.items()}
    v = A.apply_verdict(cells, summaries, "std")
    s = A.rebuttal_sentence(v, summaries, None)
    check("falsified world selects the §7 falsified sentence",
          s.startswith("The mechanism can internalize"), s[:60])


def test_channel_attribution(tmp):
    """Synthetic train_diag.csv rows exercise each channel branch."""
    def write_diag(run, wm, kl, frac):
        d = os.path.join(tmp, "train", run)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "train_diag.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "iteration", "mean_return", "frac_filtered",
                "mean_kl_exec_sampled", "wm_model_loss"])
            w.writeheader()
            for i in range(10):
                w.writerow({"iteration": i, "mean_return": 30.0,
                            "frac_filtered": frac, "mean_kl_exec_sampled": kl,
                            "wm_model_loss": wm})

    # world-model channel: elevated model loss in filtered runs, tiny KL
    write_diag("e2_reward_only_s1", 1.00, 0.0, 0.0)
    write_diag("e2_filter_only_s1", 1.50, 0.001, 0.3)
    ch = A.summarise_channels(A.channel_attribution(os.path.join(tmp, "train")))
    check("elevated model loss -> world-model channel",
          "world-model channel" in ch["verdict"], ch["verdict"])

    # PPO channel alone: flat model loss, large KL
    write_diag("e2_filter_only_s1", 1.00, 0.5, 0.3)
    ch = A.summarise_channels(A.channel_attribution(os.path.join(tmp, "train")))
    check("flat model loss + large KL -> PPO channel alone",
          "PPO channel alone" in ch["verdict"], ch["verdict"])

    # clean negative: flat everything
    write_diag("e2_filter_only_s1", 1.00, 0.0, 0.3)
    ch = A.summarise_channels(A.channel_attribution(os.path.join(tmp, "train")))
    check("flat model loss + small KL -> clean negative",
          "neither channel" in ch["verdict"], ch["verdict"])

    check("no training dir -> no channel block",
          A.channel_attribution(os.path.join(tmp, "nope")) is None)


def test_end_to_end(tmp):
    d = os.path.join(tmp, "world_both")
    out = os.path.join(tmp, "R2a")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "e2", "analyze_e2.py"),
         "--dir", d, "--train_dir", os.path.join(tmp, "train"),
         "--out", out, "--interval", "std"],
        capture_output=True, text=True)
    check("analyze_e2.py exits cleanly", r.returncode == 0,
          (r.stderr or "")[-400:])
    check("writes the markdown table", os.path.exists(out + ".md"))
    check("writes the json blob", os.path.exists(out + ".json"))
    if os.path.exists(out + ".json"):
        blob = json.load(open(out + ".json"))
        check("json records the verdict", blob["verdict"]["holds"] is True)
        check("json records per-seed values",
              len(blob["summaries"]["filter_only|off"]["collision_per_seed"]) == 3)
        check("json records the interval mode",
              blob["interval_mode"] == "std")
    check("stdout carries the §7 sentence",
          "either fails to internalize" in r.stdout, r.stdout[-200:])
    check("stdout labels the paired gap as unregistered",
          "not registered" in r.stdout)


def main():
    tmp = tempfile.mkdtemp(prefix="e2_offline_")
    try:
        print("[§1 metrics]")
        test_internalization_gap_sign()
        test_shaping_band()
        print("\n[§2 across-seed statistics]")
        test_summarise_across_seeds()
        test_no_pooling_across_seeds()
        test_paired_gap()
        print("\n[§3.2 verdict rule]")
        test_verdict_worlds(tmp)
        test_verdict_missing_rows(tmp)
        test_falsified_sentence(tmp)
        print("\n[channel attribution]")
        test_channel_attribution(tmp)
        print("\n[end to end]")
        test_end_to_end(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 70)
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
