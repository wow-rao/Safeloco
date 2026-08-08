"""Offline tests for the E5 metrics, push schedule, calibration and analysis.

Stdlib-only for the metric/analysis core (same guarantee as
test_e1_offline.py); the PushSchedule and calibration subtests need torch and
numpy respectively and skip cleanly without them.  What it covers:

  * the E5 metric helpers on hand-built records (falls/1000 steps,
    time-to-fall censoring, push-window attribution) and that `is_fall`
    behaves identically on E5 records
  * PushSchedule timing/direction math: schedule hits, cardinal cycling,
    unit norms, per-(seed, ordinal) determinism -- all without isaacgym
  * the calibration labeling and lead-time logic on a synthetic trajectory
    where an anticipatory critic must beat the instantaneous margin
  * end-to-end analyze_e5.py on synthetic CSVs + manifests + npz

Run:  python tests/test_e5_offline.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M                          # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        FAILED.append(name)


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def rec(fell, n_steps=500, n_pushes=0, within=0, steps_to_fall=0, **kw):
    r = M.blank_record()
    # blank_record zeroes every field, and a zero term_base_z_rel would be
    # *below* FALL_HEIGHT -- an upright default must be explicit.
    r.update(term_proj_grav_z=(-0.2 if fell else -0.95),
             term_base_z_rel=(0.05 if fell else 0.80),
             term_fall_flag=0, n_steps=n_steps, n_pushes=n_pushes,
             fell=int(fell), fell_within_push_window=within,
             steps_to_fall=steps_to_fall)
    r.update(kw)
    return r


# ------------------------------------------------------------- metrics ------

def test_metrics():
    recs = [rec(True, n_steps=200, steps_to_fall=200),
            rec(False, n_steps=1000),
            rec(True, n_steps=300, steps_to_fall=300,
                n_pushes=2, within=1),
            rec(False, n_steps=1000, n_pushes=3)]

    check("is_fall: E5 records use the shared definition",
          [M.is_fall(r) for r in recs] == [True, False, True, False])

    k, kk = M.falls_per_1000_steps(recs)
    check("falls_per_1000_steps: k=2 over 2.5 kilo-steps",
          k == 2 and approx(kk, 2.5), (k, kk))

    M.attach_fall_fields(recs)
    check("attach_fall_fields: fell_count / ksteps derived",
          recs[0]["fell_count"] == 1 and approx(recs[1]["ksteps"], 1.0))

    vals, n_cens = M.time_to_fall_values(recs, dt=0.02)
    check("time_to_fall: fallen only, censored counted",
          len(vals) == 2 and n_cens == 2
          and approx(vals[0], 4.0) and approx(vals[1], 6.0), (vals, n_cens))

    pk, pn = M.push_attributed_fall_rate(recs)
    check("push_attributed_fall_rate: 1 of 2 pushed episodes",
          pk == 1 and pn == 2, (pk, pn))


# -------------------------------------------------------- push schedule -----

def test_push_schedule():
    try:
        import torch  # noqa: F401
        from safeloco_eval.eval_biped import PushSchedule
    except ImportError as e:
        print("  SKIP  push schedule ({})".format(e))
        return

    ps = PushSchedule(1.0, p0=200, dp=150, dirs="all", seed=7)
    hits = [t for t in range(0, 700) if ps.is_push_step(t)]
    check("push: schedule hits at p0 + k*dp", hits == [200, 350, 500, 650],
          hits)
    check("push: zero magnitude never fires",
          not PushSchedule(0.0).is_push_step(200))

    d0 = ps.directions(0, 6)
    check("push: unit norms",
          bool((d0.norm(dim=-1) - 1.0).abs().max() < 1e-5))
    check("push: per-env phase decorrelates directions",
          not bool((d0[0] == d0[1]).all()))
    # ordinal k shifts the cycle: env 0 at k=1 matches env 1 at k=0.
    d1 = ps.directions(1, 6)
    check("push: ordinal advances the cycle",
          bool((d1[0] == d0[1]).all()))

    card = PushSchedule(1.0, dirs="cardinal", seed=7).directions(0, 8)
    axes = card.abs().max(dim=-1).values
    check("push: cardinal directions are axis-aligned",
          bool((axes - 1.0).abs().max() < 1e-6))

    ra = PushSchedule(1.0, dirs="random", seed=3).directions(2, 5)
    rb = PushSchedule(1.0, dirs="random", seed=3).directions(2, 5)
    rc = PushSchedule(1.0, dirs="random", seed=4).directions(2, 5)
    check("push: random directions deterministic in (seed, ordinal)",
          bool((ra == rb).all()) and not bool((ra == rc).all()))


# ---------------------------------------------------------- calibration -----

def test_calibration():
    try:
        import numpy as np
    except ImportError:
        print("  SKIP  calibration (numpy not available)")
        return
    sys.path.insert(0, os.path.join(ROOT, "experiments", "e5"))
    import analyze_e5 as A

    T, E, dt = 120, 2, 0.02
    vsafe = np.full((T, E), 0.4, dtype=np.float32)
    margin = np.full((T, E), 0.4, dtype=np.float32)
    live = np.ones((T, E), dtype=bool)
    done = np.zeros((T, E), dtype=bool)
    pushed = np.zeros((T, E), dtype=bool)

    # env 0 falls at t=100; the critic goes negative at t=60 (anticipates),
    # the static margin only at t=95 (detects).
    vsafe[60:, 0] = -0.3
    margin[95:, 0] = -0.5
    done[100, 0] = True
    live[101:, 0] = False

    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "run.npz")
        np.savez_compressed(
            npz, c0_vsafe=vsafe, c0_margin=margin, c0_live=live,
            c0_done=done, c0_pushed=pushed)
        run = {"records": [
                   dict(rec(True, n_steps=101, steps_to_fall=101),
                        chunk_idx=0, env_idx=0),
                   dict(rec(False, n_steps=T), chunk_idx=0, env_idx=1)],
               "npz": npz, "manifest": {}}
        # the synthetic env 1 never 'dones'; give it one at the last step
        # so the loop closes its window
        done[T - 1, 1] = True
        np.savez_compressed(
            npz, c0_vsafe=vsafe, c0_margin=margin, c0_live=live,
            c0_done=done, c0_pushed=pushed)
        out = A.calibration_for_run(run, dt, horizons_s=(1.0,))

    auc_v = out["auc"][1.0]["vsafe"]
    auc_m = out["auc"][1.0]["margin"]
    check("calibration: anticipatory critic beats instantaneous margin",
          auc_v > auc_m, (auc_v, auc_m))
    check("calibration: lead times ({:.2f}s vs {:.2f}s)".format(
              out["lead_vsafe_s"], out["lead_margin_s"]),
          approx(out["lead_vsafe_s"], (100 - 60) * dt)
          and approx(out["lead_margin_s"], (100 - 95) * dt))
    check("calibration: positive lead difference",
          out["lead_vsafe_s"] - out["lead_margin_s"] > 0)


# ------------------------------------------------------ analyze_e5 e2e ------

def synth_cell(out_dir, arm, mag, fall_rate, n=60, vx=0.8, terrain="flat"):
    run_id = "{}_vx{:g}_m{:g}".format(arm, vx, mag)
    recs = []
    for i in range(n):
        fell = i < int(round(fall_rate * n))
        r = rec(fell, n_steps=(150 if fell else 900),
                steps_to_fall=(150 if fell else 0),
                n_pushes=(3 if mag > 0 else 0),
                within=int(fell and mag > 0))
        r.update(run_id=run_id, method="E5", variant=arm, terrain=terrain,
                 cmd_vx=vx, push_mag=mag, chunk_idx=i // 30, env_idx=i % 30,
                 vel_err=0.1, mean_fall_margin=0.3, min_fall_margin=-0.2,
                 mean_vsafe=0.2, min_vsafe=-0.1, seed=1)
        r["return"] = 25.0 + (0 if fell else 5.0)
        recs.append(r)
    with open(os.path.join(out_dir, run_id + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=M.EPISODE_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    with open(os.path.join(out_dir, run_id + ".manifest.json"), "w") as f:
        json.dump({"run_id": run_id, "arm": arm, "cmd_vx": vx,
                   "push_mag": mag, "terrain": terrain, "control_dt": 0.02,
                   "vsafe_trained": arm != "pi_nom"}, f)
    return run_id


def test_analyze_e2e():
    td = tempfile.mkdtemp(prefix="e5_offline_")
    try:
        sweep = os.path.join(td, "sweep")
        os.makedirs(sweep)
        # The amended registered-prediction world (arena = stair+push):
        # pi_ours strictly safer than pi_nom on stairs and at the flat
        # boundary, pi_rs between, rates monotone in magnitude.
        for arm, base in (("pi_nom", 0.10), ("pi_rs", 0.06), ("pi_ours", 0.02)):
            for mag in (0.0, 1.0, 3.0):
                synth_cell(sweep, arm, mag, fall_rate=base * (1 + 2 * mag))
        for arm, base in (("pi_nom", 0.15), ("pi_rs", 0.09), ("pi_ours", 0.03)):
            for mag in (0.0, 1.5):
                synth_cell(sweep, arm, mag, terrain="stair",
                           fall_rate=base * (1 + 2 * mag))
        out = os.path.join(td, "R5")
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "experiments", "e5",
                                          "analyze_e5.py"),
             "--dir", sweep, "--out", out, "--no_figure"],
            capture_output=True, text=True)
        check("analyze_e5: exits 0", proc.returncode == 0,
              proc.stderr[-800:] if proc.returncode else "")
        for fname in ("table_r5a.md", "summary.json", "verdict.txt"):
            check("analyze_e5: wrote {}".format(fname),
                  os.path.exists(os.path.join(out, fname)))
        if os.path.exists(os.path.join(out, "verdict.txt")):
            with open(os.path.join(out, "verdict.txt")) as f:
                v = f.read()
            check("analyze_e5: verdict evaluates (i)/(ii)/(iv)/(v)",
                  all(tag in v for tag in ("(i)", "(ii)", "(iv)", "(v)")),
                  v[:200])
            check("analyze_e5: (i) holds on the stair+push cell",
                  "stair+push with separated CIs: 1/1 cells" in v, v)
            check("analyze_e5: (ii) pi_rs between on stair+push",
                  "pi_rs between (stair+push): 1/1 cells" in v, v)
            check("analyze_e5: (v) boundary shift on flat",
                  "boundary shift on flat at mag >= 2.5: 1/1 cells" in v, v)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    print("tests/test_e5_offline.py")
    test_metrics()
    test_push_schedule()
    test_calibration()
    test_analyze_e2e()
    if FAILED:
        print("{} FAILED".format(len(FAILED)))
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
