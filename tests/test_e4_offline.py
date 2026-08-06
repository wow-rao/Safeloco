"""Offline tests for E4's gate and the command-space CBF projection.

The gate half is standard-library only and always runs.  The projection half
needs torch and skips cleanly without it, so this file never breaks the
no-dependency guarantee of the other offline suites.

What it covers:

  * the §3.4 three-outcome gate on synthetic worlds, including the
    outcome-3 case that changes what gets run next
  * that dominance is scored on three axes and lateral velocity is not
  * §1's infeasibility rate and its activation denominator
  * the §5 sanity checks: zero activation unfiltered, activation monotone
    in alpha
  * end-to-end analyze_e4.py on synthetic CSVs
  * the CBF projection itself: inactive when already safe, constraint
    satisfied after projection, multi-obstacle convergence, infeasibility
    detection, and that yaw is never touched

Run:  python tests/test_e4_offline.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M     # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "experiments", "e4"))
import analyze_e4 as A                     # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


# ------------------------------------------------------------ synthesising --

def make_records(n, coll_frac, fall_p, ret, vel_err, lat_vel, alpha,
                 activation_frac=0.0, infeasible_frac=0.0, steps=1000):
    recs = []
    for i in range(n):
        r = M.blank_record()
        act = int(round(activation_frac * steps))
        r.update(run_id="e4_a%s" % alpha, method="E4",
                 variant=("cbf_command" if alpha is not None else "none"),
                 epsilon_or_alpha=("" if alpha is None else alpha),
                 seed=1, episode_idx=i, terrain="corridor", n_steps=steps,
                 n_collision_steps=int(round(coll_frac * steps)),
                 fell=0, term_timeout=1,
                 activation_steps=act, trigger_steps=act,
                 infeasible_steps=int(round(infeasible_frac * act)))
        r["collided"] = 1 if r["n_collision_steps"] > 0 else 0
        r["return"] = ret
        r["vel_err"] = vel_err
        r["mean_lat_vel"] = lat_vel
        # fall is recomputed from the terminal pose, so drive it that way
        r["term_proj_grav_z"] = -0.2 if i < int(fall_p * n) else -0.99
        r["term_base_z_rel"] = 0.30
        recs.append(r)
    return recs


def write_world(directory, spec):
    """spec: list of (alpha|None, coll, fall_p, ret, vel_err, lat, act, infeas)"""
    os.makedirs(directory, exist_ok=True)
    for alpha, c, f, ret, ve, lat, act, inf in spec:
        recs = make_records(120, c, f, ret, ve, lat, alpha, act, inf)
        name = "unfiltered" if alpha is None else "cbf_alpha%g" % alpha
        with open(os.path.join(directory, "pi_nom_%s.csv" % name),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=M.EPISODE_FIELDS,
                               extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)


# ------------------------------------------------------------- gate tests --

def test_outcome_1(tmp):
    """Every row worse than ours on all three scored axes."""
    d = os.path.join(tmp, "w1")
    write_world(d, [
        (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
        (0.5, 0.100, 0.03, 28.0, 0.20, 0.06, 0.30, 0.0),
        (2.0, 0.110, 0.02, 29.0, 0.16, 0.05, 0.10, 0.0),
    ])
    rows, unf = A.load_rows(d)
    g = A.apply_gate(rows)
    check("all rows worse on three axes -> outcome 1", g["outcome"] == 1,
          "got %r (%r)" % (g["outcome"], g["per_row"]))
    check("unfiltered row is separated out", unf is not None)
    check("rows are sorted by alpha", [r["alpha"] for r in rows] == [0.5, 2.0])


def test_outcome_3(tmp):
    """One row matches ours on all three -- the branch that changes plans."""
    d = os.path.join(tmp, "w3")
    write_world(d, [
        (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
        (0.5, 0.030, 0.02, 34.5, 0.070, 0.09, 0.40, 0.0),
        (2.0, 0.110, 0.02, 29.0, 0.160, 0.05, 0.10, 0.0),
    ])
    rows, _ = A.load_rows(d)
    g = A.apply_gate(rows)
    check("a dominating row -> outcome 3", g["outcome"] == 3,
          "got %r (%r)" % (g["outcome"], g["per_row"]))
    check("the dominating row is identified",
          g["per_row"]["alpha=0.5"] == "matches_or_dominates")


def test_outcome_2(tmp):
    """Competitive on collision, worse elsewhere."""
    d = os.path.join(tmp, "w2")
    write_world(d, [
        (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
        (0.5, 0.035, 0.05, 26.0, 0.30, 0.12, 0.55, 0.0),
    ])
    rows, _ = A.load_rows(d)
    g = A.apply_gate(rows)
    check("better collision but worse return/vel-err -> outcome 2",
          g["outcome"] == 2, "got %r (%r)" % (g["outcome"], g["per_row"]))
    check("that row is classified mixed, not dominating",
          g["per_row"]["alpha=0.5"] == "mixed")


def test_lat_vel_is_not_scored(tmp):
    """Two worlds differing only in lateral velocity must gate identically."""
    outcomes = []
    for i, lat in enumerate((0.02, 0.30)):
        d = os.path.join(tmp, "wlat%d" % i)
        write_world(d, [
            (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
            (1.0, 0.030, 0.02, 34.5, 0.070, lat, 0.4, 0.0),
        ])
        rows, _ = A.load_rows(d)
        outcomes.append(A.apply_gate(rows)["outcome"])
    check("lateral velocity does not change the gate",
          outcomes[0] == outcomes[1], "got %r" % (outcomes,))
    check("lateral velocity is declared unscored",
          "mean_lat_vel" in A.apply_gate(
              A.load_rows(os.path.join(tmp, "wlat0"))[0])["reported_not_scored"])


def test_infeasibility_denominator():
    """§1: infeasibility is per *active* step, not per timestep."""
    recs = make_records(10, 0.1, 0.0, 30.0, 0.15, 0.05, 1.0,
                        activation_frac=0.5, infeasible_frac=0.2)
    k, n = M.infeasibility_rate(recs)
    check("infeasibility denominator is trigger steps",
          n == sum(int(r["trigger_steps"]) for r in recs),
          "got n=%r" % n)
    check("infeasibility rate is the fraction of triggered steps",
          abs(k / n - 0.2) < 1e-6, "got %.4f" % (k / n))
    idle = make_records(5, 0.1, 0.0, 30.0, 0.15, 0.05, None,
                        activation_frac=0.0)
    ik, inn = M.infeasibility_rate(idle)
    check("no activation -> zero denominator, no crash", (ik, inn) == (0, 0))


def test_infeasible_cannot_exceed_trigger():
    """Regression: infeasible_steps > trigger_steps is not a rate.

    The command filter used to measure feasibility *before* clamping, so a
    step could be counted infeasible while the clamp returned the command to
    nominal and left it unactivated.  wilson_ci then took sqrt of a negative
    and died with a bare math domain error.
    """
    import safeloco_eval.stats as _ST
    try:
        _ST.wilson_ci(7, 3)
    except ValueError as exc:
        check("wilson_ci names the accounting bug rather than failing in sqrt",
              "exceed" in str(exc), str(exc))
    else:
        check("wilson_ci names the accounting bug rather than failing in sqrt",
              False, "-> accepted k > n")
    check("wilson_ci still handles an empty denominator",
          _ST.wilson_ci(0, 0)[1:] == (0.0, 1.0))


def test_sanity_checks(tmp):
    d = os.path.join(tmp, "wsan")
    write_world(d, [
        (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
        (0.5, 0.10, 0.02, 29.0, 0.16, 0.06, 0.40, 0.0),
        (2.0, 0.11, 0.02, 29.5, 0.15, 0.05, 0.10, 0.0),
    ])
    rows, unf = A.load_rows(d)
    res = dict((n, ok) for n, ok, _ in A.sanity_checks(rows, unf))
    check("zero activation on the unfiltered row passes",
          res.get("unfiltered row has zero activation") is True)
    check("activation falling as alpha grows passes",
          res.get("activation non-increasing as alpha grows") is True)

    d2 = os.path.join(tmp, "wsan_bad")
    write_world(d2, [
        (None, 0.118, 0.02, 29.9, 0.148, 0.049, 0.0, 0.0),
        (0.5, 0.10, 0.02, 29.0, 0.16, 0.06, 0.05, 0.0),
        (2.0, 0.11, 0.02, 29.5, 0.15, 0.05, 0.40, 0.0),
    ])
    rows2, unf2 = A.load_rows(d2)
    res2 = dict((n, ok) for n, ok, _ in A.sanity_checks(rows2, unf2))
    check("activation rising with alpha is flagged",
          res2.get("activation non-increasing as alpha grows") is False)


def test_end_to_end(tmp):
    d = os.path.join(tmp, "w1")
    out = os.path.join(tmp, "R1c")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "e4", "analyze_e4.py"),
         "--dir", d, "--out", out], capture_output=True, text=True)
    check("analyze_e4.py exits cleanly", r.returncode == 0,
          (r.stderr or "")[-400:])
    check("writes the markdown table", os.path.exists(out + ".md"))
    if os.path.exists(out + ".json"):
        blob = json.load(open(out + ".json"))
        check("json records the outcome", blob["gate"]["outcome"] == 1)
        check("json records the reference point",
              abs(blob["gate"]["reference"]["collision"] - 0.044) < 1e-9)
    check("stdout carries a §7 sentence", "Command-space filtering" in r.stdout)
    check("table includes the ours reference row", "ours (reference)" in r.stdout)


# ------------------------------------------------- CBF projection (torch) --

def test_projection():
    try:
        import torch
    except ImportError:
        print("  SKIP projection tests -- torch not available")
        return
    from safeloco_eval.command_filters import CBFCommandFilter

    class FakeEnv(object):
        """Minimal stand-in: one env, obstacles on the +x axis, yaw = 0."""

        def __init__(self, obstacles, r=0.3, cmd=(0.6, 0.0)):
            n = len(obstacles)
            self.root_states = torch.zeros(1, 13)
            self.root_states[0, 6] = 1.0            # unit quaternion, yaw = 0
            self.obstacle_positions = torch.zeros(1, max(n, 1), 3)
            for i, (x, y) in enumerate(obstacles):
                self.obstacle_positions[0, i, 0] = x
                self.obstacle_positions[0, i, 1] = y
            self.obstacle_radii = torch.full((1,), r)
            self.obstacle_mask = torch.zeros(1, max(n, 1), dtype=torch.bool)
            self.obstacle_mask[0, :n] = True
            self.has_obstacles = torch.ones(1, dtype=torch.bool)
            self.commands = torch.tensor([[cmd[0], cmd[1], 0.0, 0.0]])

    def run(env, alpha, **kw):
        f = CBFCommandFilter(alpha=alpha, **kw)
        return f.apply(env.commands, env)

    # far away: constraint slack, filter inert
    env = FakeEnv([(8.0, 0.0)])
    out, info = run(env, alpha=1.0)
    check("distant obstacle leaves the command untouched",
          bool(torch.allclose(out[:, :2], env.commands[:, :2])) and
          not bool(info["activated"][0]))

    # close ahead: must slow or steer
    env = FakeEnv([(1.2, 0.0)])
    out, info = run(env, alpha=0.5)
    check("obstacle dead ahead activates the filter", bool(info["activated"][0]))
    a_dot_u = float(out[0, 0])          # yaw=0, obstacle on +x -> a = (1, 0)
    h = 1.2 - 2 * 0.3 - 0.35
    check("projected command satisfies a.u <= alpha*h",
          a_dot_u <= 0.5 * h + 1e-5,
          "a.u=%.4f  alpha*h=%.4f" % (a_dot_u, 0.5 * h))

    # yaw must never be touched
    env = FakeEnv([(1.2, 0.0)])
    env.commands[0, 2] = 0.7
    out, _ = run(env, alpha=0.5)
    # float32 round-trip, so compare against the stored value not the literal
    check("yaw rate is left alone (relative degree 2)",
          abs(float(out[0, 2]) - float(env.commands[0, 2])) < 1e-9,
          "got %r" % float(out[0, 2]))

    # smaller alpha is more conservative
    env = FakeEnv([(1.2, 0.0)])
    lo, _ = run(env, alpha=0.25)
    hi, _ = run(env, alpha=4.0)
    check("smaller alpha yields a more conservative command",
          float(lo[0, 0]) <= float(hi[0, 0]) + 1e-9,
          "alpha=0.25 -> %.4f, alpha=4 -> %.4f" % (float(lo[0, 0]), float(hi[0, 0])))

    # two obstacles: both constraints must hold after cycling
    env = FakeEnv([(1.2, 0.6), (1.2, -0.6)])
    out, info = run(env, alpha=0.5, max_passes=30)
    f = CBFCommandFilter(alpha=0.5)
    a, b, mask, _ = f._constraints(env)
    resid = (a[0] * out[0, :2]).sum(-1) - b[0]
    check("both constraints satisfied after cyclic projection",
          bool((resid[mask[0]] <= 1e-4).all()),
          "resid=%r" % resid.tolist())

    # a single pass cannot satisfy a conflicting pair -> infeasible reported
    env = FakeEnv([(0.62, 0.0), (-0.62, 0.0)])
    _, info = run(env, alpha=0.01, max_passes=1)
    check("unsatisfied constraints are reported as infeasible",
          bool(info["infeasible"][0]))

    # no obstacles at all
    env = FakeEnv([])
    env.has_obstacles = torch.zeros(1, dtype=torch.bool)
    out, info = run(env, alpha=1.0)
    check("no obstacles -> command untouched, not activated",
          bool(torch.allclose(out, env.commands)) and
          not bool(info["activated"][0]))


def main():
    tmp = tempfile.mkdtemp(prefix="e4_offline_")
    try:
        print("[§3.4 gate]")
        test_outcome_1(tmp)
        test_outcome_3(tmp)
        test_outcome_2(tmp)
        test_lat_vel_is_not_scored(tmp)
        print("\n[§1 metrics]")
        test_infeasibility_denominator()
        test_infeasible_cannot_exceed_trigger()
        print("\n[§5 sanity checks]")
        test_sanity_checks(tmp)
        print("\n[end to end]")
        test_end_to_end(tmp)
        print("\n[command-space CBF projection]")
        test_projection()
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
