"""Offline tests for E4-Y -- the steering command filter and its gate.

The gate half is standard-library only and always runs.  The projection half
needs torch and skips cleanly without it, so this file never breaks the
no-dependency guarantee of the other offline suites.

What it covers:

  * the four-way gate, including `trivial_stop` and the `not_decidable` case
    that means nothing was learned
  * that the gate scores the **joint** frontier, never contact alone
  * that reference rows come from `safeloco_eval.reference` and that a
    fallback reference is announced rather than quietly used
  * the steer/brake decomposition, which is the evidence for prediction (iv)
  * end-to-end `analyze_e4y.py` on synthetic CSVs
  * the look-ahead projection itself: that yaw enters at first order, that the
    filter turns *away* from an obstacle, that the weighting really does
    prefer steering to braking, and -- the registered prediction (i)
    demonstrated without a GPU -- that a geometry exists where the
    yaw-disabled arm has no legal option and the steering arm does

Run:  python tests/test_e4y_offline.py
"""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M          # noqa: E402
from safeloco_eval import reference as R        # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "experiments", "e4y"))
import analyze_e4y as A                         # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


# ------------------------------------------------------------ synthesising --

def make_records(n, coll_frac, fall_p, ret, vel_err, alpha, yaw,
                 activation_frac=0.0, infeasible_frac=0.0, steps=1000,
                 fwd_vel=0.58, steer_share=0.0, replica=False):
    """One synthetic row's worth of per-episode records.

    `steer_share` is the fraction of correction magnitude spent on yaw; it is
    written into the two sum fields the decomposition reads, so the test
    exercises the real metric rather than a stub.
    """
    recs = []
    total_corr = 10.0
    for i in range(n):
        r = M.blank_record()
        act = int(round(activation_frac * steps))
        r.update(run_id="e4y_a%s_%s" % (alpha, "yaw" if yaw else "noyaw"),
                 method="E4Y",
                 variant=("cbf_unicycle_%s%s" % (
                     "yaw" if yaw else "noyaw",
                     "_L0_braking_replica" if replica else "")
                     if alpha is not None else "none"),
                 epsilon_or_alpha=("" if alpha is None else alpha),
                 seed=1, episode_idx=i, terrain="corridor", n_steps=steps,
                 n_collision_steps=int(round(coll_frac * steps)),
                 fell=0, term_timeout=1,
                 activation_steps=act, trigger_steps=act,
                 infeasible_steps=int(round(infeasible_frac * act)),
                 sum_dnorm_omega=total_corr * steer_share,
                 sum_dnorm_v=total_corr * (1.0 - steer_share))
        r["collided"] = 1 if r["n_collision_steps"] > 0 else 0
        r["return"] = ret
        r["vel_err"] = vel_err
        r["mean_fwd_vel"] = fwd_vel
        r["term_proj_grav_z"] = -0.2 if i < int(fall_p * n) else -0.99
        r["term_base_z_rel"] = 0.30
        recs.append(r)
    return recs


def write_world(directory, spec, n=60, replica=False):
    """spec entries: (alpha|None, yaw, coll, fall_p, ret, fwd, act, infeas, steer)"""
    os.makedirs(directory, exist_ok=True)
    for alpha, yaw, c, f, ret, fwd, act, inf, steer in spec:
        recs = make_records(n, c, f, ret, 0.15, alpha, yaw,
                            activation_frac=act, infeasible_frac=inf,
                            fwd_vel=fwd, steer_share=steer,
                            replica=replica and alpha is not None)
        arm = "yaw" if yaw else "noyaw"
        name = ("unfiltered" if alpha is None
                else "uni_a%g_L%g_%s" % (alpha, 0.0 if replica else 0.35, arm))
        with open(os.path.join(directory, "pi_nom_%s.csv" % name),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=M.EPISODE_FIELDS,
                               extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)


def measured_ref(collision=0.044, ret=30.0, fwd=0.58, distance=11.0):
    """A same-harness reference, i.e. one that carries speed and distance."""
    return {"measured": True,
            "ours": {"collision": collision, "ret": ret, "fwd_vel": fwd,
                     "distance": distance, "n_episodes": 1000}}


FAST = dict(n_boot=64)


# ------------------------------------------------------------- gate tests --

def test_dominated(tmp):
    """Every row worse than the reference -- the OCR/ABS row closes."""
    d = os.path.join(tmp, "dom")
    write_world(d, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        (1.0, True, 0.090, 0.03, 28.0, 0.500, 0.35, 0.10, 0.4),
        (2.0, True, 0.100, 0.02, 29.0, 0.540, 0.20, 0.08, 0.4),
        (1.0, False, 0.121, 0.03, 27.0, 0.300, 0.40, 0.85, 0.0),
    ])
    rows, unf = A.load_rows(d, **FAST)
    g = A.apply_gate(rows, unf, measured_ref())
    check("all rows above the reference contact -> dominated",
          g["outcome"] == "dominated", "got %r (%r)" % (g["outcome"], g["per_row"]))
    check("the unfiltered row is separated out", unf is not None)
    check("yaw rows sort before yaw-disabled rows",
          [r["yaw"] for r in rows] == [True, True, False],
          "got %r" % [(r["label"], r["yaw"]) for r in rows])


def test_matches_needs_the_whole_frontier(tmp):
    """Beating contact is not enough; speed and distance are scored too."""
    d = os.path.join(tmp, "front")
    write_world(d, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        # contact well under the reference, but crawling
        (1.0, True, 0.020, 0.02, 31.0, 0.360, 0.50, 0.05, 0.5),
    ])
    rows, unf = A.load_rows(d, **FAST)
    g = A.apply_gate(rows, unf, measured_ref())
    check("better contact but short of speed -> competitive, not matching",
          g["outcome"] == "competitive", "got %r (%r)" % (g["outcome"], g["per_row"]))

    d2 = os.path.join(tmp, "front2")
    write_world(d2, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        (1.0, True, 0.020, 0.02, 31.0, 0.570, 0.50, 0.05, 0.5),
    ])
    rows2, unf2 = A.load_rows(d2, **FAST)
    g2 = A.apply_gate(rows2, unf2, measured_ref())
    check("better contact at comparable speed -> matches_or_dominates",
          g2["outcome"] == "matches_or_dominates",
          "got %r (%r)" % (g2["outcome"], g2["per_row"]))
    check("the falsifying row is named",
          g2["per_row"]["pi_nom_uni_a1_L0.35_yaw"] == "matches_or_dominates")


def test_trivial_stop(tmp):
    """A parked robot is not a frontier point, however good its contact rate.

    The first real command-space sweep produced exactly this -- contact 0.0%,
    return *up*, velocity error *down* -- from a filter that had ratcheted the
    command to zero.  Without this guard it reads as a clean win.
    """
    d = os.path.join(tmp, "stop")
    write_world(d, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        (1.0, True, 0.000, 0.00, 38.6, 0.030, 0.30, 0.85, 0.1),
    ])
    rows, unf = A.load_rows(d, **FAST)
    g = A.apply_gate(rows, unf, measured_ref())
    check("a stopped robot is classified trivial_stop",
          g["per_row"]["pi_nom_uni_a1_L0.35_yaw"] == "trivial_stop",
          "got %r" % g["per_row"])
    check("every row stopped -> the gate is not decidable",
          g["outcome"] == "not_decidable", "got %r" % g["outcome"])
    # Two independent defences, and it is worth knowing which one is load
    # bearing.  Against a *measured* reference the speed axis catches a parked
    # robot on its own.  Against the Table 2 fallback, which has no speed or
    # distance column, the trivial_stop guard is the only thing standing
    # between a filter that ratcheted the command to zero and a clean "beats
    # the proposed method" verdict.
    check("a measured reference catches it on the speed axis alone",
          A.classify(rows[0], measured_ref(), None) == "competitive",
          "got %r" % A.classify(rows[0], measured_ref(), None))
    fallback = R.load_reference(os.path.join(tmp, "absent.json"))
    check("against the speed-less fallback it would have dominated",
          A.classify(rows[0], fallback, None) == "matches_or_dominates",
          "got %r" % A.classify(rows[0], fallback, None))
    check("...and the guard stops it doing so",
          A.classify(rows[0], fallback, unf) == "trivial_stop")


def test_predictions_are_computed(tmp):
    """(i) feasibility and (iv) steer share are read off the yaw/no-yaw split."""
    d = os.path.join(tmp, "pred")
    write_world(d, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        (2.0, True, 0.090, 0.02, 29.0, 0.540, 0.30, 0.12, 0.45),
        (2.0, False, 0.121, 0.03, 27.0, 0.300, 0.30, 0.90, 0.00),
    ])
    rows, unf = A.load_rows(d, **FAST)
    g = A.apply_gate(rows, unf, measured_ref())
    check("prediction (i): steering lowers the no-legal-option rate",
          g["infeasibility_yaw"] < g["infeasibility_noyaw"] - 0.05,
          "yaw=%.3f noyaw=%.3f" % (g["infeasibility_yaw"],
                                   g["infeasibility_noyaw"]))
    check("prediction (ii): the yaw arm has lower contact",
          g["contact_yaw"] < g["contact_noyaw"],
          "yaw=%.4f noyaw=%.4f" % (g["contact_yaw"], g["contact_noyaw"]))
    check("prediction (iv): the steer share is measured from the sum fields",
          abs(g["steer_share_yaw"] - 0.45) < 1e-6,
          "got %r" % g["steer_share_yaw"])
    check("the yaw-disabled arm reports no steering",
          math.isnan(M.steer_brake_split(
              make_records(3, 0.1, 0.0, 29.0, 0.15, 2.0, False,
                           activation_frac=0.0, steer_share=0.0))[0]) is False)


def test_replica_is_kept_out_of_the_yaw_contrast(tmp):
    """A base-centred row is the old study, not this study's control arm.

    Look-ahead 0 collapses the yaw term to zero identically, so such a row is
    the completed braking-only filter re-run under the corrected projection.
    It belongs in the table -- it is the corrected version of a number already
    reported -- but pooling it into the yaw / yaw-disabled contrast would
    compare two different barriers and call the difference "yaw".
    """
    d = os.path.join(tmp, "replica")
    write_world(d, [
        (None, True, 0.118, 0.02, 29.9, 0.580, 0.00, 0.00, 0.0),
        (2.0, True, 0.090, 0.02, 29.0, 0.540, 0.30, 0.12, 0.45),
        (2.0, False, 0.121, 0.03, 27.0, 0.300, 0.30, 0.90, 0.00),
    ])
    write_world(d, [(2.0, False, 0.140, 0.04, 25.0, 0.250, 0.30, 0.55, 0.00)],
                replica=True)
    rows, unf = A.load_rows(d, **FAST)
    reps = [r for r in rows if r["replica"]]
    check("the base-centred row is flagged as a replica", len(reps) == 1,
          "got %r" % [(r["label"], r["replica"]) for r in rows])
    check("replicas sort to the end of the table", rows[-1]["replica"] is True)

    g = A.apply_gate(rows, unf, measured_ref())
    check("the replica does not enter the yaw-disabled feasibility mean",
          abs(g["infeasibility_noyaw"] - 0.90) < 1e-6,
          "got %r (0.90 = the control arm alone)" % g["infeasibility_noyaw"])
    check("the replica does not enter the yaw-disabled contact mean",
          abs(g["contact_noyaw"] - 0.121) < 1e-3,
          "got %r" % g["contact_noyaw"])
    check("the replica is still gated and shown",
          reps[0]["label"] in g["per_row"])
    check("the table flags what L=0 means",
          "base centre" in A.table(rows, unf, measured_ref()))


def test_steer_brake_split_is_a_share():
    recs = make_records(5, 0.1, 0.0, 29.0, 0.15, 2.0, True,
                        activation_frac=0.3, steer_share=0.3)
    steer, brake = M.steer_brake_split(recs)
    check("steer and brake shares sum to one",
          abs(steer + brake - 1.0) < 1e-9, "%r %r" % (steer, brake))
    check("the steer share is the omega share of correction magnitude",
          abs(steer - 0.3) < 1e-9, "got %r" % steer)
    idle = M.blank_record()
    check("nothing corrected -> nan rather than a fake zero",
          all(v != v for v in M.steer_brake_split([idle])))


def test_fallback_reference_is_announced(tmp):
    """A cross-detector reference must never be used silently."""
    ref = R.load_reference(os.path.join(tmp, "does_not_exist.json"))
    check("a missing reference file falls back", ref["measured"] is False)
    check("the fallback carries the warning", "Table 2" in ref["warning"])
    check("the banner shouts about it", R.banner(ref).startswith("WARNING"))

    path = os.path.join(tmp, "reference.json")
    with open(path, "w") as fh:
        json.dump(measured_ref()["ours"] and
                  {"ours": measured_ref()["ours"]}, fh)
    got = R.load_reference(path)
    check("a present reference file is used and marked measured",
          got["measured"] is True and abs(got["ours"]["collision"] - 0.044) < 1e-9)
    check("the banner says it is same-harness",
          "measured under this harness" in R.banner(got))


def test_end_to_end(tmp):
    d = os.path.join(tmp, "dom")
    out = os.path.join(tmp, "R1d")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "e4y",
                                      "analyze_e4y.py"),
         "--dir", d, "--out", out, "--n_boot", "64",
         "--reference", os.path.join(tmp, "no_such_reference.json")],
        capture_output=True, text=True)
    check("analyze_e4y.py exits cleanly", r.returncode == 0,
          (r.stderr or "")[-600:])
    check("writes the markdown table", os.path.exists(out + ".md"))
    check("writes the json blob", os.path.exists(out + ".json"))
    if os.path.exists(out + ".json"):
        blob = json.load(open(out + ".json"))
        check("json records the outcome", blob["gate"]["outcome"] == "dominated",
              "got %r" % blob["gate"]["outcome"])
        check("json records that the reference was not measured",
              blob["gate"]["reference_measured"] is False)
    check("stdout reports the registered predictions",
          "registered predictions" in r.stdout)
    check("stdout carries the fallback warning",
          "WARNING" in r.stdout, r.stdout[:200])
    check("the markdown carries the fallback warning too",
          "Table 2" in open(out + ".md").read())


# --------------------------------------------- look-ahead projection (torch) --

def test_unicycle_projection():
    try:
        import torch
    except ImportError:
        print("  SKIP projection tests -- torch not available")
        return
    from safeloco_eval.command_filters import UnicycleCBFCommandFilter

    class FakeEnv(object):
        """One env at the origin, heading +x, obstacles placed in world XY."""

        def __init__(self, obstacles, r=0.3, cmd=(0.6, 0.0, 0.0)):
            n = len(obstacles)
            self.root_states = torch.zeros(1, 13)
            self.root_states[0, 6] = 1.0                 # unit quaternion
            self.obstacle_positions = torch.zeros(1, max(n, 1), 3)
            for i, (x, y) in enumerate(obstacles):
                self.obstacle_positions[0, i, 0] = x
                self.obstacle_positions[0, i, 1] = y
            self.obstacle_radii = torch.full((1,), r)
            self.obstacle_mask = torch.zeros(1, max(n, 1), dtype=torch.bool)
            self.obstacle_mask[0, :n] = True
            self.has_obstacles = torch.ones(1, dtype=torch.bool)
            self.commands = torch.tensor([[cmd[0], cmd[1], cmd[2], 0.0]])

    def run(env, alpha=1.0, **kw):
        kw.setdefault("vx_range", (0.0, 0.8))
        kw.setdefault("omega_range", (-1.0, 1.0))
        f = UnicycleCBFCommandFilter(alpha=alpha, **kw)
        return f.apply(env.commands, env)

    # inert when the constraint is slack
    env = FakeEnv([(8.0, 0.0)])
    out, info = run(env)
    check("distant obstacle leaves the twist untouched",
          bool(torch.allclose(out, env.commands)) and
          not bool(info["activated"][0]))

    # --- the whole point: yaw is now a live control input -------------------
    env = FakeEnv([(1.2, 0.35)])            # ahead and to the LEFT
    out, info = run(env, alpha=1.0)
    check("an obstacle ahead activates the filter", bool(info["activated"][0]))
    check("obstacle on the left -> the filter steers RIGHT",
          float(out[0, 2]) < -1e-3, "omega=%.4f" % float(out[0, 2]))

    env = FakeEnv([(1.2, -0.35)])           # ahead and to the RIGHT
    out, _ = run(env, alpha=1.0)
    check("obstacle on the right -> the filter steers LEFT",
          float(out[0, 2]) > 1e-3, "omega=%.4f" % float(out[0, 2]))

    # the projection actually satisfies the constraint it was given
    env = FakeEnv([(1.2, 0.35)])
    f = UnicycleCBFCommandFilter(alpha=1.0, vx_range=(0.0, 0.8),
                                 omega_range=(-1.0, 1.0))
    out, info = f.apply(env.commands, env)
    a, b, mask, _ = f._constraints_twist(env)
    u = torch.stack([out[:, 0], out[:, 2]], dim=-1)
    resid = (a[0] * u[0]).sum(-1) - b[0]
    check("the projected twist satisfies a.u <= alpha*h",
          bool((resid[mask[0]] <= 1e-4).all()), "resid=%r" % resid.tolist())

    # --- registered prediction (i), without a GPU ---------------------------
    # Same geometry, same filter, one axis removed.  If the braking-only
    # infeasibility rate were a property of the world rather than of the input
    # space, both arms would fail here.
    env = FakeEnv([(0.62, 0.10)])
    _, i_yaw = run(env, alpha=0.5, omega_range=(-1.0, 1.0))
    _, i_noyaw = run(env, alpha=0.5, omega_range=(0.0, 0.0))
    check("yaw-disabled has no legal option where the steering arm does",
          bool(i_noyaw["infeasible"][0]) and not bool(i_yaw["infeasible"][0]),
          "yaw=%r noyaw=%r" % (bool(i_yaw["infeasible"][0]),
                               bool(i_noyaw["infeasible"][0])))

    # --- the weighting is a deliberate choice, so test that it bites --------
    env = FakeEnv([(1.2, 0.35)])
    _, cheap = run(env, alpha=1.0, w_v=4.0, w_omega=1.0)
    _, dear = run(env, alpha=1.0, w_v=1.0, w_omega=4.0)
    check("cheap yaw spends more of the correction on steering",
          float(cheap["dnorm_omega"][0]) / max(float(cheap["dnorm_v"][0]), 1e-9)
          > float(dear["dnorm_omega"][0]) / max(float(dear["dnorm_v"][0]), 1e-9),
          "cheap w/v=%.3f/%.3f  dear w/v=%.3f/%.3f" % (
              float(cheap["dnorm_omega"][0]), float(cheap["dnorm_v"][0]),
              float(dear["dnorm_omega"][0]), float(dear["dnorm_v"][0])))

    # --- the yaw-disabled arm is exactly the braking-only filter ------------
    env = FakeEnv([(1.2, 0.0)])
    env.commands[0, 2] = 0.4
    out, _ = run(env, alpha=1.0, omega_range=(0.0, 0.0))
    check("with the yaw axis removed the filter only brakes",
          abs(float(out[0, 2])) < 1e-9 and float(out[0, 0]) < 0.6,
          "omega=%r v=%r" % (float(out[0, 2]), float(out[0, 0])))

    # --- look-ahead length controls how much leverage yaw has ---------------
    env = FakeEnv([(1.2, 0.35)])
    short, _ = run(env, alpha=1.0, lookahead=0.05)
    long_, _ = run(env, alpha=1.0, lookahead=0.60)
    check("a longer look-ahead gives yaw more leverage",
          abs(float(long_[0, 2])) > abs(float(short[0, 2])),
          "L=0.05 -> %.4f, L=0.60 -> %.4f" % (float(short[0, 2]),
                                              float(long_[0, 2])))

    # --- alpha still orders conservatism ------------------------------------
    env = FakeEnv([(1.1, 0.0)])
    lo, _ = run(env, alpha=0.25)
    hi, _ = run(env, alpha=4.0)
    check("smaller alpha yields a more conservative forward speed",
          float(lo[0, 0]) <= float(hi[0, 0]) + 1e-9,
          "alpha=0.25 -> %.4f, alpha=4 -> %.4f" % (float(lo[0, 0]),
                                                   float(hi[0, 0])))

    # --- accounting invariant that once produced a bare math domain error ---
    env = FakeEnv([(0.55, 0.0)])
    _, info = run(env, alpha=0.01)
    check("infeasible is only ever reported on triggered steps",
          not bool(info["infeasible"][0]) or bool(info["triggered"][0]))

    # --- yaw is measured in the body frame, so heading must matter ----------
    env = FakeEnv([(0.0, 1.2)])             # obstacle to the WORLD +y
    psi = math.pi / 2                        # robot already faces +y
    env.root_states[0, 6] = math.cos(psi / 2)
    env.root_states[0, 5] = math.sin(psi / 2)
    out, info = run(env, alpha=1.0)
    check("an obstacle straight ahead in the body frame is seen as ahead",
          bool(info["activated"][0]) and float(out[0, 0]) < 0.6,
          "v=%.4f activated=%r" % (float(out[0, 0]),
                                   bool(info["activated"][0])))

    # no obstacles at all
    env = FakeEnv([])
    env.has_obstacles = torch.zeros(1, dtype=torch.bool)
    out, info = run(env)
    check("no obstacles -> twist untouched, not activated",
          bool(torch.allclose(out, env.commands)) and
          not bool(info["activated"][0]))


def main():
    tmp = tempfile.mkdtemp(prefix="e4y_offline_")
    try:
        print("[four-way gate]")
        test_dominated(tmp)
        test_matches_needs_the_whole_frontier(tmp)
        test_trivial_stop(tmp)
        print("\n[registered predictions]")
        test_predictions_are_computed(tmp)
        test_replica_is_kept_out_of_the_yaw_contrast(tmp)
        test_steer_brake_split_is_a_share()
        print("\n[reference rows]")
        test_fallback_reference_is_announced(tmp)
        print("\n[end to end]")
        test_end_to_end(tmp)
        print("\n[look-ahead unicycle projection]")
        test_unicycle_projection()
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
