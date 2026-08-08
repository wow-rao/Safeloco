"""ETRAP analysis -- turn the step-trap CSVs into the claim's numbers.

The table this produces backs one structural sentence:

    Inside a pen whose only crossable side is a low sill, every body-level
    (2-D) controller parks -- exit rate exactly 0, a property of the plane
    projection, not of any gain -- while the gradient-projection policy
    trained with the volumetric l-value steps over the sill and leaves.

Columns per row (one row = one run_etrap.py invocation):

    exit%        episodes whose base passed the gate's outer face (+margin),
                 Wilson 95% CI
    t_exit       median seconds to exit, over exited episodes
    prog+        mean metres of progress beyond the gate's outer face
    parked%      timed out inside the pen with < 0.25 m of net progress over
                 the second half of the episode
    wall-hit%    episodes with >= 1 step where any link was inside a wall
                 volume (volumetric test: stepping OVER with clearance does
                 not count -- the distinction the l-value encodes)
    apex_gate    mean swing-foot apex over the gate zone, exited episodes --
                 the "lifts its legs higher than usual" mechanism number,
                 read against apex_int (the same policy's swing apex on the
                 open pen floor)

Verdict logic:
    geometry_violated   any body-level controller row exited even once ->
                        the pen is not closed; fix the terrain before
                        reading anything else
    separated           body rows all at exactly 0 exits; the ours row exits
                        with a CI clear of zero
    not_separated       body rows at 0 but ours failed to exit too (the
                        argument is not backed; likely undertrained)

Run:  python experiments/etrap/analyze_etrap.py --dir logs/etrap/sweep \
          --out logs/etrap/R_trap
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from safeloco_eval import stats as ST  # noqa: E402

CONTROL_DT = 0.02


def _f(rec, key, default=0.0):
    try:
        return float(rec.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def load_rows(directory):
    """[(run_id, records)] sorted: unfiltered policies first, then filtered."""
    rows = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(directory, name)) as fh:
            recs = list(csv.DictReader(fh))
        if not recs:
            continue
        rid = recs[0].get("run_id") or os.path.splitext(name)[0]
        rows[rid] = recs

    def key(item):
        rid, recs = item
        ctrl = recs[0].get("controller", "none")
        return (0 if ctrl == "none" else 1, rid)

    return sorted(rows.items(), key=key)


def summarize(recs):
    n = len(recs)
    k_exit = sum(int(_f(r, "exited")) for r in recs)
    p, lo, hi = ST.wilson_ci(k_exit, max(n, 1))
    exit_secs = sorted(_f(r, "exit_step") * CONTROL_DT for r in recs
                       if int(_f(r, "exited")) and _f(r, "exit_step") >= 0)
    t_exit = exit_secs[len(exit_secs) // 2] if exit_secs else float("nan")
    prog = sum(_f(r, "progress_beyond_gate") for r in recs) / max(n, 1)
    parked = sum(int(_f(r, "parked")) for r in recs) / max(n, 1)
    wall = sum(int(_f(r, "contacted_wall")) for r in recs) / max(n, 1)
    wall_steps = sum(_f(r, "wall_contact_steps") for r in recs) / max(n, 1)
    apex_gate = [_f(r, "gate_foot_apex") for r in recs
                 if int(_f(r, "exited"))]
    apex_int = [_f(r, "interior_foot_apex") for r in recs]
    fell = sum(int(_f(r, "fell")) for r in recs) / max(n, 1)
    fwd = sum(_f(r, "mean_fwd_vel") for r in recs) / max(n, 1)
    return {
        "n": n, "k_exit": k_exit,
        "exit_rate": p, "exit_lo": lo, "exit_hi": hi,
        "t_exit_median_s": t_exit,
        "progress_beyond_gate": prog,
        "parked_rate": parked,
        "wall_contact_eps_rate": wall,
        "wall_contact_steps_per_ep": wall_steps,
        "apex_gate_mean": (sum(apex_gate) / len(apex_gate)
                           if apex_gate else float("nan")),
        "apex_interior_mean": (sum(apex_int) / len(apex_int)
                               if apex_int else float("nan")),
        "fall_rate": fell,
        "mean_fwd_vel": fwd,
        "controller": recs[0].get("controller", "none"),
        "policy_tag": recs[0].get("policy_tag", "?"),
    }


def by_gate_height(recs):
    """{gate_height: (k_exit, n)} -- the exit-vs-sill-height curve."""
    out = {}
    for r in recs:
        g = round(_f(r, "gate_height"), 3)
        k, n = out.get(g, (0, 0))
        out[g] = (k + int(_f(r, "exited")), n + 1)
    return dict(sorted(out.items()))


def apply_gate(rows):
    """(verdict, detail) over the summarized rows."""
    body = [(rid, s) for rid, s in rows
            if s["controller"] in ("body", "body_noyaw")]
    ours = [(rid, s) for rid, s in rows
            if s["controller"] == "none" and "ours" in s["policy_tag"]]

    leaks = [(rid, s["k_exit"]) for rid, s in body if s["k_exit"] > 0]
    if leaks:
        return "geometry_violated", (
            "a body-level controller exited the pen ({}); the 2-D ring is "
            "not closed -- fix the terrain before reading any other row"
            .format(", ".join("{}: {} exits".format(r, k)
                              for r, k in leaks)))
    if not body:
        return "incomplete", "no body-level controller rows found"
    if not ours:
        return "incomplete", "no gradient-projection (ours) row found"

    best = max(ours, key=lambda x: x[1]["exit_rate"])
    rid, s = best
    if s["exit_lo"] > 0.0:
        return "separated", (
            "every body-level row: 0 exits in {} episodes (parked). "
            "{}: {}/{} exits ({}), progress beyond gate {:.2f} m"
            .format(sum(x[1]["n"] for x in body), rid, s["k_exit"], s["n"],
                    ST.fmt_rate(s["exit_rate"], s["exit_lo"], s["exit_hi"]),
                    s["progress_beyond_gate"]))
    return "not_separated", (
        "body-level rows are parked as predicted, but {} exited only "
        "{}/{} -- the crossing was not learned; see the training gate"
        .format(rid, s["k_exit"], s["n"]))


def table(rows):
    hdr = ("| row | policy | controller | exit % | t_exit (s) | prog+ (m) | "
           "parked % | wall-hit % | apex gate (m) | apex floor (m) | "
           "fall % | n |")
    sep = "|" + "---|" * 12
    lines = [hdr, sep]
    for rid, s in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {:.2f} | {:.0f} | {:.1f} | {} | {} "
            "| {:.1f} | {} |".format(
                rid, s["policy_tag"], s["controller"],
                ST.fmt_rate(s["exit_rate"], s["exit_lo"], s["exit_hi"]),
                ("{:.1f}".format(s["t_exit_median_s"])
                 if s["t_exit_median_s"] == s["t_exit_median_s"] else "--"),
                s["progress_beyond_gate"],
                100.0 * s["parked_rate"],
                100.0 * s["wall_contact_eps_rate"],
                ("{:.3f}".format(s["apex_gate_mean"])
                 if s["apex_gate_mean"] == s["apex_gate_mean"] else "--"),
                ("{:.3f}".format(s["apex_interior_mean"])
                 if s["apex_interior_mean"] == s["apex_interior_mean"]
                 else "--"),
                100.0 * s["fall_rate"], s["n"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="logs/etrap/sweep")
    ap.add_argument("--out", type=str, default="logs/etrap/R_trap")
    args = ap.parse_args()

    raw = load_rows(args.dir)
    if not raw:
        raise SystemExit("no CSVs in {}".format(args.dir))
    rows = [(rid, summarize(recs)) for rid, recs in raw]
    verdict, detail = apply_gate(rows)

    md = ["# Step trap -- body-level avoidance vs gradient-projection "
          "leg-lift", "",
          "A closed pen; the only crossable side is a low sill ahead. "
          "Projected to the plane the pen is a closed ring, so a body-level "
          "controller has no exit whatever its gain or steering authority; "
          "the volumetric l-value keeps the space above the sill safe, so "
          "the gradient-projection policy may step over it.", "",
          table(rows), "",
          "**Verdict: {}** -- {}".format(verdict, detail), "",
          "## Exit rate vs sill height", ""]
    for rid, recs in raw:
        curve = by_gate_height(recs)
        pieces = ["{:.0f}cm: {}/{}".format(100 * g, k, n)
                  for g, (k, n) in curve.items()]
        md.append("* `{}`: {}".format(rid, "  ".join(pieces)))
    md.append("")
    md.append("Notes: exit = base past the gate's outer face; parked = timed "
              "out inside with < 0.25 m net progress over the episode's "
              "second half; wall-hit uses the volumetric test, so stepping "
              "over the sill with clearance is not a hit; apex gate vs apex "
              "floor is the leg-lift mechanism evidence.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(args.out + ".json", "w") as fh:
        json.dump({
            "verdict": verdict, "detail": detail,
            "rows": {rid: s for rid, s in rows},
            "exit_by_gate_height": {rid: {str(g): list(v)
                                          for g, v in
                                          by_gate_height(recs).items()}
                                    for rid, recs in raw},
        }, fh, indent=2)

    print("\n".join(md))
    print("\nwrote {}.md and {}.json".format(args.out, args.out))


if __name__ == "__main__":
    main()
