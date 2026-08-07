"""E4 analysis -- applies analysis_protocol.md §3.4's gate mechanically.

The gate is read **on the joint frontier, not on collision rate alone**: each
alpha row is compared as a tuple against our Pareto point
(4.4% collision, 33.9 return, 0.085 vel err, 0.085 mean |lateral vel|).

  outcome 1  every row dominated by ours        -> include; completes taxonomy
  outcome 2  competitive collision, worse       -> include with Pareto framing
             elsewhere, or high activation
  outcome 3  some row matches or dominates ours -> pull J1 forward, treat J2
                                                   as rebuttal-critical

**On "all four".**  Dominance is scored on the three metrics that have a
direction: collision down, return up, velocity error down.  Mean |lateral
velocity| is reported beside them but not scored, because it has no monotone
sense here -- our own method sits *above* the unfiltered baseline on it
(0.085 against ~0.049), since sidestepping is how it buys clearance.  Scoring
it as "lower is better" would count our own signature against us; scoring it
as "higher is better" would reward any filter that merely wobbles.  It is a
behavioural signature and is reported as one.

Run:
    python experiments/e4/analyze_e4.py --dir logs/e4/sweep --out logs/e4/R1c
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e4/sweep")
    p.add_argument("--out", type=str, default="logs/e4/R1c")
    return p.parse_args()


def load_rows(directory):
    """One row per CSV, keyed by alpha (None for the unfiltered row)."""
    if not os.path.isdir(directory):
        raise SystemExit("no such directory: {}".format(directory))
    rows, unfiltered = [], None
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(directory, name)) as fh:
            recs = list(csv.DictReader(fh))
        if not recs:
            continue
        raw = recs[0].get("epsilon_or_alpha", "")
        try:
            alpha = float(raw)
        except (TypeError, ValueError):
            alpha = None

        row = {
            "alpha": alpha,
            "variant": recs[0].get("variant", ""),
            "n_episodes": len(recs),
            "collision": ST.wilson_ci(*M.collision_rate(recs)),
            "fall": ST.wilson_ci(*M.fall_rate(recs)),
            "activation": ST.wilson_ci(*M.activation_rate(recs)),
            "infeasible": ST.wilson_ci(*M.infeasibility_rate(recs)),
            "ret": ST.mean_ci(M.per_episode(recs, "return")),
            "vel_err": ST.mean_ci(M.per_episode(recs, "vel_err")),
            "lat_vel": ST.mean_ci(M.per_episode(recs, "mean_lat_vel")),
            "fwd_vel": ST.mean_ci(M.per_episode(recs, "mean_fwd_vel")),
        }
        if alpha is None:
            unfiltered = row
        else:
            rows.append(row)
    rows.sort(key=lambda r: r["alpha"])
    return rows, unfiltered


# ------------------------------------------------------------------- gate --

#: A filter that stops the robot trivially avoids collisions (§1's first
#: cross-cutting rule).  Below this fraction of the unfiltered forward speed,
#: the safety numbers describe a stationary robot and the row is not a
#: meaningful point on the frontier.
TRIVIAL_STOP_FRACTION = 0.5


def classify(row, fwd_ref=None):
    """Per-row comparison against our Pareto point, on the three scored axes."""
    if fwd_ref is not None and fwd_ref > 1e-6:
        if row["fwd_vel"][0] < TRIVIAL_STOP_FRACTION * fwd_ref:
            return "trivial_stop"
    c, r, v = row["collision"][0], row["ret"][0], row["vel_err"][0]
    better_or_equal = (c <= M.OURS_COLLISION_RATE and r >= M.OURS_RETURN
                       and v <= M.OURS_MEAN_LAT_VEL)
    worse_on_all = (c > M.OURS_COLLISION_RATE and r < M.OURS_RETURN
                    and v > M.OURS_MEAN_LAT_VEL)
    if better_or_equal:
        return "matches_or_dominates"
    if worse_on_all:
        return "dominated_by_ours"
    return "mixed"


def apply_gate(rows, unfiltered=None):
    fwd_ref = unfiltered["fwd_vel"][0] if unfiltered else None
    per_row = {}
    for row in rows:
        per_row["alpha={:g}".format(row["alpha"])] = classify(row, fwd_ref)

    kinds = set(per_row.values())
    live = kinds - {"trivial_stop"}
    if kinds and kinds == {"trivial_stop"}:
        # Every row stopped the robot: collision, fall and velocity error all
        # describe a stationary machine, and return is measured against a
        # command the filter rewrote.  There is no frontier point to gate on.
        outcome = 0
    elif "matches_or_dominates" in live:
        outcome = 3
    elif live and live == {"dominated_by_ours"}:
        outcome = 1
    else:
        outcome = 2

    best = min(rows, key=lambda r: r["collision"][0]) if rows else None
    return {
        "outcome": outcome,
        "per_row": per_row,
        "reference": {"collision": M.OURS_COLLISION_RATE,
                      "return": M.OURS_RETURN,
                      "vel_err": M.OURS_MEAN_LAT_VEL,
                      "lat_vel": M.OURS_MEAN_LAT_VEL},
        "best_collision_row": ("alpha={:g}".format(best["alpha"]) if best else None),
        "unfiltered_fwd_vel": fwd_ref,
        "trivial_stop_threshold": (TRIVIAL_STOP_FRACTION * fwd_ref
                                   if fwd_ref else None),
        "return_reference_warning": (
            "Return is computed by the env against env.commands, which a "
            "command filter rewrites. A filter that lowers the command is "
            "scored against the easier command and its return rises. Do not "
            "compare a filtered return against ours without accounting for "
            "this."),
        "scored_axes": ["collision", "return", "vel_err"],
        "reported_not_scored": ["mean_lat_vel"],
    }


def sanity_checks(rows, unfiltered):
    """§5, adapted: activation must rise as alpha tightens."""
    out = []
    if unfiltered is not None:
        ok = unfiltered["activation"][0] <= 1e-9
        out.append(("unfiltered row has zero activation", ok,
                    "{:.4f}%".format(100 * unfiltered["activation"][0])))
    if len(rows) >= 2:
        # alpha ascending == progressively more permissive == less activation
        acts = [r["activation"][0] for r in rows]
        mono = all(acts[i] >= acts[i + 1] - 1e-6 for i in range(len(acts) - 1))
        out.append(("activation non-increasing as alpha grows", mono,
                    "; ".join("{:g}->{:.3f}".format(r["alpha"], a)
                              for r, a in zip(rows, acts))))
    for r in rows:
        if r["infeasible"][0] > 0.05:
            out.append(("alpha={:g} infeasible box on <5% of active steps"
                        .format(r["alpha"]), False,
                        "{:.1f}%".format(100 * r["infeasible"][0])))
    return out


# ----------------------------------------------------------------- output --

S7 = {
    0: ("Command-space filtering, applied to this policy, avoids collisions by "
        "bringing the robot to a halt: forward speed collapses to a fraction "
        "of nominal and the safety numbers describe a stationary machine. No "
        "usable point on the frontier was produced, so the taxonomy row is "
        "reported with that caveat rather than as a competitive baseline."),
    1: ("Command-space filtering -- the form these methods actually take -- is "
        "included as a baseline; it is dominated by ours on collision rate, "
        "return and velocity tracking simultaneously, and cannot express "
        "joint-level constraints."),
    2: ("Command-space filtering -- the form these methods actually take -- is "
        "included as a baseline; it trades {trade} and cannot express "
        "joint-level constraints."),
    3: ("Command-space filtering suffices for pure collision avoidance, as "
        "expected given it is the setting these methods were designed for; it "
        "cannot address joint-level constraints, which we demonstrate in "
        "Tables R1b/R2b."),
}


def trade_phrase(rows):
    if not rows:
        return "collision rate against return"
    best = min(rows, key=lambda r: r["collision"][0])
    return ("collision rate down to {:.1f}% against return {:.1f} and "
            "velocity error {:.3f} (ours: {:.1f}%, {:.1f}, {:.3f})".format(
                100 * best["collision"][0], best["ret"][0], best["vel_err"][0],
                100 * M.OURS_COLLISION_RATE, M.OURS_RETURN,
                M.OURS_MEAN_LAT_VEL))


def table(rows, unfiltered):
    head = ("| alpha | Collision % ↓ | Fall % ↓ | **Fwd vel** ↑ | Return ↑† | "
            "Vel. err ↓ | \\|lat vel\\| | Activation % | Infeas. % |")
    out = [head, "|---|---|---|---|---|---|---|---|---|"]

    def line(label, r):
        return "| {} | {} | {} | **{:.3f}** | {} | {} | {} | {} | {} |".format(
            label,
            ST.fmt_rate(*r["collision"]), ST.fmt_rate(*r["fall"]),
            r["fwd_vel"][0],
            ST.fmt_mean(r["ret"][0], r["ret"][1]),
            ST.fmt_mean(r["vel_err"][0], r["vel_err"][1], digits=3),
            ST.fmt_mean(r["lat_vel"][0], r["lat_vel"][1], digits=3),
            ST.fmt_rate(*r["activation"]), ST.fmt_rate(*r["infeasible"]))

    if unfiltered is not None:
        out.append(line("unfiltered", unfiltered))
    for r in rows:
        out.append(line("{:g}".format(r["alpha"]), r))
    out.append("| **ours (reference)** | **{:.1f}** | — | — | **{:.1f}** | "
               "**{:.3f}** | {:.3f} | — | — |".format(
                   100 * M.OURS_COLLISION_RATE, M.OURS_RETURN,
                   M.OURS_MEAN_LAT_VEL, M.OURS_MEAN_LAT_VEL))
    out.append("")
    out.append("† Return is computed against `env.commands`, which the command "
               "filter rewrites; a filter that lowers the command is scored "
               "against the easier command. Not comparable across rows.")
    return "\n".join(out)


def main():
    args = parse_args()
    rows, unfiltered = load_rows(args.dir)
    if not rows and unfiltered is None:
        raise SystemExit("no E4 CSVs found in {}".format(args.dir))

    gate = apply_gate(rows, unfiltered)
    checks = sanity_checks(rows, unfiltered)

    print(table(rows, unfiltered))

    print("\n--- sanity checks (§5) ---")
    for name, ok, detail in checks:
        print(" [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

    print("\n--- gate (§3.4) ---")
    for k, v in sorted(gate["per_row"].items()):
        print("  {:<14} {}".format(k, v))
    print("  scored on: {} | reported not scored: {}".format(
        ", ".join(gate["scored_axes"]), ", ".join(gate["reported_not_scored"])))
    print("  OUTCOME {}".format(gate["outcome"]))
    if gate["outcome"] == 3:
        print("  -> §3.4: pull J1 forward and treat J2 as rebuttal-critical.")
        print("     Decide this now, from these numbers, not later.")

    sentence = S7[gate["outcome"]]
    if gate["outcome"] == 2:
        sentence = sentence.format(trade=trade_phrase(rows))
    print("\n--- §7 sentence ---")
    print(sentence)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("# Table R1c -- E4 command-space filtering\n\n")
        fh.write(table(rows, unfiltered) + "\n\n")
        fh.write("## Gate (§3.4)\n\n```\n")
        for k, v in sorted(gate["per_row"].items()):
            fh.write("{:<14} {}\n".format(k, v))
        fh.write("OUTCOME {}\n```\n\n## §7 sentence\n\n{}\n".format(
            gate["outcome"], sentence))
    with open(args.out + ".json", "w") as fh:
        json.dump({"rows": rows, "unfiltered": unfiltered, "gate": gate,
                   "sanity": [{"check": n, "pass": bool(o), "detail": d}
                              for n, o, d in checks],
                   "sentence": sentence}, fh, indent=2, default=str)
    print("\nwrote {}.md / .json".format(args.out))


if __name__ == "__main__":
    main()
