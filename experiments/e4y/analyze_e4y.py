"""E4-Y analysis -- the four-way decision gate, applied mechanically.

Differences from the braking-only analyser, all of them post-mortem items:

  * **Reference rows come from `safeloco_eval.reference`**, which prefers
    same-harness measurements and shouts when it is falling back to Table 2's
    cross-detector numbers.
  * **Episode-clustered intervals** on pooled-timestep rates, because
    timesteps inside an episode are not independent trials.
  * **Realised speed, distance and contact per metre** on every row, because
    any command-modifying filter can buy a lower contact rate with slowness.
  * **Steer/brake decomposition**, which says whether a filter given steering
    authority actually used it.

The gate compares each row against the reference on the joint frontier:

  dominated        reference better on contact, return, speed and distance
  competitive      better on contact but worse elsewhere, or very conservative
  matches          reaches the reference at comparable speed and distance
  trivial_stop     forward speed collapsed; the safety numbers describe a
                   robot that is barely moving and are not a frontier point

Run:
    python experiments/e4y/analyze_e4y.py --dir logs/e4y/sweep --out logs/e4y/R1d
"""

import argparse
import csv
import json
import os
import sys

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M          # noqa: E402
from safeloco_eval import stats as ST           # noqa: E402
from safeloco_eval.reference import load_reference, banner   # noqa: E402

TRIVIAL_STOP_FRACTION = 0.5
SPEED_TOLERANCE = 0.85      # "comparable speed" means within this of unfiltered


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e4y/sweep")
    p.add_argument("--out", type=str, default="logs/e4y/R1d")
    p.add_argument("--reference", type=str, default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--ignore_return", action="store_true",
                   help="drop the return axis from the gate. Required when "
                        "the evaluated policy was trained under a different "
                        "reward function from the reference -- a yaw-capable "
                        "baseline adds a tracking_ang_vel term, so its return "
                        "is not on the reference's scale. Auto-enabled when "
                        "the manifests say so.")
    return p.parse_args()


def load_rows(directory, n_boot=2000):
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
        variant = recs[0].get("variant", "")
        try:
            alpha = float(recs[0].get("epsilon_or_alpha", ""))
        except (TypeError, ValueError):
            alpha = None
        steer, brake = M.steer_brake_split(recs)
        row = {
            "label": name[:-4],
            "alpha": alpha,
            "yaw": ("noyaw" not in variant) and alpha is not None,
            # A look-ahead of zero puts the barrier back on the base centre,
            # which is the completed braking-only study rather than this
            # study's control arm.  It is shown in the table but kept out of
            # the yaw / yaw-disabled contrast, which is only meaningful
            # between arms that share a barrier.
            "replica": "braking_replica" in variant,
            "n_episodes": len(recs),
            "collision": ST.cluster_bootstrap_rate_ci(
                recs, "n_collision_steps", "n_steps", n_boot=n_boot),
            "fall": ST.wilson_ci(*M.fall_rate(recs)),
            "activation": ST.wilson_ci(*M.activation_rate(recs)),
            "infeasible": ST.wilson_ci(*M.infeasibility_rate(recs)),
            "ret": ST.mean_ci(M.per_episode(recs, "return")),
            "vel_err": ST.mean_ci(M.per_episode(recs, "vel_err")),
            "fwd_vel": M.mean_fwd_vel(recs),
            "distance": M.distance_travelled(recs),
            "contact_per_m": M.contact_per_metre(recs),
            "steer_share": steer,
            "yaw_exec": M.yaw_execution_gain(recs),
            "_recs": recs,
        }
        if alpha is None:
            unfiltered = row
        else:
            rows.append(row)
    rows.sort(key=lambda r: (r["replica"], not r["yaw"], r["alpha"]))
    return rows, unfiltered


def reward_function_matches(directory):
    """False when the sweep's policy was trained under a different reward.

    Table 2's return was measured with `tracking_ang_vel = 0`. A yaw-capable
    baseline needs that term non-zero to learn to turn at all, which shifts
    the whole return scale -- so comparing its return against the reference's
    is comparing two different objectives, and the gate must not score it.
    """
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".manifest.json"):
            continue
        try:
            with open(os.path.join(directory, name)) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if float(m.get("tracking_ang_vel", 0.0) or 0.0) != 0.0:
            return False
    return True


def classify(row, ref, unf, ignore_return=False):
    if unf and unf["fwd_vel"] > 1e-6:
        if row["fwd_vel"] < TRIVIAL_STOP_FRACTION * unf["fwd_vel"]:
            return "trivial_stop"
    rc = ref["ours"]["collision"]
    rr = ref["ours"].get("ret")
    ref_fwd = ref["ours"].get("fwd_vel")
    ref_dist = ref["ours"].get("distance")

    beats_contact = row["collision"][0] <= rc
    # Speed and distance only enter when the reference actually carries them,
    # which it does once the same-harness rows exist.
    speed_ok = True
    if ref_fwd:
        speed_ok = row["fwd_vel"] >= SPEED_TOLERANCE * ref_fwd
    elif unf:
        speed_ok = row["fwd_vel"] >= SPEED_TOLERANCE * unf["fwd_vel"]
    dist_ok = (row["distance"] >= SPEED_TOLERANCE * ref_dist) if ref_dist else speed_ok
    ret_ok = ignore_return or (rr is None) or (row["ret"][0] >= rr)

    if beats_contact and speed_ok and dist_ok and ret_ok:
        return "matches_or_dominates"
    if beats_contact:
        return "competitive"
    return "dominated_by_reference"


def apply_gate(rows, unfiltered, ref, ignore_return=False):
    per_row = {r["label"]: classify(r, ref, unfiltered, ignore_return)
               for r in rows}
    kinds = set(per_row.values())
    live = kinds - {"trivial_stop"}
    if kinds and kinds == {"trivial_stop"}:
        outcome = "not_decidable"
    elif "matches_or_dominates" in live:
        outcome = "matches_or_dominates"
    elif "competitive" in live:
        outcome = "competitive"
    else:
        outcome = "dominated"

    # The yaw / yaw-disabled contrast is only meaningful between arms that
    # share a barrier, so the base-centred replica is excluded from it.
    yaw_rows = [r for r in rows if r["yaw"] and not r["replica"]]
    noyaw_rows = [r for r in rows if not r["yaw"] and not r["replica"]]

    def mean_of(rs, key):
        vals = [r[key][0] if isinstance(r[key], tuple) else r[key] for r in rs]
        vals = [v for v in vals if v == v]
        return (sum(vals) / len(vals)) if vals else float("nan")

    return {
        "outcome": outcome,
        "per_row": per_row,
        "reference_measured": ref.get("measured", False),
        "return_axis_scored": not ignore_return,
        "reference": ref.get("ours", {}),
        # Registered prediction (i): the no-legal-option rate was an artifact
        # of a one-dimensional input, so it should fall once yaw is available.
        "infeasibility_yaw": mean_of(yaw_rows, "infeasible"),
        "infeasibility_noyaw": mean_of(noyaw_rows, "infeasible"),
        # Registered prediction (iv): the filter should actually use the axis.
        "steer_share_yaw": mean_of(yaw_rows, "steer_share"),
        "yaw_exec_yaw": mean_of(yaw_rows, "yaw_exec"),
        "contact_yaw": mean_of(yaw_rows, "collision"),
        "contact_noyaw": mean_of(noyaw_rows, "collision"),
    }


def table(rows, unfiltered, ref):
    head = ("| run | yaw | Contact % ↓ | Fall % ↓ | Fwd vel | Dist (m) | "
            "Contact/m | Return | Activation % | Infeas. % | Steer % |")
    out = [head, "|" + "---|" * 11]

    def line(r):
        arm = "yes" if r.get("yaw") else "no"
        if r.get("replica"):
            arm = "n/a (L=0)"
        return ("| {} | {} | {} | {} | {:.3f} | {:.2f} | {:.1f} | {} | {} | {} | {} |"
                .format(r["label"], arm,
                        ST.fmt_rate(*r["collision"]), ST.fmt_rate(*r["fall"]),
                        r["fwd_vel"], r["distance"], r["contact_per_m"],
                        ST.fmt_mean(r["ret"][0], r["ret"][1]),
                        ST.fmt_rate(*r["activation"]),
                        ST.fmt_rate(*r["infeasible"]),
                        ("{:.0f}".format(100 * r["steer_share"])
                         if r["steer_share"] == r["steer_share"] else "--")))

    if unfiltered:
        out.append(line(unfiltered))
    for r in rows:
        out.append(line(r))
    o = ref.get("ours", {})
    out.append("| **reference (proposed method)** | — | **{:.1f}** | — | {} | {} | — | {} | — | — | — |"
               .format(100 * o.get("collision", float("nan")),
                       ("{:.3f}".format(o["fwd_vel"]) if o.get("fwd_vel") else "—"),
                       ("{:.2f}".format(o["distance"]) if o.get("distance") else "—"),
                       ("{:.1f}".format(o["ret"]) if o.get("ret") else "—")))
    if any(r.get("replica") for r in rows):
        out.append("")
        out.append("Rows marked `L=0` write the barrier on the base centre, "
                   "where the yaw term vanishes identically. They are the "
                   "completed braking-only study re-run under the corrected "
                   "half-plane-plus-box projection, not a control arm of this "
                   "one, and are excluded from the yaw / yaw-disabled "
                   "comparison below.")
    if not ref.get("measured"):
        out.append("")
        out.append("**" + banner(ref) + "**")
    return "\n".join(out)


def main():
    args = parse_args()
    ref = load_reference(args.reference)
    rows, unfiltered = load_rows(args.dir, args.n_boot)
    if not rows and unfiltered is None:
        raise SystemExit("no E4-Y CSVs found in {}".format(args.dir))

    ignore_return = args.ignore_return or not reward_function_matches(args.dir)
    gate = apply_gate(rows, unfiltered, ref, ignore_return)

    print(banner(ref))
    if ignore_return:
        print("NOTE: the return axis is NOT scored. The evaluated policy was "
              "trained with a tracking_ang_vel term the reference did not "
              "have, so the two returns measure different objectives and "
              "comparing them would be meaningless. Contact, speed and "
              "distance are unaffected and still decide the gate.")
    print()
    print(table(rows, unfiltered, ref))

    print("\n--- registered predictions ---")
    iy, iny = gate["infeasibility_yaw"], gate["infeasibility_noyaw"]
    print("  (i)   no-legal-option rate  yaw {:.1f}%  vs  yaw-disabled {:.1f}%"
          .format(100 * iy, 100 * iny))
    print("        -> {}".format("HELD: steering restores feasibility"
                                 if iy < iny - 0.05 else
                                 "NOT held: feasibility did not improve"))
    cy, cn = gate["contact_yaw"], gate["contact_noyaw"]
    print("  (ii)  contact  yaw {:.2f}%  vs  yaw-disabled {:.2f}%"
          .format(100 * cy, 100 * cn))
    print("        -> {}".format("HELD" if cy < cn else "NOT held"))
    ye = gate["yaw_exec_yaw"]
    print("  (v)   yaw actually executed: {:.0f}% of what was commanded"
          .format(100 * ye) if ye == ye else
          "  (v)   yaw execution: n/a, no yaw was commanded")
    if ye == ye and ye < 0.5:
        print("        -> the platform does NOT track the yaw the filter asks")
        print("           for. The CBF certifies a twist it assumes is")
        print("           executed, so its guarantee does not hold here, and")
        print("           the filter will have braked less than a braking-only")
        print("           filter would on the strength of steering that never")
        print("           happened. Read a WORSE contact rate on the yaw arm")
        print("           as this, not as steering being unhelpful.")
    ss = gate["steer_share_yaw"]
    print("  (iv)  share of correction spent steering: {:.0f}%"
          .format(100 * ss if ss == ss else 0))
    print("        -> {}".format("HELD: the filter uses the new axis"
                                 if ss == ss and ss > 0.15 else
                                 "NOT held: it still mostly brakes"))

    print("\n--- gate ---")
    for k, v in gate["per_row"].items():
        print("  {:<34} {}".format(k, v))
    print("  OUTCOME: {}".format(gate["outcome"].upper()))
    if gate["outcome"] == "matches_or_dominates":
        print("  -> command filtering with real steering reaches the reference.")
        print("     Joint-limit constraints become the load-bearing")
        print("     differentiator; they cannot be expressed by a command")
        print("     filter at all, which is categorical rather than empirical.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("# Table R1d -- command filtering with steering authority\n\n")
        fh.write(table(rows, unfiltered, ref) + "\n\n## Gate\n\n```\n")
        for k, v in gate["per_row"].items():
            fh.write("{:<34} {}\n".format(k, v))
        fh.write("OUTCOME {}\n```\n".format(gate["outcome"]))
    for r in rows + ([unfiltered] if unfiltered else []):
        r.pop("_recs", None)
    with open(args.out + ".json", "w") as fh:
        json.dump({"rows": rows, "unfiltered": unfiltered, "gate": gate,
                   "reference": ref}, fh, indent=2, default=str)
    print("\nwrote {}.md / .json".format(args.out))


if __name__ == "__main__":
    main()
