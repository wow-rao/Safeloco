"""E1 analysis: per-episode CSVs -> Table R1a, verdict, and the headline figure.

analysis_protocol.md, in order:
  §5  sanity checks first -- each has caught real bugs in this class of study
  §1  metric definitions, applied from safeloco_eval/metrics.py only
  §2  Wilson CIs on rates; mean +/- std (and SE) on per-episode means
  §3.1 the E1 verdict rule, applied mechanically
  §7  the pre-written rebuttal sentence the outcome triggers
  §4  figure 1's right panel (becomes half of J4's asymmetry figure)

The governing principle: the predictions were registered before the runs, so
this script computes the pre-specified quantities and applies the pre-specified
rules.  It does not add metrics after seeing results.  If a prediction misses,
it prints the pre-written reframing sentence and says the prediction was
falsified -- which is a more credible rebuttal than one where everything
landed.

Usage
-----
    python experiments/e1/analyze_e1.py --dir logs/e1/sweep --out logs/e1/R1a
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from safeloco_eval import metrics as M
from safeloco_eval import stats as ST

PRIMARY_VARIANT = "A_sampling"
LARGE_EPS = 0.5
PI_NOM_LAT_VEL_REF = 0.049      # protocol §3.1 supporting diagnostic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e1/sweep")
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--out", type=str, default="logs/e1/R1a")
    p.add_argument("--qsafe_report", type=str, default=None,
                   help="qsafe .report.json; gates the §7 interpretation")
    p.add_argument("--no_figure", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------- io --

def load_runs(paths):
    runs = {}
    for path in paths:
        with open(path) as f:
            recs = list(csv.DictReader(f))
        if not recs:
            continue
        run_id = recs[0]["run_id"]
        man_path = path[:-4] + ".manifest.json"
        man = {}
        if os.path.exists(man_path):
            with open(man_path) as f:
                man = json.load(f)
        runs[run_id] = {"records": recs, "manifest": man, "csv": path}
    return runs


def epsilon_of(run):
    v = run["manifest"].get("epsilon_rad")
    if v is not None:
        return float(v)
    raw = run["records"][0]["epsilon_or_alpha"]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def is_primary_sweep(run):
    """Variant A, triggered (not always-on), primary tau, single head."""
    m, r = run["manifest"], run["records"][0]
    if r["variant"] != PRIMARY_VARIANT:
        return False
    if m.get("always_on"):
        return False
    if m and abs(float(m.get("tau", -0.25)) + 0.25) > 1e-9:
        return False
    if m.get("trigger_source", "vhat") != "vhat":
        return False
    return True


# ------------------------------------------------------------------ table --

def summarise(run_id, run):
    recs = run["records"]
    row = {"run_id": run_id, "variant": recs[0]["variant"],
           "epsilon": epsilon_of(run), "n_episodes": len(recs)}

    row["fall"] = ST.wilson_ci(*M.fall_rate(recs))
    row["collision"] = ST.wilson_ci(*M.collision_rate(recs))
    row["eps_w_coll"] = ST.wilson_ci(*M.eps_with_collision(recs))
    row["activation"] = ST.wilson_ci(*M.activation_rate(recs))
    row["target_miss"] = ST.wilson_ci(*M.target_miss_rate(recs))
    row["trigger"] = ST.wilson_ci(
        sum(int(r["trigger_steps"]) for r in recs),
        sum(int(r["n_steps"]) for r in recs))
    row["viol_per_step"] = ST.wilson_ci(*M.viol_per_step(recs))

    for name, field in (("ret", "return"), ("vel_err", "vel_err"),
                        ("lat_vel", "mean_lat_vel"), ("mean_h", "mean_h"),
                        ("peak_yaw", "peak_yaw"), ("peak_jerk", "peak_jerk"),
                        ("n_steps", "n_steps")):
        xs = M.per_episode(recs, field)
        row[name] = (ST.mean(xs), ST.std(xs), ST.sem(xs))

    row["min_h"] = M.min_over_runs(recs, "min_h")
    row["mean_dnorm"] = M.weighted_step_mean(recs, "sum_dnorm_inf")
    row["max_dnorm"] = max((float(r["max_dnorm_inf"]) for r in recs),
                           default=0.0)
    # Correction magnitude by contact phase -- feeds E6.
    for ph in ("flight", "partial", "stance"):
        num = sum(float(r["dnorm_" + ph]) for r in recs)
        den = sum(int(r["steps_" + ph]) for r in recs)
        row["dnorm_" + ph] = num / den if den else float("nan")
    return row


#: (key, markdown header, LaTeX header).  Keeping the paper's up/down arrows.
R1A_COLUMNS = [
    ("eps",          "eps (rad)",              r"$\varepsilon$ (rad)"),
    ("fall",         "Fall % &darr;",          r"Fall \% $\downarrow$"),
    ("collision",    "Collision % &darr;",     r"Collision \% $\downarrow$"),
    ("eps_w_coll",   "Eps. w/ coll. % &darr;", r"Eps.\ w/ coll.\ \% $\downarrow$"),
    ("ret",          "Return &uarr;",          r"Return $\uparrow$"),
    ("vel_err",      "Vel. err &darr;",        r"Vel.\ err $\downarrow$"),
    ("lat_vel",      "\\|lat vel\\| (m/s)",    r"$|$lat vel$|$ (m/s)"),
    ("activation",   "Activation %",           r"Activation \%"),
    ("target_miss",  "Target-miss %",          r"Target-miss \%"),
]
RATE_KEYS = ("fall", "collision", "eps_w_coll", "activation", "target_miss")


def render_table(rows, ref_rows):
    """Markdown Table R1a (protocol §14 item 4 schema).

    Bolding rule (§6): bold the best value in a column only when its CI is
    separated from the runner-up; otherwise bold both and say so in the
    caption.  Implemented for the two rate columns where it matters.
    """
    head = "| " + " | ".join(c[1] for c in R1A_COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(R1A_COLUMNS)) + "|"
    lines = [head, sep]

    def fmt(row):
        eps = ("--" if row["epsilon"] is None
               else "{:g}".format(row["epsilon"]))
        cells = [row.get("_label", eps)]
        for key, _, _ in R1A_COLUMNS[1:]:
            v = row[key]
            if key in RATE_KEYS:
                cells.append(ST.fmt_rate(*v))
            else:
                cells.append(ST.fmt_mean(
                    v[0], v[1],
                    digits=3 if key in ("vel_err", "lat_vel") else 1))
        return "| " + " | ".join(cells) + " |"

    for row in ref_rows + rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def render_latex(rows, ref_rows):
    def cell(row, key):
        v = row[key]
        if key in RATE_KEYS:
            if v[0] != v[0]:            # NaN, e.g. target-miss with no
                return "--"             # filter-active steps
            return "{:.1f}".format(100 * v[0])
        d = 3 if key in ("vel_err", "lat_vel") else 1
        return "{:.{d}f} $\\pm$ {:.{d}f}".format(v[0], v[1], d=d)

    out = ["\\begin{tabular}{l" + "c" * (len(R1A_COLUMNS) - 1) + "}",
           "\\toprule",
           " & ".join(c[2] for c in R1A_COLUMNS) + " \\\\",
           "\\midrule"]
    for row in ref_rows + rows:
        eps = row.get("_label",
                      "--" if row["epsilon"] is None
                      else "{:g}".format(row["epsilon"]))
        eps = eps.replace("$\\pi_{nom}$", "$\\pi_\\mathrm{nom}$")
        out.append(" & ".join([eps] + [cell(row, k)
                                       for k, _, _ in R1A_COLUMNS[1:]])
                   + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


# ---------------------------------------------------------------- verdict --

def apply_verdict(sweep_rows, unfiltered_row):
    """analysis_protocol.md §3.1, applied mechanically."""
    u_fall = unfiltered_row["fall"]
    checks, detail = {}, {}

    # 1. Large-eps destabilisation: fall at eps=0.5 exceeds unfiltered with
    #    non-overlapping Wilson CIs.
    big = [r for r in sweep_rows
           if r["epsilon"] is not None and abs(r["epsilon"] - LARGE_EPS) < 1e-9]
    if big:
        f = big[0]["fall"]
        checks["large_eps_destabilises"] = (
            f[0] > u_fall[0] and not ST.ci_overlap(f[1:], u_fall[1:]))
        detail["large_eps"] = {
            "epsilon": LARGE_EPS,
            "fall": ST.fmt_rate(*f), "unfiltered_fall": ST.fmt_rate(*u_fall)}
    else:
        checks["large_eps_destabilises"] = None
        detail["large_eps"] = "eps = 0.5 not present in the sweep"

    # 2. Small-eps impotence: at every eps whose fall CI overlaps unfiltered,
    #    collision rate stays >= ~10%.
    overlapping, impotent = [], True
    for r in sweep_rows:
        if ST.ci_overlap(r["fall"][1:], u_fall[1:]):
            overlapping.append(r)
            if r["collision"][0] < M.VERDICT_SMALL_EPS_FLOOR:
                impotent = False
    checks["small_eps_impotent"] = impotent if overlapping else None
    detail["fall_matched_points"] = [
        {"epsilon": r["epsilon"], "collision": ST.fmt_rate(*r["collision"]),
         "fall": ST.fmt_rate(*r["fall"])} for r in overlapping]

    # 3. No viable eps: none satisfies fall ~ unfiltered AND collision <= 6%
    #    AND return >= 30 simultaneously.
    viable = []
    for r in sweep_rows:
        if (ST.ci_overlap(r["fall"][1:], u_fall[1:])
                and r["collision"][0] <= M.VERDICT_COLLISION_CEILING
                and r["ret"][0] >= M.VERDICT_RETURN_FLOOR):
            viable.append(r)
    checks["no_viable_epsilon"] = (len(viable) == 0)
    detail["viable_points"] = [
        {"epsilon": r["epsilon"], "collision": ST.fmt_rate(*r["collision"]),
         "fall": ST.fmt_rate(*r["fall"]),
         "return": ST.fmt_mean(r["ret"][0], r["ret"][1])} for r in viable]

    holds = all(v is True for v in checks.values())
    return {"checks": checks, "detail": detail, "holds": holds,
            "falsified": bool(viable)}


def supporting_diagnostics(sweep_rows, unfiltered_row):
    """Report these even when the primary verdict is clean (§3.1)."""
    pts = sorted([r for r in sweep_rows if r["epsilon"] is not None],
                 key=lambda r: r["epsilon"])
    miss = [(r["epsilon"], r["target_miss"][0]) for r in pts]
    rises_as_eps_falls = all(a[1] >= b[1] - 1e-9
                             for a, b in zip(miss, miss[1:]))
    lat = [(r["epsilon"], r["lat_vel"][0]) for r in pts]
    max_lat = max((v for _, v in lat), default=float("nan"))
    return {
        "target_miss_vs_eps": [{"epsilon": e, "target_miss": v}
                               for e, v in miss],
        "target_miss_rises_as_eps_falls": rises_as_eps_falls,
        "lat_vel_vs_eps": [{"epsilon": e, "mean_lat_vel": v} for e, v in lat],
        "unfiltered_lat_vel": unfiltered_row["lat_vel"][0],
        "pi_nom_reference_lat_vel": PI_NOM_LAT_VEL_REF,
        "ours_lat_vel": M.OURS_MEAN_LAT_VEL,
        "max_sweep_lat_vel": max_lat,
        "no_emergent_sidestepping": max_lat < 0.5 * (
            PI_NOM_LAT_VEL_REF + M.OURS_MEAN_LAT_VEL),
    }


# ------------------------------------------------------------------- §7 -----

S7_HOLDS = (
    "A joint-space projection filter admits no viable trust region: at "
    "eps >= {eps_bad} it destabilizes the gait (fall rate {fall_bad} vs "
    "{fall_unf} unfiltered), while at eps <= {eps_small} it leaves collision "
    "rate at {coll_small}, near the reward-shaping baseline and far from ours "
    "(4.4%); target-miss rates of {miss_small} show the demanded corrections "
    "exceed the trust region, reflecting the high, contact-gated relative "
    "degree of clearance with respect to joint commands.")

S7_FALSIFIED = (
    "Learned-barrier projection can reduce collisions at deployment "
    "({coll_viable} at eps = {eps_viable}), but retains a permanent runtime "
    "loop, executes the critic's pointwise errors directly as motor commands, "
    "and cannot express joint-limit constraints; our formulation obtains "
    "comparable clearance with no deployment-time component.")

S7_QHAT_FAILED = (
    "We were unable to train a joint-space action-value of sufficient quality "
    "to filter against (rho = {rho:.2f}, AUROC = {auroc:.2f}), which is itself "
    "the point: no analytic joint-space collision barrier exists, and the "
    "learned substitute that runtime methods would require is difficult to "
    "obtain at usable fidelity.")


def rebuttal_sentence(verdict, sweep_rows, unfiltered_row, qsafe_report):
    """The §7 decision tree.  The Q-hat guard outranks the sweep outcome."""
    if qsafe_report:
        b = qsafe_report.get("battery", {})
        if not qsafe_report.get("verdict", {}).get("critic_informative", True):
            return ("qhat_failed_validation",
                    S7_QHAT_FAILED.format(
                        rho=b.get("spearman_rho_q", float("nan")),
                        auroc=b.get("auroc_q_collision20", float("nan"))))

    if verdict["falsified"]:
        v = verdict["detail"]["viable_points"][0]
        return ("e1_falsified",
                S7_FALSIFIED.format(coll_viable=v["collision"].split(" ")[0]
                                    + "%", eps_viable=v["epsilon"]))

    if verdict["holds"]:
        pts = sorted([r for r in sweep_rows if r["epsilon"] is not None],
                     key=lambda r: r["epsilon"])
        u = unfiltered_row["fall"]
        # Onset of destabilisation: the *smallest* eps whose fall CI clears
        # unfiltered, not merely the largest eps in the sweep.
        bad = next((r for r in pts if r["fall"][0] > u[0]
                    and not ST.ci_overlap(r["fall"][1:], u[1:])), pts[-1])
        # Boundary of the impotent regime: the *largest* eps still statistically
        # indistinguishable from unfiltered on falls.
        overlap = [r for r in pts if ST.ci_overlap(r["fall"][1:], u[1:])]
        small = overlap[-1] if overlap else pts[0]
        return ("e1_holds", S7_HOLDS.format(
            eps_bad="{:g}".format(bad["epsilon"]),
            fall_bad=ST.fmt_rate(*bad["fall"]).split(" ")[0] + "%",
            fall_unf=ST.fmt_rate(*u).split(" ")[0] + "%",
            eps_small="{:g}".format(small["epsilon"]),
            coll_small=ST.fmt_rate(*small["collision"]).split(" ")[0] + "%",
            miss_small=ST.fmt_rate(*small["target_miss"]).split(" ")[0] + "%"))

    return ("e1_partial",
            "E1's registered prediction held in part only: "
            + "; ".join("{} = {}".format(k, v)
                        for k, v in verdict["checks"].items())
            + ". Report the checks that failed in the rebuttal body, not an "
              "appendix (analysis_protocol.md §8).")


# ---------------------------------------------------------------- figure ----

def make_figure(sweep_rows, unfiltered_row, out_path, variant_b_rows=None):
    """Fall-rate and collision-rate vs eps with pi_nom and pi_ours reference
    lines; shaded Wilson CIs; colourblind-safe and legible in greyscale
    (§4.1) -- reviewers print."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib not available; plot data written as JSON only")
        return None

    pts = sorted([r for r in sweep_rows if r["epsilon"] is not None],
                 key=lambda r: r["epsilon"])
    xs = [r["epsilon"] for r in pts]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    def band(ax_, key, colour, marker, ls, label):
        y = [100 * r[key][0] for r in pts]
        lo = [100 * r[key][1] for r in pts]
        hi = [100 * r[key][2] for r in pts]
        ax_.plot(xs, y, marker=marker, ls=ls, color=colour, label=label, lw=1.8)
        ax_.fill_between(xs, lo, hi, color=colour, alpha=0.18, lw=0)

    band(ax, "collision", "#0072B2", "o", "-", "Collision rate (filtered)")
    ax.axhline(100 * unfiltered_row["collision"][0], color="#555555", ls=":",
               lw=1.4, label="Collision, unfiltered $\\pi_{nom}$")
    ax.axhline(100 * M.OURS_COLLISION_RATE, color="#009E73", ls="-.", lw=1.4,
               label="Collision, ours (4.4%)")

    ax2 = ax.twinx()
    band(ax2, "fall", "#D55E00", "s", "--", "Fall rate (filtered)")
    ax2.axhline(100 * unfiltered_row["fall"][0], color="#D55E00", ls=(0, (1, 3)),
                lw=1.2, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(r"trust region $\varepsilon$  (rad)")
    ax.set_ylabel("Collision rate (%)")
    ax2.set_ylabel("Fall rate (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["{:g}".format(x) for x in xs])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("E1: no $\\varepsilon$ achieves both", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("{}.{}".format(out_path, ext), dpi=200)
    plt.close(fig)
    return out_path + ".pdf"


# ------------------------------------------------------------------ main ----

def main():
    args = parse_args()
    paths = args.files or sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not paths:
        raise SystemExit("no CSVs found in " + args.dir)
    runs = load_runs(paths)

    unfiltered = [r for r in runs.values()
                  if r["records"][0]["variant"] == "none"]
    if not unfiltered:
        raise SystemExit("no unfiltered row (variant='none') found -- the "
                         "verdict rule is defined relative to it")
    unf_id = [k for k, v in runs.items() if v in unfiltered][0]

    rows = {k: summarise(k, v) for k, v in runs.items()}
    unf_row = rows[unf_id]
    unf_row["_label"] = "unfiltered $\\pi_{nom}$"

    sweep_keys = [k for k, v in runs.items() if is_primary_sweep(v)]
    sweep_keys.sort(key=lambda k: rows[k]["epsilon"])
    sweep = [rows[k] for k in sweep_keys]
    other_keys = [k for k in runs if k != unf_id and k not in set(sweep_keys)]
    others = [rows[k] for k in sorted(other_keys)]
    for r in others:
        r["_label"] = ("{} ({:g})".format(r["variant"], r["epsilon"])
                       if r["epsilon"] is not None else r["variant"])

    if not sweep:
        raise SystemExit("no primary variant-A sweep points found")

    # ---- §5 sanity checks, before believing anything -------------------
    # The trust-region assertion covers every run; monotonic activation is a
    # statement about the primary sweep only, since always-on and variant B
    # points sit on different curves.
    sanity = M.sanity_checks({k: v["records"] for k, v in runs.items()},
                             unf_id, mono_keys=sweep_keys)

    verdict = apply_verdict(sweep, unf_row)
    diagnostics = supporting_diagnostics(sweep, unf_row)

    qsafe_report = None
    qpath = args.qsafe_report
    if qpath is None:
        for v in runs.values():
            if v["manifest"].get("qsafe_battery"):
                qsafe_report = {"battery": v["manifest"]["qsafe_battery"],
                                "verdict": v["manifest"].get("qsafe_verdict",
                                                             {})}
                break
    elif os.path.exists(qpath):
        with open(qpath) as f:
            qsafe_report = json.load(f)

    key, sentence = rebuttal_sentence(verdict, sweep, unf_row, qsafe_report)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    md = render_table(sweep, [unf_row] + others)
    caption = (
        "\n\n*Table R1a. E1, deployment-time joint-space projection (variant A "
        "sampling projection unless noted), corridor, N = {} episodes per row "
        "on the shared eval seed list, domain randomisation {}. Rates carry "
        "Wilson 95% CIs; per-episode means carry +/- std. Collision = "
        "timesteps with min_cbf_h < {}; fall = terminal base tilt beyond 60 "
        "deg or base height below {} m (one definition across both branches). "
        "Episodes end at their fall, so no post-fall timesteps enter any "
        "denominator. Reference: ours 4.4% collision / 33.9 return / 0.085 "
        "m/s vel err.*\n").format(
            len(runs[unf_id]["records"]),
            runs[unf_id]["manifest"].get("dr_mode", "?"),
            M.COLLISION_H_THRESH, M.FALL_HEIGHT)

    with open(args.out + ".md", "w") as f:
        f.write("# Table R1a -- E1 joint-space projection (collision)\n\n")
        f.write(md + caption)
        f.write("\n## Sanity checks (analysis_protocol.md §5)\n\n")
        for name, ok, det in sanity:
            f.write("- [{}] {} -- {}\n".format("x" if ok else " ", name, det))
        f.write("\n## Verdict (analysis_protocol.md §3.1)\n\n")
        for k, v in verdict["checks"].items():
            f.write("- {}: **{}**\n".format(
                k, {True: "HELD", False: "FAILED", None: "N/A"}[v]))
        f.write("\nPrimary two-regime claim: **{}**\n".format(
            "HOLDS" if verdict["holds"] else
            ("FALSIFIED" if verdict["falsified"] else "PARTIAL")))
        f.write("\n## Supporting diagnostics\n\n")
        f.write("- target-miss rises as eps falls: **{}**\n".format(
            diagnostics["target_miss_rises_as_eps_falls"]))
        f.write("- max mean |lateral vel| across sweep: {:.4f} m/s "
                "(pi_nom ref {:.3f}, ours {:.3f}) -> no emergent "
                "sidestepping: **{}**\n".format(
                    diagnostics["max_sweep_lat_vel"], PI_NOM_LAT_VEL_REF,
                    M.OURS_MEAN_LAT_VEL,
                    diagnostics["no_emergent_sidestepping"]))
        if qsafe_report:
            b = qsafe_report.get("battery", {})
            f.write("- Q-hat battery: rho = {:.3f}, AUROC = {:.3f}, "
                    "baseline AUROC = {:.3f}\n".format(
                        b.get("spearman_rho_q", float("nan")),
                        b.get("auroc_q_collision20", float("nan")),
                        b.get("auroc_min_cbf_h_baseline", float("nan"))))
        f.write("\n## Rebuttal sentence (analysis_protocol.md §7, `{}`)\n\n> {}\n"
                .format(key, sentence))

    with open(args.out + ".tex", "w") as f:
        f.write(render_latex(sweep, [unf_row] + others))

    blob = {"rows": {k: _jsonable(v) for k, v in rows.items()},
            "sanity_checks": [{"name": n, "passed": p, "detail": d}
                              for n, p, d in sanity],
            "verdict": verdict, "diagnostics": diagnostics,
            "rebuttal_key": key, "rebuttal_sentence": sentence,
            "qsafe_report": qsafe_report}
    with open(args.out + ".json", "w") as f:
        json.dump(blob, f, indent=2)

    if not args.no_figure:
        make_figure(sweep, unf_row, args.out + "_fig")

    print(md)
    print("\n--- sanity checks ---")
    for name, ok, det in sanity:
        print(" [{}] {}: {}".format("PASS" if ok else "FAIL", name, det))
    print("\n--- verdict ---")
    for k, v in verdict["checks"].items():
        print("  {}: {}".format(k, {True: "HELD", False: "FAILED",
                                    None: "N/A"}[v]))
    print("  primary claim: {}".format(
        "HOLDS" if verdict["holds"] else
        ("FALSIFIED" if verdict["falsified"] else "PARTIAL")))
    print("\n--- §7 sentence ({}) ---\n{}".format(key, sentence))
    print("\nwrote {}.md / .tex / .json".format(args.out))


def _jsonable(row):
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in row.items()}


if __name__ == "__main__":
    main()
