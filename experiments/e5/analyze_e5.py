"""E5 analysis: per-episode CSVs + per-step npz -> Table R5a, calibration, verdict.

Order follows analyze_e1.py (analysis_protocol.md):
  sanity checks first; metric definitions from safeloco_eval/metrics.py only;
  Wilson CIs on rates, episode-clustered bootstrap on pooled step rates;
  the registered predictions (experiments/e5/README.md) applied mechanically.

The calibration section is the reviewer-facing headline: does the learned
reachability value V_safe predict falls that the *static* fail-set margin
cannot yet see?  Quantified two ways, both from the per-step npz series:
  * AUC of "falls within the next H seconds" for score -V_safe vs score
    -l_fall, pooled over live steps of episodes matched to the CSV records;
  * warning lead time: on fallen episodes, how long before the fall each
    signal first crossed zero (V_safe crossing earlier = dynamic prediction).

Usage
-----
    python experiments/e5/analyze_e5.py --dir logs/e5/sweep --out logs/e5/R5
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from safeloco_eval import metrics as M
from safeloco_eval import stats as ST

CALIB_HORIZONS_S = (0.5, 1.0, 2.0)
HEADLINE_VX = 0.8
PRED_MAG_FLOOR = 1.0        # prediction (i) applies at mags >= this
PRED_TRACKING_COST = 0.1    # prediction (iv): vel_err cost bound [m/s]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e5/sweep")
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--out", type=str, default="logs/e5/R5")
    p.add_argument("--no_figure", action="store_true")
    p.add_argument("--dt", type=float, default=None,
                   help="control dt; default from the first manifest")
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
        man = {}
        man_path = path[:-4] + ".manifest.json"
        if os.path.exists(man_path):
            with open(man_path) as f:
                man = json.load(f)
        npz_path = path[:-4] + ".npz"
        runs[run_id] = {"records": recs, "manifest": man, "csv": path,
                        "npz": npz_path if os.path.exists(npz_path) else None}
    return runs


def key_of(run):
    man = run["manifest"]
    rec = run["records"][0]
    arm = man.get("arm") or rec.get("variant") or "?"
    vx = float(man.get("cmd_vx", rec.get("cmd_vx", 0) or 0))
    mag = float(man.get("push_mag", rec.get("push_mag", 0) or 0))
    terrain = man.get("terrain", rec.get("terrain", "?"))
    return arm, vx, mag, terrain


def is_base_cell(r):
    """True for cells run with the default push mechanics.

    Variant cells (--push_mode set, --push_dp != 150) share an (arm, vx, mag)
    key with their base cell; anything that groups by that key must exclude
    them or the variant silently overwrites the base row.
    """
    mode = r.get("push_mode", "add") or "add"
    try:
        dp = int(float(r.get("push_dp", 150) or 150))
    except (TypeError, ValueError):
        dp = 150
    return mode == "add" and dp == 150


# ------------------------------------------------------------- sanity (§5) --

def sanity(runs):
    """Each of these has caught real bugs in this class of study.  A FAIL
    stops nothing -- it prints, and the tables carry the taint marker."""
    msgs = []

    for rid, run in runs.items():
        arm, vx, mag, terrain = key_of(run)
        pushes = [int(float(r.get("n_pushes", 0) or 0)) for r in run["records"]]
        if mag <= 0 and any(pushes):
            msgs.append("FAIL  {}: push_mag=0 but n_pushes>0".format(rid))
        if mag > 0 and not any(pushes):
            msgs.append("FAIL  {}: push_mag={} but no episode was pushed "
                        "(all episodes ended before push_p0?)".format(rid, mag))
        n = len(run["records"])
        if n < 100:
            msgs.append("WARN  {}: only {} episodes".format(rid, n))

    # Fall rate should be non-decreasing in push magnitude per (arm, vx,
    # terrain).  Non-monotonicity is usually a seed-protocol or push-window
    # bug, not physics.
    series = {}
    for rid, run in runs.items():
        arm, vx, mag, terrain = key_of(run)
        k, n = M.fall_rate(run["records"])
        series.setdefault((arm, vx, terrain), []).append((mag, k / max(1, n)))
    for skey, pts in series.items():
        pts.sort()
        for (m0, r0), (m1, r1) in zip(pts, pts[1:]):
            if r1 < r0 - 0.03:   # allow 3pp of noise
                msgs.append("WARN  {}: fall rate not monotone in push mag "
                            "({}: {:.3f} -> {}: {:.3f})".format(
                                skey, m0, r0, m1, r1))

    return msgs


# ------------------------------------------------------------- Table R5a ----

def table_r5a(runs):
    rows = []
    for rid, run in sorted(runs.items(), key=lambda kv: key_of(kv[1])):
        arm, vx, mag, terrain = key_of(run)
        recs = M.attach_fall_fields(run["records"])
        n = len(recs)
        k_fall, _ = M.fall_rate(recs)
        f_pt, f_lo, f_hi = ST.wilson_ci(k_fall, n)
        r_pt, r_lo, r_hi = ST.cluster_bootstrap_rate_ci(
            recs, "fell_count", "ksteps")
        ttf, n_cens = M.time_to_fall_values(recs)
        pk, pn = M.push_attributed_fall_rate(recs)
        rows.append({
            "run_id": rid, "arm": arm, "cmd_vx": vx, "push_mag": mag,
            "terrain": terrain, "n_eps": n,
            "push_mode": run["manifest"].get("push_mode", "add"),
            "push_dp": run["manifest"].get("push_dp", 150),
            "fall_rate": f_pt, "fall_lo": f_lo, "fall_hi": f_hi,
            "falls_per_1k": r_pt, "f1k_lo": r_lo, "f1k_hi": r_hi,
            "time_to_fall_s": (ST.mean(ttf) if ttf else float("nan")),
            "ttf_n": len(ttf), "n_censored": n_cens,
            "push_attr": (pk / pn if pn else float("nan")),
            "vel_err": ST.mean(M.per_episode(recs, "vel_err")),
            "return": ST.mean(M.per_episode(recs, "return")),
            "return_std": ST.std(M.per_episode(recs, "return"))
            if n > 1 else 0.0,
            "mean_margin": ST.mean(M.per_episode(recs, "mean_fall_margin")),
            "min_margin": M.min_over_runs(recs, "min_fall_margin"),
            "mean_vsafe": ST.mean(M.per_episode(recs, "mean_vsafe")),
            "vsafe_trained": bool(run["manifest"].get("vsafe_trained", False)),
        })
    return rows


def fmt_table(rows):
    hdr = ("| arm | vx | mag | terrain | eps | fall % (95% CI) | "
           "falls/1k steps (CI) | ttf s (n) | push-attr | vel err | "
           "return | mean l | min l | mean V_safe |")
    sep = "|" + "---|" * 14
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            "| {arm} | {cmd_vx:g} | {push_mag:g} | {terrain} | {n_eps} "
            "| {fp} | {f1k} | {ttf} | {pa} | {ve:.3f} | "
            "{ret:.1f}±{rs:.1f} | {ml:+.3f} | {mnl:+.3f} | {mv} |".format(
                fp=ST.fmt_rate(r["fall_rate"], r["fall_lo"], r["fall_hi"]),
                f1k="{:.2f} [{:.2f},{:.2f}]".format(
                    r["falls_per_1k"], r["f1k_lo"], r["f1k_hi"]),
                ttf=("{:.2f} ({})".format(r["time_to_fall_s"], r["ttf_n"])
                     if r["ttf_n"] else "--"),
                pa=("{:.1%}".format(r["push_attr"])
                    if not math.isnan(r["push_attr"]) else "--"),
                ve=r["vel_err"], ret=r["return"], rs=r["return_std"],
                ml=r["mean_margin"], mnl=r["min_margin"],
                mv=("{:+.3f}".format(r["mean_vsafe"])
                    + ("" if r["vsafe_trained"] else " (untrained)")),
                **r))
    return "\n".join(lines)


# ------------------------------------------------------ paired comparison ---

def paired_falls(runs, arm_a, arm_b):
    """McNemar over matched (chunk, env) starts, per (vx, mag, terrain)."""
    by_key = {}
    for rid, run in runs.items():
        arm, vx, mag, terrain = key_of(run)
        by_key.setdefault((vx, mag, terrain), {})[arm] = run["records"]
    out = []
    for (vx, mag, terrain), arms in sorted(by_key.items()):
        if arm_a in arms and arm_b in arms:
            res = ST.mcnemar_paired(arms[arm_a], arms[arm_b])
            res.update(vx=vx, mag=mag, terrain=terrain,
                       arm_a=arm_a, arm_b=arm_b)
            out.append(res)
    return out


# ----------------------------------------------------------- calibration ----

def _first_neg(series_col, t_done):
    for t in range(t_done + 1):
        if series_col[t] < 0:
            return t
    return -1


def calibration_for_run(run, dt, horizons_s=CALIB_HORIZONS_S):
    """AUC + lead times from the per-step npz, joined to the CSV records.

    Only envs present in the records (the kept first episodes) contribute,
    so the labels match the table rows exactly.
    """
    if run["npz"] is None:
        return None
    import numpy as np
    blob = np.load(run["npz"])

    kept = {}
    for r in run["records"]:
        kept[(int(r["chunk_idx"]), int(r["env_idx"]))] = bool(int(r["fell"]))

    chunk_ids = sorted({int(k.split("_")[0][1:]) for k in blob.files})
    per_h = {h: {"scores_v": [], "scores_m": [], "labels": []}
             for h in horizons_s}
    lead_v, lead_m = [], []
    n_v_never, n_m_never = 0, 0

    for ci in chunk_ids:
        try:
            vsafe = blob["c{}_vsafe".format(ci)]
            margin = blob["c{}_margin".format(ci)]
            live = blob["c{}_live".format(ci)].astype(bool)
            done = blob["c{}_done".format(ci)].astype(bool)
        except KeyError:
            continue
        T, E = vsafe.shape
        for e in range(E):
            if (ci, e) not in kept:
                continue
            fell = kept[(ci, e)]
            done_col = done[:, e] & live[:, e]
            ts = done_col.nonzero()[0]
            if len(ts) == 0:
                continue
            t_done = int(ts[0])
            for h in horizons_s:
                hs = int(round(h / dt))
                for t in range(t_done + 1):
                    per_h[h]["scores_v"].append(-float(vsafe[t, e]))
                    per_h[h]["scores_m"].append(-float(margin[t, e]))
                    per_h[h]["labels"].append(
                        1 if (fell and (t_done - t) <= hs) else 0)
            if fell:
                tv = _first_neg(vsafe[:, e], t_done)
                tm = _first_neg(margin[:, e], t_done)
                if tv >= 0:
                    lead_v.append((t_done - tv) * dt)
                else:
                    n_v_never += 1
                if tm >= 0:
                    lead_m.append((t_done - tm) * dt)
                else:
                    n_m_never += 1

    def med(xs):
        if not xs:
            return float("nan")
        s = sorted(xs)
        return s[len(s) // 2]

    out = {"auc": {}, "lead_vsafe_s": med(lead_v), "lead_margin_s": med(lead_m),
           "n_falls_lead": max(len(lead_v), len(lead_m)),
           "n_vsafe_never_neg": n_v_never, "n_margin_never_neg": n_m_never}
    for h in horizons_s:
        d = per_h[h]
        out["auc"][h] = {
            "vsafe": ST.auroc(d["scores_v"], d["labels"]),
            "margin": ST.auroc(d["scores_m"], d["labels"]),
            "n_pos": sum(d["labels"]), "n": len(d["labels"]),
        }
    return out


# ------------------------------------------------------------ predictions ---

def verdict(rows, calib, pairs):
    """The registered predictions, applied mechanically.

    Amended 2026-08-08 after the pi_nom envelope mapping (before any
    comparison arm was trained): the arena moved from flat+push to
    stair+push, with the flat magnitude ladder kept as the boundary-shift
    prediction (v).  Original rules are in git history.
    """
    lines = []
    base = [r for r in rows if is_base_cell(r)]

    def cells(arm, terrain):
        return {(r["cmd_vx"], r["push_mag"]): r for r in base
                if r["arm"] == arm and r["terrain"] == terrain}

    nom_s, ours_s, rs_s = (cells("pi_nom", "stair"), cells("pi_ours", "stair"),
                           cells("pi_rs", "stair"))
    nom_f, ours_f = cells("pi_nom", "flat"), cells("pi_ours", "flat")

    # (i) stair + push: ours falls/1k < nom with non-overlapping CIs
    checked = held = 0
    for k in sorted(set(nom_s) & set(ours_s)):
        vx, mag = k
        if mag <= 0:
            continue
        checked += 1
        a, b = ours_s[k], nom_s[k]
        sep = a["f1k_hi"] < b["f1k_lo"]
        ok = a["falls_per_1k"] < b["falls_per_1k"] and sep
        if ok:
            held += 1
        lines.append("  (i) stair vx={} mag={}: ours {:.2f} [{:.2f},{:.2f}] "
                     "vs nom {:.2f} [{:.2f},{:.2f}] -> {}".format(
                         vx, mag, a["falls_per_1k"], a["f1k_lo"], a["f1k_hi"],
                         b["falls_per_1k"], b["f1k_lo"], b["f1k_hi"],
                         "HELD" if ok else "not separated"))
    lines.insert(0, "(i)   ours < nom on stair+push with separated CIs: "
                 "{}/{} cells".format(held, checked))

    # (ii) pi_rs between, on the stair+push cells
    n_between = n_cells = 0
    for k in sorted(set(nom_s) & set(ours_s) & set(rs_s)):
        vx, mag = k
        if mag <= 0:
            continue
        n_cells += 1
        lo = min(ours_s[k]["falls_per_1k"], nom_s[k]["falls_per_1k"])
        hi = max(ours_s[k]["falls_per_1k"], nom_s[k]["falls_per_1k"])
        if lo <= rs_s[k]["falls_per_1k"] <= hi:
            n_between += 1
    lines.append("(ii)  pi_rs between (stair+push): {}/{} cells".format(
        n_between, n_cells))

    # (iii) calibration: V_safe AUC > margin AUC at H=1 s, positive lead diff
    if calib:
        h = 1.0
        aucs = [(rid, c["auc"][h]["vsafe"], c["auc"][h]["margin"],
                 c["lead_vsafe_s"], c["lead_margin_s"])
                for rid, c in calib.items() if h in c["auc"]]
        ok = sum(1 for _, av, am, lv, lm in aucs
                 if av > am and (lv - lm) > 0)
        lines.append("(iii) V_safe beats static margin (AUC@1s and lead): "
                     "{}/{} runs".format(ok, len(aucs)))
        for rid, av, am, lv, lm in aucs:
            lines.append("      {}: AUC {:.3f} vs {:.3f}; lead {:.2f}s vs "
                         "{:.2f}s".format(rid, av, am, lv, lm))
    else:
        lines.append("(iii) no npz series found -- calibration not evaluated")

    # (iv) tracking cost at mag 0, on flat AND stair
    k0 = (HEADLINE_VX, 0.0)
    for terrain, nom_c, ours_c in (("flat", nom_f, ours_f),
                                   ("stair", nom_s, ours_s)):
        if k0 in nom_c and k0 in ours_c:
            cost = ours_c[k0]["vel_err"] - nom_c[k0]["vel_err"]
            lines.append("(iv)  tracking cost at mag 0 ({}): {:+.3f} m/s "
                         "(bound {}) -> {}".format(
                             terrain, cost, PRED_TRACKING_COST,
                             "HELD" if cost < PRED_TRACKING_COST
                             else "EXCEEDED"))

    # (v) boundary shift: flat cells at mag >= 2.5, ours < nom separated CIs
    checked = held = 0
    for k in sorted(set(nom_f) & set(ours_f)):
        vx, mag = k
        if vx != HEADLINE_VX or mag < 2.5:
            continue
        checked += 1
        a, b = ours_f[k], nom_f[k]
        sep = a["f1k_hi"] < b["f1k_lo"]
        ok = a["falls_per_1k"] < b["falls_per_1k"] and sep
        if ok:
            held += 1
        lines.append("  (v) flat vx={} mag={}: ours {:.2f} [{:.2f},{:.2f}] "
                     "vs nom {:.2f} [{:.2f},{:.2f}] -> {}".format(
                         vx, mag, a["falls_per_1k"], a["f1k_lo"], a["f1k_hi"],
                         b["falls_per_1k"], b["f1k_lo"], b["f1k_hi"],
                         "HELD" if ok else "not separated"))
    lines.append("(v)   boundary shift on flat at mag >= 2.5: {}/{} cells"
                 .format(held, checked))

    if pairs:
        lines.append("paired (McNemar, pi_nom vs pi_ours):")
        for p in pairs:
            lines.append("      vx={} mag={} {}: nom {:.1%} vs ours {:.1%} "
                         "(discordant {}, p={:.4f})".format(
                             p["vx"], p["mag"], p["terrain"], p["rate_a"],
                             p["rate_b"], p["discordant"], p["p_value"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- figures ---

def figures(rows, calib, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib unavailable; skipping figures")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Fall rate vs push magnitude, per arm (flat, headline vx).
    fig, ax = plt.subplots(figsize=(5, 3.4))
    base = [r for r in rows if is_base_cell(r)]
    arms = sorted({r["arm"] for r in base})
    for arm in arms:
        pts = sorted((r["push_mag"], r["fall_rate"], r["fall_lo"],
                      r["fall_hi"]) for r in base
                     if r["arm"] == arm and r["terrain"] == "flat"
                     and r["cmd_vx"] == HEADLINE_VX)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        # max(0, .) guards zero-fall / all-fall cells, where floating point
        # can put a CI bound an ulp past the point estimate and matplotlib
        # refuses negative error bars.
        lo = [max(0.0, 100 * (p[1] - p[2])) for p in pts]
        hi = [max(0.0, 100 * (p[3] - p[1])) for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=arm)
    ax.set_xlabel("push magnitude [m/s]")
    ax.set_ylabel("fall rate [% episodes]")
    ax.set_title("Falls vs push magnitude (vx={} m/s, flat)".format(
        HEADLINE_VX))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "falls_vs_push.png"), dpi=160)
    plt.close(fig)

    # AUC vs horizon per run with npz.
    if calib:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for rid, c in sorted(calib.items()):
            hs = sorted(c["auc"])
            ax.plot(hs, [c["auc"][h]["vsafe"] for h in hs], marker="o",
                    label="{} V_safe".format(rid))
            ax.plot(hs, [c["auc"][h]["margin"] for h in hs], marker="x",
                    linestyle="--", label="{} static l".format(rid))
        ax.set_xlabel("prediction horizon H [s]")
        ax.set_ylabel("AUC (fall within H)")
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "calibration_auc.png"), dpi=160)
        plt.close(fig)


# ------------------------------------------------------------------- main ---

def main():
    args = parse_args()
    paths = args.files or sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    runs = load_runs(paths)
    if not runs:
        raise SystemExit("no CSVs found under {}".format(args.dir))
    os.makedirs(args.out, exist_ok=True)

    dt = args.dt
    if dt is None:
        for run in runs.values():
            if "control_dt" in run["manifest"]:
                dt = float(run["manifest"]["control_dt"])
                break
        dt = dt or M.CONTROL_DT

    print("== sanity (§5) ==")
    msgs = sanity(runs)
    print("\n".join(msgs) if msgs else "  all clear")

    rows = table_r5a(runs)
    table = fmt_table(rows)
    print("\n== Table R5a ==")
    print(table)

    calib = {}
    for rid, run in runs.items():
        arm, vx, mag, terrain = key_of(run)
        if mag <= 0:
            continue          # calibration needs fall events; pushes make them
        c = calibration_for_run(run, dt)
        if c is not None:
            calib[rid] = c

    pairs = paired_falls(runs, "pi_nom", "pi_ours")

    print("\n== registered predictions ==")
    v = verdict(rows, calib, pairs)
    print(v)

    with open(os.path.join(args.out, "table_r5a.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"rows": rows, "calibration": calib,
                   "paired": pairs, "sanity": msgs, "dt": dt},
                  f, indent=2, default=str)
    with open(os.path.join(args.out, "verdict.txt"), "w") as f:
        f.write(("\n".join(msgs) + "\n\n" if msgs else "") + v + "\n")

    if not args.no_figure:
        figures(rows, calib, args.out)
    print("\nwrote {}".format(args.out))


if __name__ == "__main__":
    main()
