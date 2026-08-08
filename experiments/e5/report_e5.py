"""E5 results, compiled into one self-contained HTML report.

Gathers everything under a sweep directory -- per-cell CSVs, manifests,
per-step npz series -- plus each arm's training record (policy_config.json),
recomputes the tables/calibration/verdict through analyze_e5's functions (so
the report can never disagree with the analyzer), and writes ONE .html file
with the figures embedded as base64.  No server, no external assets: open it
in any browser, mail it around, drop it in the paper's supplementary.

Works with however many arms exist -- a pi_nom-only sweep renders the pi_nom
story (health at zero push, the falls-vs-magnitude ladder, calibration of the
diagnostically-fitted critic); rows for other arms appear as their cells are
produced.

Usage
-----
    python experiments/e5/report_e5.py --dir logs/e5/sweep --out logs/e5/report.html
    # then open logs/e5/report.html in a browser

matplotlib and numpy are optional: without matplotlib the report has tables
but no figures, without numpy the calibration section reports itself skipped.
"""

import argparse
import base64
import glob
import html as html_mod
import json
import os
import subprocess
import sys
import tempfile
import time

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, CURDIR)

import analyze_e5 as A                                          # noqa: E402
from safeloco_eval import metrics as M                          # noqa: E402
from safeloco_eval import stats as ST                           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="logs/e5/sweep")
    p.add_argument("--out", type=str, default="logs/e5/report.html")
    p.add_argument("--train_log_root", type=str,
                   default="logs/cassie_viploco_warp_v2",
                   help="where the arms' policy_config.json records live")
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--dt", type=float, default=None)
    return p.parse_args()


def esc(x):
    return html_mod.escape(str(x))


def fmt(x, spec="{:.3f}", dash_on=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "&ndash;"
    if v != v or (dash_on is not None and v == dash_on):
        return "&ndash;"
    return spec.format(v)


# ------------------------------------------------------------ gathering ----

def training_records(train_log_root, arms):
    """arm -> policy_config.json contents, matched by run_name."""
    out = {}
    for path in glob.glob(os.path.join(train_log_root, "*",
                                       "policy_config.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        name = rec.get("run_name") or os.path.basename(os.path.dirname(path))
        if name in arms or rec.get("policy") in arms:
            out[name] = rec
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# -------------------------------------------------------------- tables -----

def coverage_table(runs):
    """arms x (terrain, vx, mag) grid with the episode count per cell."""
    arms, cells, count = set(), set(), {}
    for run in runs.values():
        arm, vx, mag, terrain = A.key_of(run)
        arms.add(arm)
        cell = (terrain, vx, mag)
        cells.add(cell)
        count[(arm, cell)] = len(run["records"])
    arms = sorted(arms)
    cells = sorted(cells)
    head = "".join("<th>{} vx={:g} m={:g}</th>".format(esc(t), v, m)
                   for t, v, m in cells)
    body = []
    for arm in arms:
        tds = []
        for cell in cells:
            n = count.get((arm, cell))
            tds.append("<td class='{}'>{}</td>".format(
                "ok" if n else "miss", n if n else "&ndash;"))
        body.append("<tr><th>{}</th>{}</tr>".format(esc(arm), "".join(tds)))
    return ("<table><tr><th>arm</th>{}</tr>{}</table>"
            .format(head, "".join(body)))


def pivot_table(rows, vx=A.HEADLINE_VX, terrain="flat"):
    """Fall %% (Wilson CI) by arm x push magnitude -- the headline grid.

    Base cells only: set-mode / custom-dp variant runs share the (arm, mag)
    key and would silently overwrite the base entry.
    """
    sel = [r for r in rows if r["cmd_vx"] == vx and r["terrain"] == terrain
           and A.is_base_cell(r)]
    if not sel:
        return "<p class='note'>no cells at vx={:g} on {}</p>".format(
            vx, terrain)
    arms = sorted({r["arm"] for r in sel})
    mags = sorted({r["push_mag"] for r in sel})
    by = {(r["arm"], r["push_mag"]): r for r in sel}
    head = "".join("<th>mag {:g}</th>".format(m) for m in mags)
    body = []
    for arm in arms:
        tds = []
        for m in mags:
            r = by.get((arm, m))
            if r is None:
                tds.append("<td class='miss'>&ndash;</td>")
            else:
                tds.append(
                    "<td>{:.1%} <span class='ci'>[{:.1%}, {:.1%}]</span>"
                    "<br><span class='sub'>{:.2f}/1k steps</span></td>"
                    .format(r["fall_rate"], r["fall_lo"], r["fall_hi"],
                            r["falls_per_1k"]))
        body.append("<tr><th>{}</th>{}</tr>".format(esc(arm), "".join(tds)))
    return ("<table><tr><th>fall rate (vx={:g}, {})</th>{}</tr>{}</table>"
            .format(vx, esc(terrain), head, "".join(body)))


def r5a_table(rows):
    cols = [("arm", None), ("cmd_vx", "{:g}"), ("push_mag", "{:g}"),
            ("terrain", None), ("n_eps", "{:d}")]
    head = ("<tr><th>run</th><th>arm</th><th>vx</th><th>mag</th>"
            "<th>terrain</th>"
            "<th>eps</th><th>fall % (CI)</th><th>falls/1k (CI)</th>"
            "<th>ttf s (n | cens)</th><th>push-attr</th><th>vel err</th>"
            "<th>return</th><th>mean &#8467;</th><th>min &#8467;</th>"
            "<th>mean V<sub>safe</sub></th></tr>")
    body = []
    for r in rows:
        cells = ["<td style='text-align:left'>{}</td>".format(
            esc(r["run_id"]))]
        for key, spec in cols:
            v = r[key]
            cells.append("<td>{}</td>".format(
                esc(v) if spec is None else spec.format(
                    int(v) if spec == "{:d}" else v)))
        cells.append("<td>{:.1%} <span class='ci'>[{:.1%}, {:.1%}]</span></td>"
                     .format(r["fall_rate"], r["fall_lo"], r["fall_hi"]))
        cells.append("<td>{:.2f} <span class='ci'>[{:.2f}, {:.2f}]</span></td>"
                     .format(r["falls_per_1k"], r["f1k_lo"], r["f1k_hi"]))
        cells.append("<td>{} ({} | {})</td>".format(
            fmt(r["time_to_fall_s"], "{:.2f}"), r["ttf_n"], r["n_censored"]))
        cells.append("<td>{}</td>".format(fmt(r["push_attr"], "{:.1%}")))
        cells.append("<td>{:.3f}</td>".format(r["vel_err"]))
        cells.append("<td>{:.1f} &plusmn; {:.1f}</td>".format(
            r["return"], r["return_std"]))
        cells.append("<td>{:+.3f}</td>".format(r["mean_margin"]))
        cells.append("<td>{:+.3f}</td>".format(r["min_margin"]))
        cells.append("<td>{:+.3f}{}</td>".format(
            r["mean_vsafe"], "" if r["vsafe_trained"]
            else " <span class='warn'>(untrained)</span>"))
        body.append("<tr>{}</tr>".format("".join(cells)))
    return "<table>{}{}</table>".format(head, "".join(body))


def calibration_table(calib):
    if not calib:
        return ("<p class='note'>no calibration series (needs push cells "
                "with their .npz files, and numpy)</p>")
    horizons = sorted({h for c in calib.values() for h in c["auc"]})
    head = ("<tr><th>run</th>"
            + "".join("<th>AUC@{:g}s<br>V<sub>safe</sub> / &#8467;</th>"
                      .format(h) for h in horizons)
            + "<th>lead V<sub>safe</sub></th><th>lead &#8467;</th>"
              "<th>&Delta; lead</th><th>falls</th></tr>")
    body = []
    for rid, c in sorted(calib.items()):
        tds = ["<td>{}</td>".format(esc(rid))]
        for h in horizons:
            a = c["auc"].get(h)
            if a is None:
                tds.append("<td>&ndash;</td>")
            else:
                better = a["vsafe"] > a["margin"]
                tds.append("<td class='{}'>{:.3f} / {:.3f}</td>".format(
                    "good" if better else "bad", a["vsafe"], a["margin"]))
        dl = c["lead_vsafe_s"] - c["lead_margin_s"]
        tds.append("<td>{} s</td>".format(fmt(c["lead_vsafe_s"], "{:.2f}")))
        tds.append("<td>{} s</td>".format(fmt(c["lead_margin_s"], "{:.2f}")))
        tds.append("<td class='{}'>{} s</td>".format(
            "good" if dl > 0 else "bad", fmt(dl, "{:+.2f}")))
        tds.append("<td>{}</td>".format(c["n_falls_lead"]))
        body.append("<tr>{}</tr>".format("".join(tds)))
    return "<table>{}{}</table>".format(head, "".join(body))


def training_table(trained):
    if not trained:
        return ("<p class='note'>no policy_config.json found under the "
                "training log root; pass --train_log_root</p>")
    keys = ["policy", "safety_coef", "rewards.scales.safety_value",
            "train_safety_critic_when_off", "fall_safety.mode",
            "safety_return_alpha", "d_safe", "d_danger", "max_iterations",
            "num_envs", "max_push_vel_xy", "terrain_proportions", "seed"]
    head = ("<tr><th>run</th>"
            + "".join("<th>{}</th>".format(esc(k)) for k in keys) + "</tr>")
    body = []
    for name, rec in sorted(trained.items()):
        tds = ["<td>{}</td>".format(esc(name))]
        for k in keys:
            tds.append("<td>{}</td>".format(
                esc(rec[k]) if k in rec else "&ndash;"))
        body.append("<tr>{}</tr>".format("".join(tds)))
    return "<table>{}{}</table>".format(head, "".join(body))


def paired_table(pairs):
    if not pairs:
        return ("<p class='note'>paired comparison needs both pi_nom and "
                "pi_ours cells on matched seeds</p>")
    head = ("<tr><th>vx</th><th>mag</th><th>terrain</th><th>pairs</th>"
            "<th>pi_nom falls</th><th>pi_ours falls</th>"
            "<th>discordant</th><th>p</th></tr>")
    body = []
    for p in pairs:
        body.append(
            "<tr><td>{vx:g}</td><td>{mag:g}</td><td>{t}</td><td>{n}</td>"
            "<td>{ra:.1%}</td><td>{rb:.1%}</td><td>{d}</td><td>{p:.4f}</td>"
            "</tr>".format(vx=p["vx"], mag=p["mag"], t=esc(p["terrain"]),
                           n=p["n_pairs"], ra=p["rate_a"], rb=p["rate_b"],
                           d=p["discordant"], p=p["p_value"]))
    return "<table>{}{}</table>".format(head, "".join(body))


def embed_figures(rows, calib):
    imgs = []
    with tempfile.TemporaryDirectory() as td:
        A.figures(rows, calib, td)
        for png in sorted(glob.glob(os.path.join(td, "*.png"))):
            with open(png, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            imgs.append("<figure><img src='data:image/png;base64,{}'/>"
                        "<figcaption>{}</figcaption></figure>".format(
                            b64, esc(os.path.basename(png))))
    return ("".join(imgs) if imgs
            else "<p class='note'>no figures (matplotlib unavailable "
                 "or no plottable cells)</p>")


CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
       margin: 2em auto; max-width: 1200px; color: #1a1a1a; }
h1 { border-bottom: 2px solid #333; padding-bottom: .2em; }
h2 { margin-top: 1.6em; border-bottom: 1px solid #ccc; padding-bottom: .15em; }
table { border-collapse: collapse; margin: .8em 0; font-size: 13px; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: right; }
th { background: #f0f0f0; }
td.ok { background: #e8f5e9; } td.miss { background: #fafafa; color: #999; }
td.good { background: #e8f5e9; } td.bad { background: #fdecea; }
.ci { color: #777; font-size: 11px; } .sub { color: #555; font-size: 11px; }
.warn { color: #b26a00; font-size: 11px; }
.note { color: #777; font-style: italic; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: .8em;
      overflow-x: auto; font-size: 12px; }
figure { margin: 1em 0; } figure img { max-width: 100%; border: 1px solid #ddd; }
figcaption { color: #777; font-size: 12px; }
.meta { color: #555; font-size: 13px; }
"""


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    runs = A.load_runs(paths)
    if not runs:
        raise SystemExit("no CSVs found under {}".format(args.dir))

    dt = args.dt
    if dt is None:
        for run in runs.values():
            if "control_dt" in run["manifest"]:
                dt = float(run["manifest"]["control_dt"])
                break
        dt = dt or M.CONTROL_DT

    rows = A.table_r5a(runs)
    msgs = A.sanity(runs)
    arms_early = sorted({r["arm"] for r in rows})
    trained = training_records(args.train_log_root, set(arms_early))
    # The eval manifest's vsafe_trained flag is sourced from the *eval-time*
    # config; the training record is the truth.  Override where it exists.
    for r in rows:
        rec = trained.get(r["arm"])
        if rec is None:
            rec = next((v for v in trained.values()
                        if v.get("policy") == r["arm"]), None)
        if rec is not None:
            r["vsafe_trained"] = bool(
                float(rec.get("safety_coef", 0) or 0) > 0
                or rec.get("train_safety_critic_when_off"))
    calib = {}
    for rid, run in runs.items():
        _, _, mag, _ = A.key_of(run)
        if mag <= 0:
            continue
        try:
            c = A.calibration_for_run(run, dt)
        except ImportError:
            c = None
        if c is not None:
            calib[rid] = c
    pairs = A.paired_falls(runs, "pi_nom", "pi_ours")
    verdict = A.verdict(rows, calib, pairs)
    arms = arms_early

    sanity_html = ("<ul>" + "".join(
        "<li class='{}'>{}</li>".format(
            "bad" if m.startswith("FAIL") else "", esc(m)) for m in msgs)
        + "</ul>") if msgs else "<p class='note'>all clear</p>"

    figures_html = ("<p class='note'>figures disabled</p>"
                    if args.no_figures else embed_figures(rows, calib))

    doc = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>E5 report</title>
<style>{css}</style></head><body>
<h1>E5 &mdash; bipedal fall safety: results</h1>
<p class="meta">generated {ts} &middot; sweep dir <code>{dirp}</code>
&middot; {ncells} cell(s), arms: {arms} &middot; git {commit} &middot;
dt {dt:g}s</p>

<h2>Cell coverage</h2>
{coverage}

<h2>Headline: fall rate by arm &times; push magnitude</h2>
{pivot}

<h2>Figures</h2>
{figures}

<h2>Table R5a (all cells)</h2>
{r5a}
<p class="note">ttf = mean time-to-fall over fallen episodes (n fallen |
censored-by-timeout count); falls/1k steps is the censoring-robust rate.
&#8467; is the fail-set fall margin; V<sub>safe</sub> the reachability
critic's value on the same states.</p>

<h2>Fall-prediction calibration (dynamic vs static)</h2>
{calib}
<p class="note">AUC pairs are V<sub>safe</sub> / static margin for "falls
within H seconds" on that run's own rollouts; lead = median time before the
fall at which each signal first crossed zero.  Green means the learned
reachability value out-predicts the static margin &mdash; the dynamic-safety
claim, per run.</p>

<h2>Paired comparison (McNemar on matched starts)</h2>
{paired}

<h2>Sanity checks</h2>
{sanity}

<h2>Registered predictions (mechanical verdict)</h2>
<pre>{verdict}</pre>

<h2>Training configuration per arm</h2>
{training}
</body></html>""".format(
        css=CSS, ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        dirp=esc(args.dir), ncells=len(runs), arms=esc(", ".join(arms)),
        commit=esc(git_commit()), dt=dt,
        coverage=coverage_table(runs), pivot=pivot_table(rows),
        figures=figures_html, r5a=r5a_table(rows),
        calib=calibration_table(calib), paired=paired_table(pairs),
        sanity=sanity_html, verdict=esc(verdict),
        training=training_table(trained))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(doc)
    print("wrote {} ({} cells, arms: {})".format(
        args.out, len(runs), ", ".join(arms)))


if __name__ == "__main__":
    main()
