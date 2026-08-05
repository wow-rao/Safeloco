"""Offline tests for the E1 statistics, metric definitions, verdict rule and
analysis pipeline.

Runs with the standard library only -- no IsaacGym, no torch, no GPU -- so the
part of E1 that turns numbers into a *claim* can be checked before any sweep
burns GPU time.  What it covers:

  * stats.py against hand-computed values (Wilson, Spearman, AUROC, bootstrap)
  * the single fall definition and the pooled-rate conventions
  * the §5 sanity checks, including a deliberately broken trust region
  * the §3.1 verdict rule on three synthetic worlds: verdict-holds,
    verdict-falsified, and Q-hat-failed-validation
  * end-to-end analyze_e1.py on synthetic CSVs

Run:  python tests/test_e1_offline.py
"""

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from safeloco_eval import metrics as M
from safeloco_eval import stats as ST
from safeloco_eval import seeds as S

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        FAILURES.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- stats -----

def test_stats():
    print("\n[stats]")
    # Wilson at k=0 must not produce a negative lower bound or a zero-width
    # interval -- that is exactly why the protocol prefers it near 0.
    p, lo, hi = ST.wilson_ci(0, 1000)
    check("wilson k=0 point", approx(p, 0.0))
    check("wilson k=0 lower bound is 0", approx(lo, 0.0))
    check("wilson k=0 upper bound positive", hi > 0.003 and hi < 0.005,
          "hi={}".format(hi))

    # Reference value: 44/1000 -> Wilson 95% CI is approximately [3.3%, 5.9%].
    p, lo, hi = ST.wilson_ci(44, 1000)
    check("wilson 44/1000 point", approx(p, 0.044))
    check("wilson 44/1000 CI", 0.032 < lo < 0.034 and 0.058 < hi < 0.060,
          "[{:.4f}, {:.4f}]".format(lo, hi))

    check("ci_overlap disjoint", not ST.ci_overlap((0.0, 0.1), (0.2, 0.3)))
    check("ci_overlap touching", ST.ci_overlap((0.0, 0.2), (0.2, 0.3)))
    check("ci_overlap nested", ST.ci_overlap((0.0, 0.5), (0.1, 0.2)))

    check("spearman monotone = 1",
          approx(ST.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]), 1.0))
    check("spearman antitone = -1",
          approx(ST.spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]), -1.0))
    # Ties must be handled: the clamp(max=0.5) on safety returns creates them.
    r = ST.spearman_rho([1, 2, 2, 3], [1, 2, 2, 3])
    check("spearman with ties = 1", approx(r, 1.0), "rho={}".format(r))

    check("auroc perfect", approx(ST.auroc([0, 1, 2, 3], [0, 0, 1, 1]), 1.0))
    check("auroc inverted", approx(ST.auroc([3, 2, 1, 0], [0, 0, 1, 1]), 0.0))
    check("auroc chance", approx(ST.auroc([1, 2, 1, 2], [0, 1, 1, 0]), 0.5))

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    check("mean", approx(ST.mean(xs), 3.0))
    check("std ddof=1", approx(ST.std(xs), math.sqrt(2.5)))
    check("sem", approx(ST.sem(xs), math.sqrt(2.5) / math.sqrt(5)))

    m, lo, hi = ST.bootstrap_ci([1.0] * 50)
    check("bootstrap degenerate", approx(m, 1.0) and approx(lo, 1.0))
    m, lo, hi = ST.bootstrap_ci([float(i) for i in range(100)], n_boot=500)
    check("bootstrap brackets mean", lo < m < hi, "{} {} {}".format(lo, m, hi))

    check("fmt_rate", ST.fmt_rate(0.044, 0.032, 0.059) == "4.4 [3.2, 5.9]",
          ST.fmt_rate(0.044, 0.032, 0.059))
    check("fmt_mean", ST.fmt_mean(33.9, 4.21) == "33.9 ± 4.2",
          ST.fmt_mean(33.9, 4.21))


# --------------------------------------------------------------- metrics ----

def rec(**kw):
    r = M.blank_record()
    r.update({"n_steps": 100, "term_proj_grav_z": -1.0, "term_base_z_rel": 0.35,
              "term_fall_flag": 0})
    r.update(kw)
    return r


def test_metrics():
    print("\n[metrics]")
    check("fall: upright timeout is not a fall", not M.is_fall(rec()))
    check("fall: tilted past 60 deg is a fall",
          M.is_fall(rec(term_proj_grav_z=-0.4)))
    check("fall: 60 deg exactly is not (strict >)",
          not M.is_fall(rec(term_proj_grav_z=-0.5)))
    check("fall: base on the floor is a fall",
          M.is_fall(rec(term_base_z_rel=0.10)))
    check("fall: env hard flag is a fall",
          M.is_fall(rec(term_fall_flag=1)))

    recs = [rec(n_steps=100, n_collision_steps=10),
            rec(n_steps=100, n_collision_steps=0),
            rec(n_steps=50, n_collision_steps=5)]
    k, n = M.collision_rate(recs)
    check("collision rate pools timesteps, not episodes",
          k == 15 and n == 250, "{}/{}".format(k, n))
    k, n = M.eps_with_collision(recs)
    check("episodes-with-collision", k == 2 and n == 3)

    recs = [rec(activation_steps=10, target_miss_steps=7, n_steps=100),
            rec(activation_steps=0, target_miss_steps=0, n_steps=100)]
    k, n = M.target_miss_rate(recs)
    check("target-miss denominator is filter-active steps, not all steps",
          k == 7 and n == 10, "{}/{}".format(k, n))

    lo, hi = M.hard_dof_limits(-0.5, 0.5, 0.9)
    check("hard limits recovered from soft",
          approx(lo, -0.5 / 0.9) and approx(hi, 0.5 / 0.9),
          "{} {}".format(lo, hi))
    lo, hi = M.hard_dof_limits(-1.0, 3.0, 1.0)
    check("hard limits identity at soft=1",
          approx(lo, -1.0) and approx(hi, 3.0))


def test_sanity_checks():
    print("\n[sanity checks §5]")
    good = {
        "unfiltered": [rec(epsilon_or_alpha="", activation_steps=0)],
        "eps0.05": [rec(epsilon_or_alpha=0.05, activation_steps=5,
                        max_dnorm_inf=0.049)],
        "eps0.20": [rec(epsilon_or_alpha=0.20, activation_steps=30,
                        max_dnorm_inf=0.199)],
    }
    out = dict((n, ok) for n, ok, _ in M.sanity_checks(good, "unfiltered"))
    check("clean sweep passes every check", all(out.values()), str(out))

    broken = dict(good)
    broken["eps0.05"] = [rec(epsilon_or_alpha=0.05, activation_steps=5,
                             max_dnorm_inf=0.31)]
    res = M.sanity_checks(broken, "unfiltered")
    tr = [ok for n, ok, _ in res if n.startswith("||da||_inf")]
    check("trust-region violation is caught", not all(tr), str(res))

    nonmono = dict(good)
    nonmono["eps0.20"] = [rec(epsilon_or_alpha=0.20, activation_steps=1,
                              max_dnorm_inf=0.19)]
    res = dict((n, ok) for n, ok, _ in M.sanity_checks(nonmono, "unfiltered"))
    check("non-monotone activation is caught",
          not res["activation non-decreasing in eps"])

    leaky = dict(good)
    leaky["unfiltered"] = [rec(epsilon_or_alpha="", activation_steps=3)]
    res = dict((n, ok) for n, ok, _ in M.sanity_checks(leaky, "unfiltered"))
    check("unfiltered row that moves actions is caught",
          not res["unfiltered row has zero activation"])


def test_seeds():
    print("\n[seeds]")
    a, b = S.generate(50), S.generate(50)
    check("seed generation is deterministic", a == b)
    check("seed file has 1000 unique entries",
          len(set(S.load()[0])) == 1000)
    plan = S.chunk_plan(S.load()[0], 250, 1000)
    check("chunk plan covers exactly n_episodes",
          sum(k for _, _, k in plan) == 1000 and len(plan) == 4, str(plan))
    plan = S.chunk_plan(S.load()[0], 300, 1000)
    check("chunk plan handles a ragged last chunk",
          sum(k for _, _, k in plan) == 1000 and plan[-1][2] == 100, str(plan))


# ------------------------------------------------- synthetic sweep -> CSV ----

def synth_run(path, run_id, variant, eps, n, coll_p, fall_p, ret_mu,
              act_p=0.0, miss_frac=0.0, lat=0.05, rng=None, manifest=None):
    rng = rng or random.Random(0)
    rows = []
    for i in range(n):
        steps = 400
        ncoll = sum(1 for _ in range(steps) if rng.random() < coll_p)
        fell = rng.random() < fall_p
        nact = int(steps * act_p)
        rows.append(dict(
            M.blank_record(),
            run_id=run_id, method="E1", variant=variant,
            epsilon_or_alpha=("" if eps is None else eps),
            seed=1, episode_idx=i, terrain="corridor",
            collided=int(ncoll > 0), n_collision_steps=ncoll,
            n_proximity_steps=min(steps, int(ncoll * 2.5)),
            n_viol_steps=0, n_steps=steps, fell=int(fell),
            **{"return": rng.gauss(ret_mu, 4.0)},
            vel_err=0.08, mean_lat_vel=lat, mean_h=0.4, min_h=-0.1,
            activation_steps=nact,
            target_miss_steps=int(nact * miss_frac),
            trigger_steps=nact, handoffs=0, peak_yaw=3.0, peak_jerk=1.0,
            term_timeout=int(not fell), term_fall_flag=int(fell),
            term_vel_violate=0, term_contact=0,
            term_proj_grav_z=(-0.2 if fell else -1.0),
            term_base_z_rel=(0.08 if fell else 0.35),
            term_base_vel_z=0.0,
            max_dnorm_inf=(0.0 if eps is None else eps * 0.99),
            sum_dnorm_inf=0.0, sum_dnorm_l2=0.0, mean_q_gap=0.0, n_q_gap=0,
            steps_flight=100, steps_partial=150, steps_stance=150,
            dnorm_flight=0.0, dnorm_partial=0.0, dnorm_stance=0.0,
            chunk_idx=0, env_idx=i, terrain_level=i % 10))
    with open(os.path.join(path, run_id + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=M.EPISODE_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    man = {"epsilon_rad": eps, "tau": -0.25, "always_on": False,
           "trigger_source": "vhat", "dr_mode": "off"}
    man.update(manifest or {})
    with open(os.path.join(path, run_id + ".manifest.json"), "w") as f:
        json.dump(man, f)


def build_world(path, kind, qsafe_ok=True):
    """Three synthetic worlds with known ground-truth verdicts."""
    os.makedirs(path, exist_ok=True)
    rng = random.Random(7)
    battery = {"spearman_rho_q": 0.72 if qsafe_ok else 0.31,
               "auroc_q_collision20": 0.86 if qsafe_ok else 0.62,
               "auroc_min_cbf_h_baseline": 0.70}
    man_q = {"qsafe_battery": battery,
             "qsafe_verdict": {"critic_informative": qsafe_ok,
                               "beats_distance_heuristic": True}}
    synth_run(path, "pi_nom_unfiltered", "none", None, 400, 0.139, 0.05, 34.0,
              rng=rng, manifest=man_q)
    if kind == "holds":
        spec = [(0.02, 0.137, 0.05, 33.5, 0.10, 0.95),
                (0.05, 0.133, 0.06, 33.0, 0.18, 0.88),
                (0.10, 0.128, 0.09, 31.0, 0.30, 0.72),
                (0.20, 0.121, 0.20, 26.0, 0.44, 0.55),
                (0.50, 0.115, 0.46, 15.0, 0.61, 0.30)]
    elif kind == "falsified":
        spec = [(0.02, 0.130, 0.05, 33.5, 0.10, 0.90),
                (0.05, 0.090, 0.05, 33.2, 0.19, 0.70),
                (0.10, 0.040, 0.05, 32.5, 0.31, 0.40),
                (0.20, 0.035, 0.11, 28.0, 0.45, 0.25),
                (0.50, 0.030, 0.40, 16.0, 0.60, 0.12)]
    else:
        raise ValueError(kind)
    for eps, cp, fp, rm, ap, mf in spec:
        synth_run(path, "pi_nom_A_eps{:.3f}".format(eps), "A_sampling", eps,
                  400, cp, fp, rm, act_p=ap, miss_frac=mf, rng=rng,
                  manifest=man_q)


def run_analyzer(path, out):
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments/e1/analyze_e1.py"),
         "--dir", path, "--out", out, "--no_figure"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        return None
    with open(out + ".json") as f:
        return json.load(f)


def test_verdict_pipeline():
    print("\n[verdict pipeline §3.1 + §7]")
    tmp = tempfile.mkdtemp()
    try:
        # --- world 1: the registered prediction holds -------------------
        d = os.path.join(tmp, "holds")
        build_world(d, "holds")
        blob = run_analyzer(d, os.path.join(tmp, "R1a_holds"))
        check("analyzer runs end to end", blob is not None)
        if blob:
            v = blob["verdict"]
            check("holds: large-eps destabilisation detected",
                  v["checks"]["large_eps_destabilises"] is True)
            check("holds: small-eps impotence detected",
                  v["checks"]["small_eps_impotent"] is True)
            check("holds: no viable epsilon",
                  v["checks"]["no_viable_epsilon"] is True)
            check("holds: primary claim holds", v["holds"] is True)
            check("holds: §7 key is e1_holds",
                  blob["rebuttal_key"] == "e1_holds", blob["rebuttal_key"])
            check("holds: target-miss rises as eps falls",
                  blob["diagnostics"]["target_miss_rises_as_eps_falls"])
            check("holds: no emergent sidestepping flagged",
                  blob["diagnostics"]["no_emergent_sidestepping"])
            check("holds: sanity checks all pass",
                  all(c["passed"] for c in blob["sanity_checks"]),
                  str(blob["sanity_checks"]))
            check("holds: markdown table written",
                  os.path.exists(os.path.join(tmp, "R1a_holds.md")))
            check("holds: latex table written",
                  os.path.exists(os.path.join(tmp, "R1a_holds.tex")))

        # --- world 2: some epsilon works -> falsified -------------------
        d = os.path.join(tmp, "falsified")
        build_world(d, "falsified")
        blob = run_analyzer(d, os.path.join(tmp, "R1a_fals"))
        if blob:
            v = blob["verdict"]
            check("falsified: a viable epsilon is found",
                  v["checks"]["no_viable_epsilon"] is False)
            check("falsified: flagged as falsified", v["falsified"] is True)
            check("falsified: §7 key is e1_falsified",
                  blob["rebuttal_key"] == "e1_falsified",
                  blob["rebuttal_key"])
            check("falsified: viable point reported",
                  len(v["detail"]["viable_points"]) > 0)

        # --- world 3: Q-hat failed its battery; guard outranks the sweep --
        d = os.path.join(tmp, "badq")
        build_world(d, "holds", qsafe_ok=False)
        blob = run_analyzer(d, os.path.join(tmp, "R1a_badq"))
        if blob:
            check("bad Q-hat: §7 key is qhat_failed_validation",
                  blob["rebuttal_key"] == "qhat_failed_validation",
                  blob["rebuttal_key"])
            check("bad Q-hat: sentence names rho and AUROC",
                  "rho = 0.31" in blob["rebuttal_sentence"]
                  and "AUROC = 0.62" in blob["rebuttal_sentence"],
                  blob["rebuttal_sentence"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _wm_parser():
    """Same shape as WMPRunner._build_world_model's parser: six fixed flags
    plus one argument per top-level key of dreamer/configs.yaml's `defaults`."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--task", default="go1_amp")
    p.add_argument("--sim_device", default="cuda:0")
    p.add_argument("--wm_device", default="None")
    p.add_argument("--terrain", default="climb")
    p.add_argument("--checkpoint", default=-1)
    for key in ("horizon", "batch_size", "device", "units", "train_ratio"):
        p.add_argument("--" + key, default=None)
    return p


def test_argv_passthrough():
    """Three parsers read sys.argv on the way to a runner: this script's own,
    legged_gym's get_args, and WMPRunner._build_world_model's.  The last one
    knows only its six flags plus the dreamer config keys, so every legged_gym
    flag has to survive it.  Regression test for `--rl_device`/`--run_name`/
    `--max_iterations` aborting `_build_world_model`."""
    print("\n[argv plumbing]")
    import argparse
    import contextlib
    import io
    from safeloco_eval.cli import parse_and_strip

    cmdlines = {
        "train_policy": [
            "train_policy.py", "--policy", "pi_nom", "--task", "go1_amp",
            "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
            "--run_name", "pi_nom", "--max_iterations", "2000"],
        "run_e1": [
            "run_e1.py", "--filter", "A", "--epsilon", "0.1", "--task",
            "go1_amp", "--headless", "--sim_device", "cuda:0", "--rl_device",
            "cuda:0", "--load_run", "pi_nom", "--checkpoint", "-1",
            "--num_envs", "250", "--seed", "1"],
    }
    own = {"train_policy": [("--policy", str), ("--rs_weight", float)],
           "run_e1": [("--filter", str), ("--epsilon", float)]}

    saved = sys.argv
    try:
        for name, argv in cmdlines.items():
            sys.argv = list(argv)
            p = argparse.ArgumentParser(add_help=False)
            for flag, typ in own[name]:
                p.add_argument(flag, type=typ)
            parse_and_strip(p)
            check("{}: own flags stripped from argv".format(name),
                  all(f not in sys.argv for f, _ in own[name]),
                  " ".join(sys.argv))

            cfg, unknown = _wm_parser().parse_known_args()
            check("{}: world-model parser survives the CLI".format(name),
                  cfg.task == "go1_amp" and cfg.sim_device == "cuda:0"
                  and cfg.headless is True,
                  "task={} dev={} headless={}".format(
                      cfg.task, cfg.sim_device, cfg.headless))
            check("{}: legged_gym flags reported unknown, not fatal".format(name),
                  "--rl_device" in unknown)

            # The same parser under parse_args (the pre-fix behaviour) aborts.
            aborted = False
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    _wm_parser().parse_args()
            except SystemExit:
                aborted = True
            check("{}: parse_args would have aborted (bug reproduced)".format(name),
                  aborted)
    finally:
        sys.argv = saved

    src = open(os.path.join(ROOT, "rsl_rl/runners/wmp_runner.py")).read()
    check("wmp_runner uses parse_known_args",
          "parser.parse_known_args()" in src
          and "self.wm_config = parser.parse_args()" not in src)


def test_collision_geometry_if_torch():
    """The collision predicate, transcribed from play_plan.py.  A link
    collides when it is within r_obs + tol of an obstacle axis in XY *and* at
    or below the cylinder top."""
    print("\n[collision geometry]")
    try:
        import torch
    except ImportError:
        print("  SKIP  torch not installed in this environment")
        return
    from safeloco_eval.eval_common import link_obstacle_collision

    r, tol, hgt = 0.15, M.BODY_COLLISION_TOL, M.CYLINDER_HEIGHT
    obs = torch.tensor([[[0.0, 0.0, 0.0]]])          # one cylinder at origin
    rad = torch.tensor([r])
    mask = torch.tensor([[True]])
    has = torch.tensor([True])

    def hit(x, y, z, mask=mask, has=has):
        body = torch.tensor([[[x, y, z]]])
        return bool(link_obstacle_collision(body, obs, rad, mask, has)[0])

    check("link on the axis, low, collides", hit(0.0, 0.0, 0.05))
    check("link far away does not", not hit(1.0, 0.0, 0.05))
    check("just inside r+tol collides", hit(r + tol - 1e-3, 0.0, 0.05))
    check("just outside r+tol does not", not hit(r + tol + 1e-3, 0.0, 0.05))
    check("stepping over the top does not collide", not hit(0.0, 0.0, hgt + 0.1),
          "z above cylinder top + tol")
    check("at the top within tol still collides", hit(0.0, 0.0, hgt))
    check("masked-out obstacle is ignored",
          not hit(0.0, 0.0, 0.05, mask=torch.tensor([[False]])))
    check("env with no obstacles never collides",
          not hit(0.0, 0.0, 0.05, has=torch.tensor([False])))

    # any-link semantics: one colliding foot is enough, and it must not be
    # masked out by the other links being clear.
    body = torch.tensor([[[2.0, 2.0, 0.3], [0.0, 0.0, 0.05], [3.0, 3.0, 0.3]]])
    check("any link colliding flags the timestep",
          bool(link_obstacle_collision(body, obs, rad, mask, has)[0]))

    # Vectorised over envs and obstacles, with padding.
    obs2 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
                         [[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]])
    rad2 = torch.tensor([r, r])
    mask2 = torch.tensor([[True, False], [True, False]])   # 2nd is padding
    has2 = torch.tensor([True, True])
    body2 = torch.tensor([[[5.0, 5.0, 0.05]],      # only the padded obstacle
                          [[0.0, 0.0, 0.05]]])     # the real one
    out = link_obstacle_collision(body2, obs2, rad2, mask2, has2)
    check("padded obstacles do not produce phantom collisions",
          not bool(out[0]) and bool(out[1]), str(out.tolist()))


def test_qsafe_helpers_if_torch():
    print("\n[qsafe helpers]")
    try:
        import torch
    except ImportError:
        print("  SKIP  torch not installed in this environment")
        return
    from safeloco_eval.qsafe import (compute_safety_returns,
                                     collision_within_horizon,
                                     SafetyActionValue)

    # Recursion matches the closed form on a 3-step episode with alpha=0.7.
    sv = torch.tensor([[0.4], [-0.2], [0.1]])
    done = torch.tensor([[0.0], [0.0], [1.0]])
    R = compute_safety_returns(sv, done, alpha=0.7)
    r2 = 0.7 * min(0.1, 0.0) + 0.3 * 0.1
    r1 = 0.7 * min(-0.2, r2) + 0.3 * -0.2
    r0 = 0.7 * min(0.4, r1) + 0.3 * 0.4
    check("safety-return recursion", abs(float(R[0, 0]) - r0) < 1e-6,
          "{} vs {}".format(float(R[0, 0]), r0))

    # Labels must not leak across an episode boundary.
    h = torch.tensor([[0.5], [0.5], [-0.5], [0.5]])
    done = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    lab = collision_within_horizon(h, done, horizon=5, thresh=-0.05)
    check("collision label at the hit", bool(lab[2, 0]))
    check("collision label does not cross a done", not bool(lab[1, 0]))
    check("collision label before the boundary is clean", not bool(lab[0, 0]))

    # Factorised first layer == an un-factorised Linear over [s; a].
    m = SafetyActionValue(num_critic_obs=8, num_actions=4, wm_feature_dim=6,
                          grid_enabled=False)
    s = torch.randn(5, m.state_dim)
    a = torch.randn(5, 4)
    q_flat = m.q_from_features(s, a)
    q_batched = m.q_from_features(s, a.unsqueeze(1).repeat(1, 3, 1))
    check("Q broadcast over K equals the per-action Q",
          torch.allclose(q_flat, q_batched[:, 0], atol=1e-6))
    check("Q respects the clamp at 0.5", bool((q_flat <= 0.5 + 1e-6).all()))


def main():
    print("=" * 70)
    print("E1 offline test suite (no IsaacGym / GPU required)")
    print("=" * 70)
    test_stats()
    test_metrics()
    test_sanity_checks()
    test_seeds()
    test_argv_passthrough()
    test_verdict_pipeline()
    test_collision_geometry_if_torch()
    test_qsafe_helpers_if_torch()
    print("\n" + "=" * 70)
    if FAILURES:
        print("{} FAILURE(S): {}".format(len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
