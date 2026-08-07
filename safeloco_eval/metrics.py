"""Metric definitions, fixed once (analysis_protocol.md §1).

`Ambiguity here is what makes tables un-mergeable across the two branches.`
Every threshold and every derived quantity lives here and nowhere else.  Copy
this file verbatim to the joint-limit branch (§1.1 invariant).

Nothing in this module imports torch, so `analyze_*.py` and the offline tests
can use it directly.
"""

import math

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

#: **Collision** is a geometric link-vs-cylinder intersection test, matching
#: `legged_gym/scripts/play_plan.py` -- the definition the paper's own numbers
#: were produced under, so E1's rows are directly comparable to them.
#:
#: A timestep collides when **any** robot rigid body sits within
#: `obstacle_radius + BODY_COLLISION_TOL` of an obstacle axis in XY, **and**
#: that body is at or below the cylinder's top (`obstacle_z + CYLINDER_HEIGHT`)
#: within the same tolerance.  All links, not just the base: this is real
#: leg-environment contact, not a proximity proxy.
CYLINDER_HEIGHT = 0.3
BODY_COLLISION_TOL = 0.03

#: Secondary *proximity* diagnostic, reported alongside but never as the
#: headline: `min_cbf_h < COLLISION_H_THRESH`, the raw spatial margin against
#: Table 4's violation threshold.  Uses `min_cbf_h` and **not** `safety_value`
#: -- the latter is the rate-augmented `sv = 0.3*h_dot + h` and goes negative on
#: approach with no proximity at all.  Note `h = dist - 2*r_obs - 0.35` measures
#: base-to-obstacle-centre in 3-D, so it crosses this threshold at roughly
#: 0.37 m of real clearance; it is a near-miss counter, not contact.
COLLISION_H_THRESH = -0.05

#: A timestep counts as a *joint violation* when any joint sits within this
#: many radians of a **hard** limit (Table 4 eval margin).  Distinct from the
#: 15% `d_margin` guard band used during training.
JOINT_VIOL_MARGIN = 0.02

#: Fall definition (one definition across both branches, and retroactively
#: applied to the paper's existing methods).  A terminating episode is a fall
#: when either the base pitched/rolled past `FALL_TILT_COS` or the base sank
#: below `FALL_HEIGHT` above its terrain cell.  `projected_gravity[2]` is
#: -1 when upright and 0 at 90 degrees of tilt, so -0.5 is a 60 degree tilt.
FALL_TILT_PROJ_GRAV_Z = -0.5
FALL_HEIGHT = 0.15

#: An action-filter step counts as *active* when it actually moved the action.
ACTIVATION_DNORM_THRESH = 1e-6

#: Control period, seconds.  Only used to turn per-step speeds into distances.
CONTROL_DT = 0.02

#: Handoff-attributed falls (E3/J3): a fall this many seconds after a switch.
HANDOFF_WINDOW_S = 0.5

#: Reference values the E1 verdict rule compares against (protocol §3.1).
OURS_COLLISION_RATE = 0.044
OURS_RETURN = 33.9
OURS_MEAN_LAT_VEL = 0.085
REWARD_SHAPING_COLLISION_RATE = 0.139
# E2 (analysis_protocol.md §3.2).  "Reduces to shaping" means the collision
# rate lands in the reward-shaping band rather than near ours -- the protocol
# writes that band as "~12-14%", around the paper's 13.9% row.
SHAPING_BAND_LO = 0.12
SHAPING_BAND_HI = 0.14

VERDICT_COLLISION_CEILING = 0.06   # "collision <= 6%"
VERDICT_RETURN_FLOOR = 30.0        # "return >= 30"
VERDICT_SMALL_EPS_FLOOR = 0.10     # "collision rate remains >= ~10%"


# --------------------------------------------------------------------------
# Per-episode record schema (analysis_protocol.md §6)
# --------------------------------------------------------------------------
#
# Everything in §1 and §3 is a groupby over this table, which is also why the
# two branches merge by concatenating CSVs.  Columns after the protocol's
# required set are additive only -- never reorder or rename the first block.

EPISODE_FIELDS = [
    # --- protocol §6 required schema ------------------------------------
    "run_id", "method", "variant", "epsilon_or_alpha", "seed", "episode_idx",
    "terrain", "collided", "n_collision_steps", "n_viol_steps", "n_steps",
    "fell", "return", "vel_err", "mean_lat_vel", "mean_h", "min_h",
    "mean_fwd_vel",
    "activation_steps", "target_miss_steps", "handoffs", "peak_yaw",
    "peak_jerk",
    # --- termination diagnostics (so `fell` is recomputable post hoc) ----
    "term_timeout", "term_fall_flag", "term_vel_violate", "term_contact",
    "term_proj_grav_z", "term_base_z_rel", "term_base_vel_z",
    # --- secondary proximity diagnostic ---------------------------------
    "n_proximity_steps",
    # --- command-filter diagnostics (E4 / E4-Y) --------------------------
    "infeasible_steps", "sum_dnorm_v", "sum_dnorm_omega",
    # Commanded vs realised yaw rate, on steps where the filter asked for
    # yaw. A CBF certifies a twist assuming the twist is executed; if the
    # policy cannot track the yaw it was given, the executed motion does not
    # satisfy the constraint the filter certified, and the filter has braked
    # less on the strength of steering that never happened.
    "sum_yaw_cmd_abs", "sum_yaw_realised_abs", "sum_yaw_signed",
    "yaw_cmd_steps",
    # --- E1 filter diagnostics ------------------------------------------
    "trigger_steps", "sum_dnorm_inf", "max_dnorm_inf", "sum_dnorm_l2",
    "mean_q_gap", "n_q_gap",
    # --- contact-phase buckets, feeds E6 --------------------------------
    "steps_flight", "steps_partial", "steps_stance",
    "dnorm_flight", "dnorm_partial", "dnorm_stance",
    # --- provenance ------------------------------------------------------
    "chunk_idx", "env_idx", "terrain_level",
]


def blank_record():
    return {k: 0 for k in EPISODE_FIELDS}


# --------------------------------------------------------------------------
# Derived metrics -- the only implementations, used by every analyze script
# --------------------------------------------------------------------------

def is_fall(rec):
    """The single fall definition (§1).

    A timeout that ends an otherwise-upright episode is not a fall; a timeout
    that coincides with the robot lying on its side is.  We therefore read the
    *terminal state*, not the termination cause, plus the simulator's own
    hard fall flag (base z-velocity < -3 m/s or fully inverted).
    """
    if float(rec["term_proj_grav_z"]) > FALL_TILT_PROJ_GRAV_Z:
        return True
    if float(rec["term_base_z_rel"]) < FALL_HEIGHT:
        return True
    return bool(int(rec["term_fall_flag"]))


def collision_rate(recs):
    """Collision env-timesteps / total env-timesteps, pooled over episodes --
    the convention `play_plan.py::_write_collision_summary` uses and the one
    behind Table 2's 13.9% / 4.4%.  Returns (k, n)."""
    k = sum(int(r["n_collision_steps"]) for r in recs)
    n = sum(int(r["n_steps"]) for r in recs)
    return k, n


def eps_with_collision(recs):
    return sum(1 for r in recs if int(r["n_collision_steps"]) > 0), len(recs)


def internalization_gap(recs_without, recs_with, rate_fn=None):
    """metric(without filter) - metric(with filter), §1's E2/J2 row.

    Sign convention is the protocol's: **positive means worse without the
    filter**, i.e. the policy was leaning on it and never internalized the
    constraint.  Defaults to collision rate; pass `rate_fn` for any other
    (k, n) metric.
    """
    fn = rate_fn or collision_rate
    kw, nw = fn(recs_without)
    kv, nv = fn(recs_with)
    return (kw / max(nw, 1)) - (kv / max(nv, 1))


def in_shaping_band(rate):
    """True if a collision rate sits inside the reward-shaping band."""
    return SHAPING_BAND_LO <= rate <= SHAPING_BAND_HI


def proximity_rate(recs):
    """Secondary: timesteps with `min_cbf_h < COLLISION_H_THRESH`, the margin
    proxy.  Reported next to the geometric rate so the two are never confused
    and so threshold sensitivity is visible."""
    k = sum(int(r.get("n_proximity_steps", 0) or 0) for r in recs)
    n = sum(int(r["n_steps"]) for r in recs)
    return k, n


def fall_rate(recs):
    return sum(1 for r in recs if is_fall(r)), len(recs)


def viol_per_step(recs):
    k = sum(int(r["n_viol_steps"]) for r in recs)
    n = sum(int(r["n_steps"]) for r in recs)
    return k, n


def viol_per_ep(recs):
    return [int(r["n_viol_steps"]) for r in recs]


def eps_with_viol(recs):
    return sum(1 for r in recs if int(r["n_viol_steps"]) > 0), len(recs)


def infeasibility_rate(recs):
    """§1: filter-active timesteps where the constraint set was empty.

    Denominator is **trigger** steps -- those where the nominal command already
    violated a constraint, so the filter had work to do.  Activation is the
    wrong denominator: a step can be infeasible precisely *because* the filter
    could not move the command, which would leave it triggered but not
    activated and let the numerator exceed the denominator.
    """
    k = sum(int(r.get("infeasible_steps", 0) or 0) for r in recs)
    n = sum(int(r.get("trigger_steps", 0) or 0) for r in recs)
    return k, n


def activation_rate(recs):
    """Timesteps where the filter modified the action / total timesteps."""
    k = sum(int(r["activation_steps"]) for r in recs)
    n = sum(int(r["n_steps"]) for r in recs)
    return k, n


def target_miss_rate(recs):
    """Filter-active steps where no sample cleared the Q-hat margin, over
    filter-active steps.  E1 only -- the small-epsilon diagnostic."""
    k = sum(int(r["target_miss_steps"]) for r in recs)
    n = sum(int(r["activation_steps"]) for r in recs)
    return k, n


def per_episode(recs, field):
    return [float(r[field]) for r in recs]


def weighted_step_mean(recs, sum_field):
    """Mean over timesteps of a per-step quantity accumulated as a sum."""
    num = sum(float(r[sum_field]) for r in recs)
    den = sum(int(r["n_steps"]) for r in recs)
    return num / den if den else float("nan")


def min_over_runs(recs, field):
    """Per-run minimum -- report without +/- (§1)."""
    vals = [float(r[field]) for r in recs]
    return min(vals) if vals else float("nan")


def chatter_rate(recs, dt):
    """Switches per second of episode time (E3/J3)."""
    h = sum(int(r["handoffs"]) for r in recs)
    t = sum(int(r["n_steps"]) for r in recs) * dt
    return h / t if t else float("nan")


# --------------------------------------------------------------------------
# Sanity checks (analysis_protocol.md §5) -- run before believing anything
# --------------------------------------------------------------------------

def sanity_checks(recs_by_key, unfiltered_key, mono_keys=None, dnorm_tol=1e-4):
    """Returns a list of (name, passed, detail) triples.

    `recs_by_key` maps a label to its episode records; `unfiltered_key` is the
    label of the filter-disabled reference row.  `mono_keys` restricts the
    activation-monotonicity check to one sweep curve (always-on and gradient
    variants sit on different curves and would break it spuriously).
    """
    out = []

    # 1. filter disabled reproduces unfiltered pi_nom to within seed noise
    if unfiltered_key in recs_by_key:
        base = recs_by_key[unfiltered_key]
        k, n = activation_rate(base)
        out.append(("unfiltered row has zero activation", k == 0,
                    "{}/{} steps active".format(k, n)))

    # 2. ||da||_inf <= epsilon for every filtered step
    for key, recs in recs_by_key.items():
        eps = _row_epsilon(recs)
        if eps is None or eps <= 0:
            continue
        worst = max((float(r["max_dnorm_inf"]) for r in recs), default=0.0)
        out.append(("||da||_inf <= eps for {}".format(key),
                    worst <= eps + dnorm_tol,
                    "max {:.6f} vs eps {:.6f}".format(worst, eps)))

    # 3. activation rises monotonically with epsilon
    pts = []
    for key in (mono_keys if mono_keys is not None else list(recs_by_key)):
        recs = recs_by_key.get(key) or []
        eps = _row_epsilon(recs)
        if eps is None:
            continue
        k, n = activation_rate(recs)
        pts.append((eps, k / n if n else 0.0))
    pts.sort()
    mono = all(b[1] >= a[1] - 1e-6 for a, b in zip(pts, pts[1:]))
    out.append(("activation non-decreasing in eps", mono,
                "; ".join("{:.3g}->{:.3f}".format(e, a) for e, a in pts)))

    return out


def _row_epsilon(recs):
    if not recs:
        return None
    try:
        return float(recs[0]["epsilon_or_alpha"])
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Joint limits -- shared with the Phase 2 branch
# --------------------------------------------------------------------------

def hard_dof_limits(soft_lo, soft_hi, soft_ratio):
    """Recover the URDF hard limits from the softened ones the env stores.

    `_process_dof_props` rewrites `dof_pos_limits` as m +/- 0.5*r*soft_ratio
    around the midpoint m of the hard range r.  Inverting is exact and saves
    touching the env.
    """
    m = 0.5 * (soft_lo + soft_hi)
    r = (soft_hi - soft_lo) / max(soft_ratio, 1e-9)
    return m - 0.5 * r, m + 0.5 * r


def isfinite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def distance_travelled(recs):
    """Mean forward distance per episode, metres.

    Any intervention that changes the command can buy a lower contact rate by
    going slower, so distance is reported next to every safety metric (E4
    post-mortem).  Derived from realised speed and episode length, not from the
    command, so a filter cannot flatter it.
    """
    if not recs:
        return float("nan")
    tot = 0.0
    for r in recs:
        v = float(r.get("mean_fwd_vel", 0) or 0)
        n = float(r.get("n_steps", 0) or 0)
        tot += v * n * CONTROL_DT
    return tot / len(recs)


def contact_per_metre(recs):
    """Contact timesteps per metre travelled.

    A per-timestep contact rate falls when the robot slows down even if its
    behaviour near obstacles is unchanged; normalising by distance separates
    the two.  This is what showed that a permissive command filter was
    genuinely worse than no filter rather than merely slower.
    """
    d = distance_travelled(recs)
    if not recs or not (d > 0):
        return float("nan")
    steps = sum(float(r.get("n_collision_steps", 0) or 0) for r in recs) / len(recs)
    return steps / d


def steer_brake_split(recs):
    """(steer share, brake share) of total correction magnitude.

    The command-space analogue of E1's gait-phase decomposition: it says
    whether a filter given steering authority actually uses it, or quietly
    falls back to braking.  Returns (nan, nan) when nothing was corrected.
    """
    sv = sum(float(r.get("sum_dnorm_v", 0) or 0) for r in recs)
    sw = sum(float(r.get("sum_dnorm_omega", 0) or 0) for r in recs)
    tot = sv + sw
    if tot <= 0:
        return (float("nan"), float("nan"))
    return (sw / tot, sv / tot)


def mean_fwd_vel(recs):
    """Realised forward speed, immune to the command being rewritten."""
    if not recs:
        return float("nan")
    return sum(float(r.get("mean_fwd_vel", 0) or 0) for r in recs) / len(recs)


def yaw_execution_gain(recs):
    """Yaw realised *in the commanded direction*, over commanded yaw.

    Uses the signed projection sign(w_cmd) * w_realised, not |w_realised|.
    That distinction decides the metric: a walking quadruped's yaw rate
    swings to 1.4-2.0 rad/s from gait alone -- an order of magnitude above
    the 0.1-0.8 rad/s a filter commands -- so |w_realised| is dominated by
    wobble and reads as near-perfect tracking even when the robot is not
    turning at all. The signed projection averages to ~0 under symmetric
    wobble and to |w_cmd| under real tracking.

    Falls back to the absolute ratio for records written before the signed
    column existed, and says so via `yaw_execution_gain_is_signed`.

    A CBF certifies a twist on the assumption the twist is executed.  If the
    policy cannot track the yaw the filter hands it, the executed motion does
    not satisfy the constraint that was certified -- and worse, the filter
    will have braked *less* than a braking-only filter would, on the strength
    of steering that never happened.  So a steering filter on a platform that
    cannot steer is not merely no better than braking; it can be worse.

    Returns nan when the filter never commanded yaw, which is the correct
    answer for the yaw-disabled arm rather than a misleading zero.
    """
    cmd = sum(float(r.get("sum_yaw_cmd_abs", 0) or 0) for r in recs)
    if cmd <= 1e-9:
        return float("nan")
    if yaw_execution_gain_is_signed(recs):
        return sum(float(r.get("sum_yaw_signed", 0) or 0) for r in recs) / cmd
    return sum(float(r.get("sum_yaw_realised_abs", 0) or 0)
               for r in recs) / cmd


def yaw_execution_gain_is_signed(recs):
    """True when the records carry the signed column, i.e. the gain is real.

    Without it the value is an absolute ratio inflated by gait wobble and
    must not be read as a tracking gain.
    """
    return any("sum_yaw_signed" in r for r in recs)
