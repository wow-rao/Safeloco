"""Reference rows every comparison is measured against.

The paper's Table 2 quotes 4.4% collision for the proposed method and 13.9%
for reward shaping.  Those were produced under a **different collision
detector** (base-level `min_cbf_h < -0.05`) and a **different harness** than
E1/E4 use (geometric link-vs-cylinder, DR off, fixed 0.6 m/s command).  Every
"is the baseline dominated" question turns on that boundary, so comparing
against a number measured a different way is the single largest hole in the
package.

The fix is to evaluate the reference policies under this harness once, write
the measured values here, and have every analyser read them.  Until that file
exists the Table 2 constants are used, and `source` says so loudly, so a
report can never quietly present a cross-detector comparison as a like-for-like
one.

    from safeloco_eval.reference import load_reference
    ref = load_reference()
    ref["ours"]["collision"]      # fraction
    ref["measured"]               # False until the same-harness rows exist
"""

import json
import os

from . import metrics as M

DEFAULT_PATH = os.path.join("logs", "reference", "reference.json")

#: Table 2, retained only as the fallback.  Different detector, different
#: harness -- see the module docstring.
TABLE2 = {
    "ours": {"collision": M.OURS_COLLISION_RATE, "ret": M.OURS_RETURN,
             "vel_err": M.OURS_MEAN_LAT_VEL, "lat_vel": M.OURS_MEAN_LAT_VEL,
             "fwd_vel": None, "distance": None},
    "reward_shaping": {"collision": M.REWARD_SHAPING_COLLISION_RATE,
                       "ret": None, "vel_err": None, "lat_vel": None,
                       "fwd_vel": None, "distance": None},
}

FALLBACK_WARNING = (
    "reference values come from Table 2, which used a base-level margin "
    "detector (min_cbf_h < -0.05) and a different harness. They are NOT "
    "like-for-like with the geometric link-vs-cylinder detector used here. "
    "Run experiments/reference/sweep_reference.sh to replace them."
)


def load_reference(path=None):
    """Measured same-harness reference rows, or the Table 2 fallback."""
    p = path or DEFAULT_PATH
    if os.path.exists(p):
        with open(p) as fh:
            blob = json.load(fh)
        blob["measured"] = True
        blob.setdefault("warning", None)
        return blob
    out = {k: dict(v) for k, v in TABLE2.items()}
    out["measured"] = False
    out["warning"] = FALLBACK_WARNING
    return out


def summarise_records(recs):
    """The reference-row fields, computed from per-episode records."""
    from . import stats as ST

    ck, cn = M.collision_rate(recs)
    fk, fn = M.fall_rate(recs)
    return {
        "collision": ck / max(cn, 1),
        "collision_ci": ST.cluster_bootstrap_rate_ci(
            recs, "n_collision_steps", "n_steps")[1:],
        "fall": fk / max(fn, 1),
        "ret": ST.mean(M.per_episode(recs, "return")),
        "vel_err": ST.mean(M.per_episode(recs, "vel_err")),
        "lat_vel": ST.mean(M.per_episode(recs, "mean_lat_vel")),
        "fwd_vel": M.mean_fwd_vel(recs),
        "distance": M.distance_travelled(recs),
        "contact_per_m": M.contact_per_metre(recs),
        "n_episodes": len(recs),
    }


def banner(ref):
    """One line an analyser should print before any comparison."""
    if ref.get("measured"):
        return ("reference rows: measured under this harness "
                "({} episodes)".format(
                    ref.get("ours", {}).get("n_episodes", "?")))
    return "WARNING: " + FALLBACK_WARNING
