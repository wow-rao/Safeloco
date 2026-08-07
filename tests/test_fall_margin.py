"""Unit tests for the E5 bipedal fall-safety margin (safeloco_eval/fall_margin.py).

The margin is the exact production math `Cassie._compute_safety_value` feeds
to the reachability pipeline, so its sign conventions and normalizations are
load-bearing: a sign slip here silently inverts the safety gradient.  Checked
against the shared fall definition's thresholds (metrics.py) and the
linear-inverted-pendulum capture-point identities.

Needs torch; skips cleanly without it.

Run:  python tests/test_fall_margin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
except ImportError:                                            # pragma: no cover
    print("SKIP tests/test_fall_margin.py -- torch not available")
    sys.exit(0)

from safeloco_eval import fall_margin as FM                     # noqa: E402
from safeloco_eval import metrics as M                          # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        FAILED.append(name)


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def t(*vals):
    return torch.tensor(vals, dtype=torch.float)


# ---------------------------------------------------------------- tilt ------

def test_tilt():
    m = FM.tilt_margin(t(-1.0, -0.5, 0.0, 1.0))
    check("tilt: upright -> +1", approx(m[0], 1.0), m)
    check("tilt: 60 deg (threshold) -> 0", approx(m[1], 0.0), m)
    check("tilt: 90 deg -> -1", approx(m[2], -1.0), m)
    check("tilt: inverted -> -3", approx(m[3], -3.0), m)
    check("tilt: threshold matches metrics.FALL_TILT_PROJ_GRAV_Z",
          approx(FM.tilt_margin(t(M.FALL_TILT_PROJ_GRAV_Z))[0], 0.0))


# -------------------------------------------------------------- height ------

def test_height():
    h_nom = 0.85
    m = FM.height_margin(t(h_nom, M.FALL_HEIGHT, 0.0, 5.0), h_nom=h_nom)
    check("height: nominal -> 1", approx(m[0], 1.0), m)
    check("height: h_fall -> 0", approx(m[1], 0.0), m)
    check("height: on the ground -> negative", m[2] < 0, m)
    check("height: flight clamped at 1.5", approx(m[3], 1.5), m)
    check("height: zero crossing matches metrics.FALL_HEIGHT",
          approx(FM.height_margin(t(M.FALL_HEIGHT), h_nom=h_nom)[0], 0.0))


# ------------------------------------------------------------- contact ------

def test_contact():
    m = FM.contact_margin(torch.tensor([True, False]))
    check("contact: illegal -> -1", approx(m[0], -1.0), m)
    check("contact: clean -> inactive sentinel", approx(m[1], FM.INACTIVE), m)

    # The floor must dominate the compose min even when pose margins are fine.
    sv, fail_min = FM.compose([t(0.9, 0.9), t(1.0, 1.0),
                               FM.contact_margin(torch.tensor([True, False]))])
    check("contact: floor dominates compose", approx(sv[0], -1.0), sv)
    check("contact: clean env keeps pose margin (clamped)",
          approx(sv[1], 0.5), sv)


# ----------------------------------------------------------------- dcm ------

def test_dcm():
    r_cap = 0.5
    base = t(0.0, 0.0).view(1, 2).repeat(4, 1)
    feet = base.clone()
    h = t(0.85, 0.85, 0.85, 0.85)

    # Standing still, capture point over the support midpoint -> +1.
    v0 = torch.zeros(4, 2)
    m = FM.dcm_margin(base, v0, h, feet, r_cap=r_cap)
    check("dcm: static over support -> +1", approx(m[0], 1.0), m)

    # Velocity placing xi exactly r_cap away -> 0 (capturability boundary).
    omega0 = (9.81 / 0.85) ** 0.5
    v_edge = torch.zeros(4, 2)
    v_edge[1, 0] = r_cap * omega0
    m = FM.dcm_margin(base, v_edge, h, feet, r_cap=r_cap)
    check("dcm: xi at r_cap -> 0", approx(m[1], 0.0, tol=1e-4), m)

    # Twice that velocity -> margin -1 (xi at 2 r_cap).
    v_out = torch.zeros(4, 2)
    v_out[2, 0] = 2 * r_cap * omega0
    m = FM.dcm_margin(base, v_out, h, feet, r_cap=r_cap)
    check("dcm: xi at 2 r_cap -> -1", approx(m[2], -1.0, tol=1e-4), m)

    # omega0 height clamp: below z_c_min, two different heights give the
    # same margin because both clamp to z_c_min.
    va = torch.zeros(2, 2)
    va[:, 0] = 1.0
    ma = FM.dcm_margin(t(0., 0.).view(1, 2).repeat(2, 1), va,
                       t(0.10, 0.25), feet[:2], r_cap=r_cap, z_c_min=0.3)
    check("dcm: omega0 clamped below z_c_min", approx(ma[0], ma[1]), ma)


# ------------------------------------------------------------- compose ------

def test_compose():
    # Clamp at +0.5 (the pipeline's ceiling).
    sv, fail_min = FM.compose([t(1.0), t(1.2)])
    check("compose: sv clamped at 0.5", approx(sv[0], 0.5), sv)
    check("compose: fail_min unclamped", approx(fail_min[0], 1.0), fail_min)

    # fail_min excludes the DCM term: a doomed-but-upright state has
    # sv < 0 (through dcm) while the pure fail-set margin stays positive --
    # exactly the gap the reachability critic is supposed to learn.
    sv, fail_min = FM.compose([t(0.8), t(0.9)], dcm=t(-0.6))
    check("compose: dcm can drive sv negative", approx(sv[0], -0.6), sv)
    check("compose: fail_min ignores dcm", approx(fail_min[0], 0.8), fail_min)

    # Batched shapes.
    E = 7
    sv, fail_min = FM.compose([torch.rand(E), torch.rand(E)],
                              dcm=torch.rand(E))
    check("compose: shapes [E]", sv.shape == (E,) and fail_min.shape == (E,))


# ---------------------------------------------------- fall-def consistency --

def test_consistency_with_is_fall():
    """A state with fail_min < 0 must be one metrics.is_fall would call
    fallen (tilt/height parts; the contact floor has no is_fall analogue)."""
    for gz in (-0.9, -0.51, -0.49, -0.1):
        margin = FM.tilt_margin(t(gz))[0]
        rec = M.blank_record()
        rec.update(term_proj_grav_z=gz, term_base_z_rel=0.8,
                   term_fall_flag=0)
        check("consistency: tilt gz={:+.2f}".format(gz),
              (margin < 0) == M.is_fall(rec),
              "margin={:+.3f} is_fall={}".format(margin, M.is_fall(rec)))
    for h in (0.5, 0.16, 0.14, 0.05):
        margin = FM.height_margin(t(h))[0]
        rec = M.blank_record()
        rec.update(term_proj_grav_z=-0.9, term_base_z_rel=h,
                   term_fall_flag=0)
        check("consistency: height h={:.2f}".format(h),
              (margin < 0) == M.is_fall(rec),
              "margin={:+.3f} is_fall={}".format(margin, M.is_fall(rec)))


def main():
    print("tests/test_fall_margin.py")
    test_tilt()
    test_height()
    test_contact()
    test_dcm()
    test_compose()
    test_consistency_with_is_fall()
    if FAILED:
        print("{} FAILED".format(len(FAILED)))
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
