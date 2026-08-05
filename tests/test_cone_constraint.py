"""Regression tests for the damped null-space projection.

`damped_null_space_project` used to overwrite its `d_safe` / `d_danger`
arguments with hardcoded values (cone_constraint.py:63-64), so every caller's
configuration was silently discarded and any ablation that varied the two
thresholds differed only by seed noise.  These tests pin the arguments as
load-bearing.

Needs torch (cone_constraint does tensor algebra); skips cleanly without it,
so it never breaks the stdlib-only guarantee of test_e1_offline.py.

Run:  python tests/test_cone_constraint.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
except ImportError:                                            # pragma: no cover
    print("SKIP tests/test_cone_constraint.py -- torch not available")
    sys.exit(0)

from rsl_rl.algorithms.cone_constraint import (                # noqa: E402
    damped_null_space_project, DEFAULT_D_SAFE, DEFAULT_D_DANGER)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


def _fixture():
    """Non-parallel task/safety gradients so the null component is non-zero."""
    g_task = torch.tensor([1.0, 0.0, 0.0])
    g_safety = torch.tensor([1.0, 2.0, 0.0])
    return g_task, g_safety


def _project(sv, **kw):
    g_task, g_safety = _fixture()
    return damped_null_space_project(g_task, g_safety, torch.tensor(sv), **kw)


def test_thresholds_are_honoured():
    """The whole point: passing different thresholds must change the output."""
    g_task, g_safety = _fixture()
    # safety_value = 0.0 sits above d_safe=-0.4 (full null projection, t=1) but
    # below d_safe=0.3 with d_danger=-0.3 ... which also saturates at t=1.
    # Use a value that lands mid-ramp for one setting and saturated for the other.
    sv = -0.45
    hist = _project(sv, d_safe=DEFAULT_D_SAFE, d_danger=DEFAULT_D_DANGER)
    alt = _project(sv, d_safe=0.3, d_danger=-0.3)
    check("passed thresholds change the result",
          not torch.allclose(hist, alt),
          "-> both gave %s; arguments are being ignored" % hist.tolist())


def test_saturation_endpoints():
    """Above d_safe -> pure null component; below d_danger -> untouched."""
    g_task, g_safety = _fixture()
    proj_coeff = torch.dot(g_safety, g_task) / torch.dot(g_task, g_task)
    g_null = g_safety - proj_coeff * g_task

    safe = _project(0.0, d_safe=-0.4, d_danger=-0.6)
    check("safety_value > d_safe gives the null-space component",
          torch.allclose(safe, g_null, atol=1e-6),
          "got %s want %s" % (safe.tolist(), g_null.tolist()))

    danger = _project(-1.0, d_safe=-0.4, d_danger=-0.6)
    check("safety_value < d_danger leaves g_safety unprojected",
          torch.allclose(danger, g_safety, atol=1e-6),
          "got %s want %s" % (danger.tolist(), g_safety.tolist()))


def test_midpoint_interpolates():
    """Halfway up the ramp is the mean of the two endpoints."""
    g_task, g_safety = _fixture()
    proj_coeff = torch.dot(g_safety, g_task) / torch.dot(g_task, g_task)
    g_null = g_safety - proj_coeff * g_task
    want = 0.5 * g_null + 0.5 * g_safety

    mid = _project(-0.5, d_safe=-0.4, d_danger=-0.6)
    check("midpoint of the ramp interpolates evenly",
          torch.allclose(mid, want, atol=1e-6),
          "got %s want %s" % (mid.tolist(), want.tolist()))


def test_defaults_match_historical_behaviour():
    """Calling with no thresholds must reproduce the old hardcoded values."""
    implicit = _project(-0.5)
    explicit = _project(-0.5, d_safe=-0.4, d_danger=-0.6)
    check("defaults reproduce the previously hardcoded -0.4 / -0.6",
          torch.allclose(implicit, explicit, atol=1e-8))


def test_inverted_thresholds_rejected():
    """d_safe <= d_danger inverts the ramp; refuse rather than mislead."""
    for bad in ((-0.6, -0.4), (-0.5, -0.5)):
        try:
            _project(-0.5, d_safe=bad[0], d_danger=bad[1])
        except ValueError:
            check("rejects d_safe=%s d_danger=%s" % bad, True)
        else:
            check("rejects d_safe=%s d_danger=%s" % bad, False,
                  "-> no ValueError raised")


def test_degenerate_task_gradient():
    """A vanishing task gradient has no null space to project into."""
    out = damped_null_space_project(
        torch.zeros(3), torch.tensor([1.0, 2.0, 0.0]), torch.tensor(0.0))
    check("zero task gradient returns g_safety unchanged",
          torch.allclose(out, torch.tensor([1.0, 2.0, 0.0]), atol=1e-8))


if __name__ == "__main__":
    print("cone_constraint: damped null-space projection")
    test_thresholds_are_honoured()
    test_saturation_endpoints()
    test_midpoint_interpolates()
    test_defaults_match_historical_behaviour()
    test_inverted_thresholds_rejected()
    test_degenerate_task_gradient()
    print()
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        sys.exit(1)
    print("all cone_constraint tests passed")
