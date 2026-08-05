"""Pins the input contract of safeloco_eval.filters.ActionFilter.apply().

The filter clamps its *candidates* to +/- clip_actions but never the action it
is handed, so `apply()` only guarantees ||da||_inf <= epsilon when the action
is already inside the box.  eval_common.EvalCollector satisfies that by
clamping first; E2's training hook did not, and raw Gaussian training samples
land outside the box routinely -- which surfaced as

    AssertionError: trust region violated: 15.000008 > eps 0.050000

after four iterations, where 15.0 = (66 - 6) * action_scale.  These tests hold
both halves of that in place: unclamped input violates the trust region,
clamped input cannot.

Needs torch; skips cleanly without it.

Run:  python tests/test_train_filter_contract.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
except ImportError:                                            # pragma: no cover
    print("SKIP tests/test_train_filter_contract.py -- torch not available")
    sys.exit(0)

from safeloco_eval.filters import SamplingProjectionFilter     # noqa: E402

FAILED = []
CLIP, EPS, SCALE = 6.0, 0.05, 0.25


def check(name, cond, detail=""):
    if cond:
        print("  [PASS] %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, detail))
        FAILED.append(name)


class StubQ(object):
    """Q that rewards actions nearer the origin, so the filter always moves."""

    delta = 1e-6

    def q_from_features(self, s_feat, actions):
        return -actions.pow(2).sum(dim=-1)


def make_filter():
    return SamplingProjectionFilter(
        qsafe=StubQ(), epsilon_rad=EPS, action_scale=SCALE, clip_actions=CLIP,
        tau=1e9, delta=1e-6, always_on=True, device="cpu", seed=0,
        k_samples=32)


def run(a):
    filt = make_filter()
    s_feat = torch.zeros(a.shape[0], 4)
    trigger = torch.full((a.shape[0],), -1.0)
    return filt.apply(a, s_feat, trigger)


def test_unclamped_input_violates():
    """The failure mode as reported: one action far outside the box."""
    a = torch.zeros(4, 12)
    a[2, 5] = 66.0
    try:
        run(a)
    except AssertionError as exc:
        check("out-of-box action trips the trust-region assertion",
              "trust region violated" in str(exc), str(exc))
    else:
        check("out-of-box action trips the trust-region assertion", False,
              "-> apply() accepted it silently, which is worse")


def test_clamped_input_holds():
    a = torch.zeros(4, 12)
    a[2, 5] = 66.0
    a_pol = a.clamp(-CLIP, CLIP)
    try:
        a_exec, info = run(a_pol)
    except AssertionError as exc:
        check("clamped input satisfies the trust region", False, str(exc))
        return
    dn = (a_exec - a_pol).abs().amax(dim=-1) * SCALE
    check("clamped input satisfies the trust region", True)
    check("||da||_inf <= eps for every row",
          bool((dn <= EPS + 1e-5).all()), "max %.6f" % float(dn.max()))
    check("the filter actually moved the action",
          float(dn.max()) > 0.0, "filter was inert, test proves nothing")


def test_boundary_action_only_shrinks():
    """An action pinned to the box edge can only be pulled inward."""
    a = torch.full((3, 12), CLIP)
    a_exec, _ = run(a)
    check("action at the clip boundary stays inside the box",
          bool((a_exec.abs() <= CLIP + 1e-6).all()))
    dn = (a_exec - a).abs().amax(dim=-1) * SCALE
    check("boundary displacement still respects eps",
          bool((dn <= EPS + 1e-5).all()), "max %.6f" % float(dn.max()))


def test_scale_conversion():
    """eps_raw * action_scale == epsilon_rad is what makes the bound hold."""
    filt = make_filter()
    check("eps_raw converts back to epsilon_rad",
          abs(filt.eps_raw * SCALE - EPS) < 1e-12,
          "eps_raw=%r" % filt.eps_raw)


if __name__ == "__main__":
    print("filters.apply() input contract")
    test_unclamped_input_violates()
    test_clamped_input_holds()
    test_boundary_action_only_shrinks()
    test_scale_conversion()
    print()
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        sys.exit(1)
    print("all contract tests passed")
