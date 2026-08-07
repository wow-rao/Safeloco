"""Fall-safety margin for bipedal locomotion (E5).

The signed margin l(s) that the reachability pipeline consumes for the biped:
`Cassie._compute_safety_value` calls `compose` every step and assigns the
result to `env.safety_values`, which `WMPRunner` feeds to the safety critic's
worst-case return recursion (`rollout_storage.compute_safety_returns`).

Design (docs/bipedal_dynamic_safety.md): the *fail-set* components are
deliberately configuration-space quantities -- they detect a fall, they do not
predict one.  Prediction is the learned reachability value's job; the gap
between this margin and V_safe is the dynamic-safety content.  The optional
DCM component is the model-based densification used by the `fail_set_dcm`
ablation, and is excluded from `fail_min` so eval always sees the pure
fail-set margin.

Sign conventions match `metrics.is_fall` exactly:
  tilt_margin   < 0  <=>  proj_grav_z > FALL_TILT_PROJ_GRAV_Z  (60 deg tilt)
  height_margin < 0  <=>  h_rel < FALL_HEIGHT
so a step with fail_min < 0 is a step the shared fall definition would call
fallen (up to the contact floor, which has no is_fall analogue: is_fall reads
the terminal pose, the margin has to fire while the episode is still alive).

Torch-only: importable without isaacgym, so the exact production math is
unit-tested offline (tests/test_fall_margin.py) and reusable by analyze_e5.
"""

import torch

from safeloco_eval.metrics import FALL_TILT_PROJ_GRAV_Z, FALL_HEIGHT

GRAVITY = 9.81

#: Sentinel for "component not active"; never binds the min (margins live in
#: [-3, 1.5]) -- same convention as `_build_obstacle_tensor`'s 10.0.
INACTIVE = 10.0


def tilt_margin(proj_grav_z, g_z_thresh=FALL_TILT_PROJ_GRAV_Z):
    """Normalized tilt margin from the projected-gravity z component.

    proj_grav_z is -1 upright, 0 at 90 degrees of tilt, +1 inverted.
    Returns +1 upright, 0 at the fall threshold (60 deg for -0.5), negative
    beyond (-1 at 90 deg, -3 inverted).
    """
    return (g_z_thresh - proj_grav_z) / (1.0 + g_z_thresh)


def height_margin(h_rel, h_fall=FALL_HEIGHT, h_nom=0.85, clamp_max=1.5):
    """Normalized base-height margin: 1 at nominal height, 0 at h_fall.

    h_rel is the base height above the local terrain.  Clamped above so a
    flight phase cannot dominate the min through the compose clamp anyway.
    """
    return ((h_rel - h_fall) / (h_nom - h_fall)).clamp(max=clamp_max)


def contact_margin(illegal_contact):
    """Hard floor: -1 wherever a non-foot body carries ground contact.

    A fallen biped is unambiguously unsafe no matter what the pose margins
    say, so this overrides them through the min.
    """
    return torch.where(
        illegal_contact,
        torch.full_like(illegal_contact, -1.0, dtype=torch.float),
        torch.full_like(illegal_contact, INACTIVE, dtype=torch.float),
    )


def dcm_margin(base_xy, vel_xy, h_rel, feet_mid_xy, r_cap=0.5,
               z_c_min=0.3, z_c_max=1.2):
    """One-step capturability margin on the divergent component of motion.

    xi = p_xy + v_xy / omega0 with omega0 = sqrt(g / z_c) is the instantaneous
    capture point of the linear inverted pendulum; the margin is positive while
    xi stays within r_cap of the support midpoint (the reachable foothold
    disc), zero on its boundary, negative outside.

    This is the paper's spatial-temporal margin l = h + alpha_h * h_dot with
    the time constant fixed by the pendulum (alpha_h = 1/omega0) instead of
    hand-tuned.
    """
    z_c = h_rel.clamp(min=z_c_min, max=z_c_max)
    omega0 = torch.sqrt(GRAVITY / z_c)
    xi = base_xy + vel_xy / omega0.unsqueeze(-1)
    return (r_cap - torch.norm(xi - feet_mid_xy, dim=-1)) / r_cap


def compose(fail_components, dcm=None, clamp_max=0.5):
    """Aggregate margins: sv = min over components, clamped for the pipeline.

    Returns (sv, fail_min):
      sv        min over fail-set components and (if given) the DCM term,
                clamped at clamp_max -- what env.safety_values carries.
      fail_min  min over the fail-set components only, unclamped -- the pure
                fallen-state margin, exported as extras["fall_margin_min"]
                (the biped analogue of cbf_h_min).
    """
    fail_min = torch.min(torch.stack(fail_components, dim=0), dim=0).values
    sv = fail_min if dcm is None else torch.minimum(fail_min, dcm)
    return sv.clamp(max=clamp_max), fail_min
