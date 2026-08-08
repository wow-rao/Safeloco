"""The custom l-value for box (step-over) obstacles -- pure torch.

Why the cylinder safety value cannot serve here
-----------------------------------------------
`legged_robot._compute_safety_value` scores clearance as a radial distance to
an obstacle *point*: h = ||o - p|| - 2r - 0.35.  That failure set is a solid
of revolution -- it has no notion of "over".  On the step-trap terrain the
whole argument is that the space *above* a low sill is safe, so the margin
function must be three-dimensional or the trained policy would refuse the
crossing exactly as a planar controller does.

The l-value
-----------
In the Hamilton-Jacobi convention, l(x) is a margin whose sub-zero level set
is the failure set: l(x) >= 0 iff the state is outside it.  Here the failure
set is "any tracked keypoint of the robot inside a wall volume, inflated by
that keypoint's margin":

    l(x) = min over keypoints k, boxes B of [ sdf_B(p_k) - margin_k ]

with sdf_B the exact signed distance to the axis-aligned box B.  For a point
above a box, sdf is the height above its top -- so a swing foot at
sill_top + margin has l >= 0 and the safe set contains the step-over
trajectory.  The 0.4 m walls stay un-crossable through the same formula: the
base keypoint would have to pass through their inflated volume (they are
taller than the base runs), so every crossing path dips l < 0.

Keypoints are the four feet, the four calf links (the knee is the leading
edge of a swing leg and hits a sill first), and the base.  Margins grow with
how much unmodelled body surrounds each point.

The published value keeps the codebase's CBF-style shape

    sv = beta * l_dot + l,     clamped to <= clamp_max

with l_dot computed from the analytic SDF gradient dotted with the keypoint
velocity, so the safety critic sees approach speed exactly as it does on the
cylinder terrains, and everything downstream (safety critic regression,
advantage, the damped null-space projection in `cone_constraint.py`) is
unchanged.

This module deliberately imports nothing beyond torch, so the offline tests
can exercise the exact production code without isaacgym.
"""

import torch

# Defaults, overridable from `cfg.lvalue` (go1_amp_config.py).
BETA = 0.3            # weight on l_dot, matching the cylinder sv
CLAMP_MAX = 0.5       # matching the cylinder sv
FOOT_MARGIN = 0.03    # feet legitimately skim the sill; keep this tight
CALF_MARGIN = 0.05    # knee/calf leading edge
BASE_MARGIN = 0.10    # base half-height plus clearance

_EPS = 1e-8


def box_sdf(points, boxes):
    """Exact signed distance from points to axis-aligned boxes.

    points: [..., K, 3]; boxes: [..., M, 6] as (cx, cy, cz, hx, hy, hz).
    Returns sdf [..., K, M].  Positive outside, negative inside; for a point
    directly above a box, this is the height above its top face.
    """
    p = points.unsqueeze(-2)                      # [..., K, 1, 3]
    c = boxes[..., :3].unsqueeze(-3)              # [..., 1, M, 3]
    e = boxes[..., 3:].unsqueeze(-3)              # [..., 1, M, 3]
    q = (p - c).abs() - e                         # [..., K, M, 3]
    outside = q.clamp(min=0.0)
    dist_out = outside.norm(dim=-1)
    max_q = q.max(dim=-1).values
    return dist_out + max_q.clamp(max=0.0)


def box_sdf_gradient(points, boxes):
    """d sdf / d point, [..., K, M, 3] -- exact a.e., piecewise smooth.

    Outside: the unit vector from the closest box point.  Inside: the unit
    normal of the nearest face (one-hot on the largest q component).  Ties and
    the measure-zero surface are resolved arbitrarily but with unit norm,
    which is all a rate term needs.
    """
    p = points.unsqueeze(-2)
    c = boxes[..., :3].unsqueeze(-3)
    e = boxes[..., 3:].unsqueeze(-3)
    d = p - c
    q = d.abs() - e
    outside = q.clamp(min=0.0)
    dist_out = outside.norm(dim=-1, keepdim=True)
    sgn = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))

    grad_out = sgn * outside / (dist_out + _EPS)

    nearest_face = torch.nn.functional.one_hot(
        q.argmax(dim=-1), num_classes=3).to(q.dtype)
    grad_in = sgn * nearest_face

    is_outside = (dist_out > _EPS)
    return torch.where(is_outside, grad_out, grad_in)


def box_lvalue(points, velocities, margins, boxes, box_mask,
               beta=BETA, clamp_max=CLAMP_MAX):
    """The l-value and its CBF-style safety value, batched over envs.

    points, velocities: [E, K, 3] keypoint positions / world velocities.
    margins:            [K] per-keypoint inflation.
    boxes:              [E, M, 6]; box_mask: [E, M] (padding is masked out).
    Returns (sv, l_min): both [E].
        l_min = min_k,B sdf - margin          (the pure margin, no rate term)
        sv    = min_k,B (beta * l_dot + l),   clamped to <= clamp_max
    Envs whose box_mask is all-False return (clamp_max, big) -- i.e. "safe",
    matching the cylinder path's convention for obstacle-free envs.
    """
    sdf = box_sdf(points, boxes)                          # [E, K, M]
    l = sdf - margins.view(1, -1, 1)

    grad = box_sdf_gradient(points, boxes)                # [E, K, M, 3]
    l_dot = (grad * velocities.unsqueeze(-2)).sum(dim=-1)  # [E, K, M]

    sv = beta * l_dot + l

    big = torch.tensor(1e6, device=points.device, dtype=points.dtype)
    live = box_mask.unsqueeze(1)                          # [E, 1, M]
    l_min = l.masked_fill(~live, big).flatten(1).min(dim=1).values
    sv_min = sv.masked_fill(~live, big).flatten(1).min(dim=1).values

    has_box = box_mask.any(dim=1)
    l_min = torch.where(has_box, l_min, big)
    sv_min = torch.where(has_box, sv_min, big)
    return sv_min.clamp(max=clamp_max), l_min


def any_point_in_boxes(points, boxes, box_mask, tol=0.0):
    """[E] bool -- does any point sit inside any (tol-inflated) box volume?

    The box analogue of `link_obstacle_collision`: contact with a wall is a
    3-D volume test, so a link that passes *over* a sill with clearance > tol
    does not count -- which is exactly the distinction this terrain exists to
    measure.  points: [E, B, 3], boxes: [E, M, 6], box_mask: [E, M].
    """
    inside = box_sdf(points, boxes) < tol                 # [E, B, M]
    inside = inside & box_mask.unsqueeze(1)
    return inside.any(dim=-1).any(dim=-1)
