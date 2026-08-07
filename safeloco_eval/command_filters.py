"""Command-space safety filtering -- the steel-man baseline for E4.

Why this exists, and why it is not the same experiment as E1.

E1 filters the **action**: joint position targets.  Clearance between the body
and an obstacle does not respond to a joint command at first order -- the
command has to travel through PD control, joint torque, limb motion, foot
contact and only then base acceleration -- so the barrier has high,
contact-gated relative degree and the projection has no first-order handle to
pull on.

E4 filters the **command**: the base velocity the locomotion policy is asked
to track.  The same barrier

    h_i = ||o_i - p|| - 2 r_i - 0.35            (legged_robot.py's definition)

now has relative degree **one**, because

    h_dot_i = -<v_world, d_i>,      d_i = (o_i - p) / ||o_i - p||

and v_world is, to first order, exactly what the command asks for.  The CBF
condition h_dot >= -alpha * h is therefore a single *linear inequality in the
command*, with a closed-form projection and no learned critic anywhere:

    <R(psi) u, d_i> <= alpha h_i     ==>     a_i . u <= b_i
    a_i = R(psi)^T d_i[:2],   b_i = alpha h_i,   u = (v_x, v_y)

That is the whole point of the comparison.  Command space is where these
methods actually live, and it is a genuinely easy setting; E4 exists to give
it its best shot rather than to knock it down.

**Yaw.**  With the barrier written on the base centre, turning does not move
the base at first order, so omega_z does not appear in h_dot and the filter
reduces to braking.  That is a property of *where the barrier is written*, not
a law -- see UnicycleCBFCommandFilter, which puts the barrier on a look-ahead
point so that both v and omega enter at first order.  The braking-only filter
is retained as the yaw-disabled control arm.

**alpha** is the class-K gain.  Small alpha permits only slow approach and is
*more* conservative; large alpha is permissive, and alpha -> infinity
recovers the unfiltered policy.  Sweeping it traces the frontier.
"""

import torch

# ||da|| above which a step counts as filtered (§1 "Activation %").
ACTIVATION_CMD_THRESH = 1e-6

# --------------------------------------------------------------------------
# Barrier presets
# --------------------------------------------------------------------------
#
# `sv` reproduces legged_robot.py:1397 exactly: h = dist_3d - 2*r - 0.35.
# That expression is a *reward-shaping* signal, not a deployment constraint,
# and as a hard CBF it demands the base stay 0.95 m from every obstacle
# centre -- more than the corridor's 0.8 m minimum lateral clearance (App. G),
# so its safe set is empty at the tightest points and no command satisfies it.
#
# `geometric` matches what the collision metric actually measures: contact
# when any link comes within r_obs + BODY_COLLISION_TOL of the cylinder axis
# in XY.  Distance is taken in XY because the obstacles are vertical
# cylinders, and `body_extent` accounts for links reaching beyond the base
# the barrier is written on.
CBF_BODY_RADIUS_MARGIN = 0.35
BODY_COLLISION_TOL = 0.03
DEFAULT_BODY_EXTENT = 0.20        # Go1 base-to-outer-link, approximate

# Look-ahead geometry for the steering variant (see UnicycleCBFCommandFilter).
DEFAULT_LOOKAHEAD = 0.35          # metres ahead of the base
DEFAULT_W_V = 4.0                 # cost of changing forward speed
DEFAULT_W_OMEGA = 1.0             # cost of changing yaw rate (cheaper)

BARRIER_PRESETS = {
    # name:        (radius_mult, body_margin,                    use_xy)
    "sv":          (2.0, CBF_BODY_RADIUS_MARGIN,                 False),
    "geometric":   (1.0, BODY_COLLISION_TOL + DEFAULT_BODY_EXTENT, True),
}


class CommandStepInfo(dict):
    """Per-step, per-env command-filter diagnostics as [B] tensors."""

    @staticmethod
    def empty(n, device):
        def z(dtype):
            return torch.zeros(n, device=device, dtype=dtype)
        return CommandStepInfo(triggered=z(torch.bool),
                               dnorm_v=z(torch.float),
                               dnorm_omega=z(torch.float),
                               activated=z(torch.bool),
                               infeasible=z(torch.bool),
                               dnorm=z(torch.float),
                               min_h=z(torch.float))


def project_halfplane_box(u0, a, b, lo, hi, winv, tol=1e-6):
    """Exact W-norm projection of u0 onto {a.u <= b} INTERSECT [lo, hi], in R^2.

    Returns (u, empty), where `empty` says the intersection contains no point
    at all -- as opposed to "this solver did not find one".

    **Why this is exact rather than iterative.**  The obvious implementation
    projects onto the half-plane and clamps to the box afterwards, and it is
    wrong in a way that matters here: the clamp can push the result back out
    of the half-plane, so a command that was perfectly legal gets reported as
    "no legal option".  With lateral velocity pinned to zero and an obstacle
    off the heading, the half-plane projection spends part of its correction
    on v_y, the clamp deletes exactly that part, and v_x is left too large.
    Alternating the two projections does converge, but at a rate set by the
    angle between the sets -- in the near-tangent cases that dominate the
    no-legal-option statistic it needs hundreds of passes, so a fixed budget
    of a dozen still reports feasible problems as infeasible.

    Since the no-legal-option rate is the headline number of the command-space
    studies, the solver must not be what produces it.  In two variables the
    exact answer is closed form:

      * **emptiness** -- a.u is minimised over a box at a corner, so the
        intersection is empty iff  min_u in box (a.u)  >  b.
      * **projection** -- try the unconstrained half-plane projection clamped
        to the box; if it still violates, the optimum lies on the box
        boundary, so pin each coordinate in turn to the corner value that
        reduces a.u, solve the remaining coordinate exactly in 1-D, and keep
        the feasible candidate closest to u0.

    `winv` is the diagonal of W^-1, so cost is sum_i (u_i - u0_i)^2 / winv_i.
    """
    corner = torch.where(a > 0, lo, hi)               # argmin of a.u over box
    empty = (a * corner).sum(-1) > b + tol

    def cost(c):
        return ((c - u0) ** 2 / winv).sum(-1)

    def feasible(c):
        return (a * c).sum(-1) <= b + tol

    viol = ((a * u0).sum(-1) - b).clamp_min(0.0)
    denom = ((a * a) * winv).sum(-1).clamp_min(1e-12)
    cands = [torch.maximum(torch.minimum(
        u0 - (viol / denom).unsqueeze(-1) * (a * winv), hi), lo)]

    for j in (0, 1):
        k = 1 - j
        c = u0.clone()
        c[:, j] = corner[:, j]
        need = b - a[:, j] * c[:, j]                  # a_k u_k <= need
        ak, uk = a[:, k], u0[:, k]
        tgt = need / torch.where(ak.abs() > 1e-12, ak, torch.ones_like(ak))
        # Move u_k only as far as the constraint demands, and only inwards.
        uk = torch.where(ak > 1e-12, torch.minimum(uk, tgt),
                         torch.where(ak < -1e-12, torch.maximum(uk, tgt), uk))
        c[:, k] = torch.maximum(torch.minimum(uk, hi[k]), lo[k])
        cands.append(c)

    def scored(c):
        """Cost of a candidate, or +inf if it is not feasible."""
        raw = cost(c)
        return torch.where(feasible(c), raw, torch.full_like(raw, float("inf")))

    best, best_cost = cands[0], scored(cands[0])
    for c in cands[1:]:
        cst = scored(c)
        best = torch.where((cst < best_cost).unsqueeze(-1), c, best)
        best_cost = torch.minimum(best_cost, cst)

    # Where nothing was feasible the intersection is empty and every candidate
    # scored +inf; fall back to the corner, which is the closest a command can
    # get to legal.  `empty` is what the caller reports, not this value.
    return torch.where(torch.isinf(best_cost).unsqueeze(-1), corner, best), empty


class CommandFilter(object):
    """Interface mirroring filters.ActionFilter, but over the command."""

    name = "none"

    def apply(self, cmd, env):
        """cmd: [B, >=3] command buffer. Returns (filtered_cmd, info)."""
        raise NotImplementedError


class NoCommandFilter(CommandFilter):
    """The unfiltered row."""

    name = "none"

    def apply(self, cmd, env):
        return cmd, CommandStepInfo.empty(cmd.shape[0], cmd.device)


class CBFCommandFilter(CommandFilter):
    """First-order CBF on the commanded planar velocity.

    Solves, per environment,

        min ||u - u_nom||^2   s.t.   a_i . u <= b_i  for every obstacle i

    by cyclic projection.  Each constraint is a half-plane, so its projection
    is closed-form; cycling converges for a feasible intersection and the
    residual violation is reported when it does not.  A QP solver would give
    the same answer here and would not be worth the dependency for two
    variables.
    """

    name = "cbf_command"

    def __init__(self, alpha, max_passes=12, vx_range=None, vy_range=None,
                 barrier="geometric", radius_mult=None, body_margin=None,
                 use_xy=None):
        self.alpha = float(alpha)
        self.max_passes = int(max_passes)
        self.vx_range = vx_range
        self.vy_range = vy_range
        if barrier not in BARRIER_PRESETS:
            raise ValueError("unknown barrier preset: {}".format(barrier))
        rm, bm, xy = BARRIER_PRESETS[barrier]
        self.barrier = barrier
        self.radius_mult = float(rm if radius_mult is None else radius_mult)
        self.body_margin = float(bm if body_margin is None else body_margin)
        self.use_xy = bool(xy if use_xy is None else use_xy)

    def required_clearance(self, r_obs):
        """Centre-to-centre distance the barrier insists on, for reporting."""
        return self.radius_mult * float(r_obs) + self.body_margin

    # -- barrier ---------------------------------------------------------
    def _constraints(self, env):
        """(a, b, mask, min_h) with a: [B, N, 2] body-frame, b: [B, N]."""
        p = env.root_states[:, :3]
        delta = env.obstacle_positions - p.unsqueeze(1)          # [B, N, 3]
        # Obstacles are vertical cylinders, so the barrier that corresponds to
        # the collision test is a distance to the *axis*, in XY.
        dist = (torch.norm(delta[..., :2], dim=-1) if self.use_xy
                else torch.norm(delta, dim=-1))
        h = (dist - self.radius_mult * env.obstacle_radii.unsqueeze(-1)
             - self.body_margin)
        d = delta / (torch.norm(delta, dim=-1).unsqueeze(-1) + 1e-8)

        # Body-frame constraint normal: a = R(psi)^T d_xy.  Recovering yaw
        # from the quaternion keeps this independent of whatever heading
        # convention the task config happens to use.
        q = env.root_states[:, 3:7]
        siny = 2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1])
        cosy = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        psi = torch.atan2(siny, cosy)
        c, s = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)

        dx, dy = d[..., 0], d[..., 1]
        a = torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=-1)

        mask = env.obstacle_mask & torch.isfinite(h)
        h_masked = h.masked_fill(~mask, 1e6)
        min_h = h_masked.min(dim=-1).values
        return a, self.alpha * h, mask, min_h

    # -- projection ------------------------------------------------------
    def _box(self, device):
        """(lo, hi) of the command box, as [2] tensors.

        An unset range means "unbounded on that axis", which is what the
        pre-clamp behaviour amounted to; making it explicit lets the same
        exact projection serve both cases.
        """
        big = float("inf")
        vx = self.vx_range if self.vx_range is not None else (-big, big)
        vy = self.vy_range if self.vy_range is not None else (-big, big)
        lo = torch.tensor([vx[0], vy[0]], device=device)
        hi = torch.tensor([vx[1], vy[1]], device=device)
        return lo, hi

    def apply(self, cmd, env):
        info = CommandStepInfo.empty(cmd.shape[0], cmd.device)
        if not bool(env.has_obstacles.any()):
            return cmd, info

        with torch.no_grad():
            a, b, mask, min_h = self._constraints(env)
            info["min_h"] = min_h

            u0 = cmd[:, :2].clone()
            u = u0.clone()
            lo, hi = self._box(u.device)
            winv = torch.ones(2, device=u.device)                # unweighted

            live = env.has_obstacles.unsqueeze(-1) & mask

            # "Triggered" = the nominal command already violates a constraint,
            # i.e. the filter has work to do.  This is the denominator for
            # infeasibility: a step where the filter never needed to act says
            # nothing about whether the constraint set was satisfiable.
            v0 = (a * u0.unsqueeze(1)).sum(-1) - b
            v0 = torch.where(live, v0, torch.full_like(v0, -1.0))
            info["triggered"] = v0.max(dim=-1).values > 1e-6

            empty_any = torch.zeros_like(info["triggered"])
            for _ in range(self.max_passes):
                viol = (a * u.unsqueeze(1)).sum(-1) - b           # [B, N]
                viol = torch.where(live, viol, torch.zeros_like(viol))
                worst = viol.max(dim=-1)
                if bool((worst.values <= 1e-6).all()):
                    break
                # Project onto the single most-violated half-plane *and the
                # box together*, exactly.  Cycling one constraint at a time is
                # what makes the multi-obstacle case converge instead of
                # oscillating; doing each one exactly is what keeps a feasible
                # problem from being reported infeasible.
                idx = worst.indices
                rows = torch.arange(u.shape[0], device=u.device)
                u, empty = project_halfplane_box(
                    u, a[rows, idx], b[rows, idx], lo, hi, winv)
                empty_any = empty_any | (empty & live[rows, idx])

            resid = (a * u.unsqueeze(1)).sum(-1) - b
            resid = torch.where(live, resid, torch.full_like(resid, -1.0))
            # Infeasible means "no legal command exists", so a single
            # unsatisfiable constraint is enough; the residual catches the
            # case where each constraint is satisfiable but not jointly.
            info["infeasible"] = ((resid.max(dim=-1).values > 1e-4) | empty_any) \
                & info["triggered"]

            d_norm = (u - u0).abs().amax(dim=-1)
            info["dnorm"] = d_norm
            info["activated"] = d_norm > ACTIVATION_CMD_THRESH

            out = cmd.clone()
            out[:, :2] = u
        return out, info


def build_command_filter(variant, alpha=1.0, vx_range=None, vy_range=None,
                         max_passes=12, barrier="geometric",
                         body_margin=None, omega_range=None,
                         lookahead=DEFAULT_LOOKAHEAD, w_v=DEFAULT_W_V,
                         w_omega=DEFAULT_W_OMEGA):
    if variant in ("none", "unfiltered", None):
        return NoCommandFilter()
    if variant in ("cbf", "cbf_command", "A"):
        return CBFCommandFilter(alpha=alpha, max_passes=max_passes,
                                vx_range=vx_range, vy_range=vy_range,
                                barrier=barrier, body_margin=body_margin)
    if variant in ("unicycle", "cbf_unicycle", "Y"):
        return UnicycleCBFCommandFilter(
            alpha=alpha, max_passes=max_passes, vx_range=vx_range,
            vy_range=vy_range, barrier=barrier, body_margin=body_margin,
            omega_range=omega_range, lookahead=lookahead,
            w_v=w_v, w_omega=w_omega)
    raise ValueError("unknown command filter variant: {}".format(variant))


# ==========================================================================
# Unicycle filter with steering authority (E4-Y)
# ==========================================================================
#
# The braking-only filter above writes the barrier on the base centre, where
# h_dot = -<v_world, d> contains no yaw term: turning does not move the base
# at first order.  With a one-dimensional input the only response to "too
# close" is "stop", which is why that study saw a 79-100% no-legal-option rate
# and a robot that parked rather than steered.  That is not what OCR/ABS-style
# filters do -- they filter a twist (v, omega) on a unicycle -- so the study as
# run did not test their mechanism.
#
# The standard remedy is a **look-ahead point**.  Put the barrier on
#
#     P = p + L * e(theta),        e(theta) = (cos theta, sin theta)
#
# a fixed distance L ahead of the base.  Then
#
#     P_dot = v e + L omega e_perp
#     h_dot = -<P_dot, d> = -v <e,d> - L omega <e_perp,d>
#
# and **both** v and omega appear at first order.  The CBF condition is again
# a single linear inequality, now in two variables:
#
#     a . u <= b,   a = (<e,d>, L <e_perp,d>),  u = (v, omega),  b = alpha h
#
# L trades authority against honesty: larger L gives yaw more leverage but
# guards a point further from the body the collision test actually measures.
#
# The projection is done in a weighted metric W so the filter prefers steering
# over braking when both would satisfy the constraint -- that preference is
# the entire point of this variant and is a deliberate, reported choice.


class UnicycleCBFCommandFilter(CBFCommandFilter):
    """First-order CBF on the commanded twist, via a look-ahead point.

    Set `w_omega < w_v` to make yaw the cheaper correction.  With
    `omega_range = (0, 0)` this degenerates to the braking-only filter, which
    is how the yaw-disabled control arm is run through identical code.
    """

    name = "cbf_unicycle"

    def __init__(self, alpha, lookahead=DEFAULT_LOOKAHEAD,
                 w_v=DEFAULT_W_V, w_omega=DEFAULT_W_OMEGA,
                 omega_range=None, **kw):
        super(UnicycleCBFCommandFilter, self).__init__(alpha, **kw)
        self.lookahead = float(lookahead)
        self.w_v = float(w_v)
        self.w_omega = float(w_omega)
        self.omega_range = omega_range

    def _yaw(self, env):
        q = env.root_states[:, 3:7]
        siny = 2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1])
        cosy = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return torch.atan2(siny, cosy)

    def _box(self, device):
        """(lo, hi) of the twist box.  Axis 1 is yaw rate, not v_y.

        The box matters more here than for the braking-only filter, because
        braking and steering trade off against each other: the unconstrained
        projection routinely asks for a yaw rate past the trained range, and
        deleting that part of the correction afterwards would leave forward
        speed too high and report that the filter had no legal option.  The
        no-legal-option rate is the number this study exists to measure, so it
        must not be an artifact of how the box is applied.

        `omega_range = (0, 0)` is the yaw-disabled control arm: a degenerate
        box on the yaw axis, which is exactly the braking-only filter.
        """
        big = float("inf")
        vx = self.vx_range if self.vx_range is not None else (-big, big)
        w = self.omega_range if self.omega_range is not None else (-big, big)
        lo = torch.tensor([vx[0], w[0]], device=device)
        hi = torch.tensor([vx[1], w[1]], device=device)
        return lo, hi

    def _constraints_twist(self, env):
        """(a, b, mask, min_h) with a: [B, N, 2] over (v, omega)."""
        psi = self._yaw(env)
        c, s = torch.cos(psi), torch.sin(psi)
        e = torch.stack([c, s], dim=-1)                       # heading
        e_perp = torch.stack([-s, c], dim=-1)

        p = env.root_states[:, :2]
        P = p + self.lookahead * e                            # look-ahead point
        delta = env.obstacle_positions[..., :2] - P.unsqueeze(1)
        dist = torch.norm(delta, dim=-1)
        h = (dist - self.radius_mult * env.obstacle_radii.unsqueeze(-1)
             - self.body_margin)
        d = delta / (dist.unsqueeze(-1) + 1e-8)               # [B, N, 2]

        a_v = (e.unsqueeze(1) * d).sum(-1)                    # <e, d>
        a_w = self.lookahead * (e_perp.unsqueeze(1) * d).sum(-1)
        a = torch.stack([a_v, a_w], dim=-1)                   # [B, N, 2]

        mask = env.obstacle_mask & torch.isfinite(h)
        min_h = h.masked_fill(~mask, 1e6).min(dim=-1).values
        return a, self.alpha * h, mask, min_h

    def apply(self, cmd, env):
        info = CommandStepInfo.empty(cmd.shape[0], cmd.device)
        info["dnorm_v"] = torch.zeros_like(info["dnorm"])
        info["dnorm_omega"] = torch.zeros_like(info["dnorm"])
        if not bool(env.has_obstacles.any()):
            return cmd, info

        with torch.no_grad():
            a, b, mask, min_h = self._constraints_twist(env)
            info["min_h"] = min_h

            # u = (forward speed, yaw rate); command layout is [vx, vy, wz, ...]
            u0 = torch.stack([cmd[:, 0], cmd[:, 2]], dim=-1)
            u = u0.clone()
            live = env.has_obstacles.unsqueeze(-1) & mask

            v0 = (a * u0.unsqueeze(1)).sum(-1) - b
            v0 = torch.where(live, v0, torch.full_like(v0, -1.0))
            info["triggered"] = v0.max(dim=-1).values > 1e-6

            # Weighted projection: correcting yaw is cheaper than braking, so
            # W^-1 puts more of the step onto omega whenever both would work.
            # That preference is the whole point of this variant.
            Winv = torch.tensor([1.0 / self.w_v, 1.0 / self.w_omega],
                                device=u.device)
            lo, hi = self._box(u.device)

            empty_any = torch.zeros_like(info["triggered"])
            for _ in range(self.max_passes):
                viol = (a * u.unsqueeze(1)).sum(-1) - b
                viol = torch.where(live, viol, torch.zeros_like(viol))
                worst = viol.max(dim=-1)
                if bool((worst.values <= 1e-6).all()):
                    break
                idx = worst.indices
                rows = torch.arange(u.shape[0], device=u.device)
                u, empty = project_halfplane_box(
                    u, a[rows, idx], b[rows, idx], lo, hi, Winv)
                empty_any = empty_any | (empty & live[rows, idx])

            resid = (a * u.unsqueeze(1)).sum(-1) - b
            resid = torch.where(live, resid, torch.full_like(resid, -1.0))
            info["infeasible"] = ((resid.max(dim=-1).values > 1e-4) | empty_any) \
                & info["triggered"]

            d_v = (u[:, 0] - u0[:, 0]).abs()
            d_w = (u[:, 1] - u0[:, 1]).abs()
            info["dnorm_v"] = d_v
            info["dnorm_omega"] = d_w
            info["dnorm"] = torch.maximum(d_v, d_w)
            info["activated"] = info["dnorm"] > ACTIVATION_CMD_THRESH

            out = cmd.clone()
            out[:, 0] = u[:, 0]
            out[:, 2] = u[:, 1]
        return out, info
