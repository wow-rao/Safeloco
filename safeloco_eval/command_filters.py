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

**What is not filtered.**  Yaw rate is left alone.  Turning does not move the
base at first order, so omega_z has relative degree two with respect to h and
cannot appear in a first-order CBF constraint.  Filtering it would be
unsound, not conservative.

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
                               activated=z(torch.bool),
                               infeasible=z(torch.bool),
                               dnorm=z(torch.float),
                               min_h=z(torch.float))


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
    def apply(self, cmd, env):
        info = CommandStepInfo.empty(cmd.shape[0], cmd.device)
        if not bool(env.has_obstacles.any()):
            return cmd, info

        with torch.no_grad():
            a, b, mask, min_h = self._constraints(env)
            info["min_h"] = min_h

            u0 = cmd[:, :2].clone()
            u = u0.clone()
            a_sq = (a * a).sum(-1).clamp_min(1e-8)               # [B, N]

            live = env.has_obstacles.unsqueeze(-1) & mask

            # "Triggered" = the nominal command already violates a constraint,
            # i.e. the filter has work to do.  This is the denominator for
            # infeasibility: a step where the filter never needed to act says
            # nothing about whether the constraint set was satisfiable.
            v0 = (a * u0.unsqueeze(1)).sum(-1) - b
            v0 = torch.where(live, v0, torch.full_like(v0, -1.0))
            info["triggered"] = v0.max(dim=-1).values > 1e-6

            for _ in range(self.max_passes):
                viol = (a * u.unsqueeze(1)).sum(-1) - b           # [B, N]
                viol = torch.where(live, viol, torch.zeros_like(viol))
                worst = viol.max(dim=-1)
                if bool((worst.values <= 1e-6).all()):
                    break
                # Project onto the single most-violated half-plane, then
                # re-evaluate: cycling one constraint at a time is what makes
                # the multi-obstacle case converge instead of oscillating.
                idx = worst.indices
                rows = torch.arange(u.shape[0], device=u.device)
                a_w = a[rows, idx]                                # [B, 2]
                step = (worst.values / a_sq[rows, idx]).clamp_min(0.0)
                u = u - step.unsqueeze(-1) * a_w

            # Clamp *before* measuring the residual: the clamp is part of the
            # filter, so feasibility has to be judged on the command that is
            # actually executed.  Measuring it pre-clamp let infeasible_steps
            # exceed activation_steps, which is not a rate.
            if self.vx_range is not None:
                u[:, 0] = u[:, 0].clamp(self.vx_range[0], self.vx_range[1])
            if self.vy_range is not None:
                u[:, 1] = u[:, 1].clamp(self.vy_range[0], self.vy_range[1])

            resid = (a * u.unsqueeze(1)).sum(-1) - b
            resid = torch.where(live, resid, torch.full_like(resid, -1.0))
            info["infeasible"] = (resid.max(dim=-1).values > 1e-4) & info["triggered"]

            d_norm = (u - u0).abs().amax(dim=-1)
            info["dnorm"] = d_norm
            info["activated"] = d_norm > ACTIVATION_CMD_THRESH

            out = cmd.clone()
            out[:, :2] = u
        return out, info


def build_command_filter(variant, alpha=1.0, vx_range=None, vy_range=None,
                         max_passes=12, barrier="geometric",
                         body_margin=None):
    if variant in ("none", "unfiltered", None):
        return NoCommandFilter()
    if variant in ("cbf", "cbf_command", "A"):
        return CBFCommandFilter(alpha=alpha, max_passes=max_passes,
                                vx_range=vx_range, vy_range=vy_range,
                                barrier=barrier, body_margin=body_margin)
    raise ValueError("unknown command filter variant: {}".format(variant))
