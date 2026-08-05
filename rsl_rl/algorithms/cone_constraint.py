import math
import torch
import torch.nn.functional as F


# Until this was fixed, `damped_null_space_project` overwrote whatever the
# caller passed with these two values, so every policy trained in this repo
# used them regardless of what its config said.  They stay the default so
# existing results reproduce; set `d_safe` / `d_danger` in the algorithm config
# to vary them.
DEFAULT_D_SAFE = -0.4
DEFAULT_D_DANGER = -0.6


def cone_constrained_update(g_task, g_safe, alpha_deg=60.0):
    """Project g_task into the cone of half-angle alpha_deg around g_safe.

    g_safe is the cone axis only -- it is NOT added to the output.
    The magnitude of g_task's parallel component is preserved exactly;
    only the perpendicular component is clipped when outside the cone.

    Returns:
        g_final: same shape as g_task.
    """
    eps = 1e-8
    g_safe_norm = torch.norm(g_safe)
    if g_safe_norm < eps:
        return g_task

    alpha = math.radians(alpha_deg)
    tan_alpha = math.tan(alpha)

    g_safe_hat = g_safe / g_safe_norm
    parallel_mag = torch.dot(g_task, g_safe_hat)
    parallel_vec = parallel_mag * g_safe_hat
    perp_vec = g_task - parallel_vec
    perp_mag = torch.norm(perp_vec)

    if parallel_mag > 0 and perp_mag <= tan_alpha * parallel_mag + eps:
        return g_task

    if parallel_mag <= 0:
        return perp_vec

    perp_hat = perp_vec / (perp_mag + eps)
    return parallel_vec + tan_alpha * parallel_mag * perp_hat


def dynamic_alpha(h_min, d_min=0.0, d_free=2.0, alpha_max_deg=70.0):
    """Cone half-angle as a function of min CBF margin h_min.

    h_min <= d_min  ->  alpha = 0   (must align with g_safe exactly)
    h_min >= d_free ->  alpha = alpha_max  (task dominates)
    """
    t = (h_min - d_min) / max(d_free - d_min, 1e-6)
    t = max(0.0, min(1.0, float(t)))
    return alpha_max_deg * t


def damped_null_space_project(g_task, g_safety, safety_value_batch,
                              d_safe=DEFAULT_D_SAFE, d_danger=DEFAULT_D_DANGER):
    """Project g_safety into the null space of g_task with damping.

    When safety_value > d_safe: full null projection (task dominates).
    When safety_value < d_danger: no projection (safety dominates).
    Between: linear interpolation.

    Both thresholds are honoured exactly as passed.  `d_safe` must be strictly
    greater than `d_danger` -- the ramp runs upward from d_danger to d_safe,
    and inverting them silently reverses the safety response.
    """
    eps = 1e-8

    if d_safe <= d_danger:
        raise ValueError(
            "d_safe (%r) must be > d_danger (%r); the damping ramp runs "
            "from d_danger up to d_safe." % (d_safe, d_danger))

    if safety_value_batch.dim() > 0:
        sv_min = safety_value_batch.min().item()
    else:
        sv_min = safety_value_batch.item()

    t = (sv_min - d_danger) / (d_safe - d_danger + eps)
    t = max(0.0, min(1.0, t))

    g_task_norm_sq = torch.dot(g_task, g_task)
    if g_task_norm_sq < eps:
        return g_safety

    proj_coeff = torch.dot(g_safety, g_task) / g_task_norm_sq
    g_safety_null = g_safety - proj_coeff * g_task

    g_safety_projected = t * g_safety_null + (1.0 - t) * g_safety
    return g_safety_projected


def extract_flat_grads(model):
    """Concatenate all parameter gradients into a single flat vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def write_flat_grads(model, g_flat):
    """Write a flat gradient vector back into each parameter's .grad field."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = g_flat[idx:idx + numel].reshape(p.shape).clone()
        idx += numel
