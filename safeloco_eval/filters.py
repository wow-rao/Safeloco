"""Deployment-time action filters.

E1 (experiment_protocol_phased.md §3) instantiates two variants of joint-space
projection against the learned Q_safe, plus the unfiltered reference:

  A  Sampling projection  -- K=64 uniform samples in the ||a - a_pol||_inf <= eps
                            box, execute the *minimum-distortion* sample whose
                            Q_hat clears Q_hat(o, a_pol) + delta; if none does,
                            execute the max-Q_hat sample and log a
                            **target-miss event**.
  B  Gradient projection  -- m = 3 steps of a <- clip_box(a + eta grad_a Q_hat),
                            eta = eps/3.  Run at eps in {0.1, 0.5} to preempt
                            "sampling is a strawman".

Units.  Every epsilon in the protocol is **radians of joint command**.  The env
turns a raw action into a joint target via `action_scale` (0.25 for go1), so
the box in raw action units is eps / action_scale.  Filters take eps in rad,
convert internally, and report ||da|| back in rad.  There is no tanh on this
actor (`actor_critic_wmp.py:163` has it commented out); the raw action box is
`normalization.clip_actions` = 6.0.

Trust region.  The *constraint* is the infinity-norm box (that is what eps
bounds).  The *objective* ranking candidates is the L2 distortion
||a - a_pol||_2, which is what "minimum-distortion" means for a projection.
Both norms are logged.

Phase 2's analytic joint-limit clamp (J1) lands in this same file, which is why
it is part of the verbatim-copied shared module rather than E1-local code.
"""

import torch

from .metrics import ACTIVATION_DNORM_THRESH


def _index(s_feat, idx):
    """Index state features, transparently for the ensemble (list) case."""
    if isinstance(s_feat, (list, tuple)):
        return [f[idx] for f in s_feat]
    return s_feat[idx]


class StepInfo(dict):
    """Per-step, per-env filter diagnostics as [B] tensors."""

    @staticmethod
    def empty(n, device):
        def z(dtype):
            return torch.zeros(n, device=device, dtype=dtype)
        return StepInfo(triggered=z(torch.bool), activated=z(torch.bool),
                        target_miss=z(torch.bool), dnorm_inf=z(torch.float),
                        dnorm_l2=z(torch.float), q_gap=z(torch.float),
                        q_gap_valid=z(torch.bool))


class ActionFilter(object):
    """Base class.  `apply` receives the *raw* policy action (already clipped
    to the env's action box, so ||da|| is exactly what the env executes) and
    returns the raw action to execute plus diagnostics."""

    name = "none"
    epsilon_rad = 0.0
    always_on = False
    tau = float("-inf")

    def apply(self, a_pol, s_feat, trigger_value):
        raise NotImplementedError

    def _trigger(self, trigger_value, n, device):
        if self.always_on:
            return torch.ones(n, dtype=torch.bool, device=device)
        return trigger_value < self.tau


class NoFilter(ActionFilter):
    """Filter disabled.  Must reproduce unfiltered pi_nom exactly -- the first
    of the §5 sanity checks.

    If a `tau` is supplied it still *logs* when the trigger would have fired,
    without acting on it.  That makes the unfiltered row carry the counterfactual
    trigger rate, and makes "activation % is 0 when the trigger never fires"
    an observation rather than a tautology.
    """

    name = "none"

    def __init__(self, tau=None):
        self.log_tau = tau

    def apply(self, a_pol, s_feat, trigger_value):
        info = StepInfo.empty(a_pol.shape[0], a_pol.device)
        if self.log_tau is not None and trigger_value is not None:
            info["triggered"] = trigger_value < self.log_tau
        return a_pol, info


class _QFilter(ActionFilter):
    def __init__(self, qsafe, epsilon_rad, action_scale, clip_actions,
                 tau=-0.25, delta=None, always_on=False, device="cuda:0",
                 seed=0):
        self.qsafe = qsafe
        self.epsilon_rad = float(epsilon_rad)
        self.action_scale = float(action_scale)
        self.eps_raw = self.epsilon_rad / self.action_scale
        self.clip_actions = float(clip_actions)
        self.tau = float(tau)
        self.delta = float(qsafe.delta if delta is None else delta)
        self.always_on = bool(always_on)
        self.device = device
        # Deterministic sampling seeds (§3 "Pitfalls"): a filter-local
        # generator, so re-running a sweep point reproduces its perturbations
        # exactly and independently of whatever else consumes the global RNG.
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))

    def _finish(self, info, idx, a_pol_sub, a_exec_sub, q_pol, q_exec, miss):
        delta_sub = a_exec_sub - a_pol_sub
        dn_inf = delta_sub.abs().amax(dim=-1) * self.action_scale
        dn_l2 = delta_sub.norm(dim=-1) * self.action_scale
        info["activated"][idx] = dn_inf > ACTIVATION_DNORM_THRESH
        info["dnorm_inf"][idx] = dn_inf
        info["dnorm_l2"][idx] = dn_l2
        info["q_gap"][idx] = q_exec - q_pol
        info["q_gap_valid"][idx] = True
        info["target_miss"][idx] = miss
        # §5: ||da||_inf <= eps must hold for every filtered step.
        assert float(dn_inf.max()) <= self.epsilon_rad + 1e-5, (
            "trust region violated: {:.6f} > eps {:.6f}".format(
                float(dn_inf.max()), self.epsilon_rad))


class SamplingProjectionFilter(_QFilter):
    """E1-A (primary).

    Trigger V_hat(o) < tau; K uniform samples in the eps-box; execute the
    minimum-distortion sample clearing Q_hat(o, a_pol) + delta; otherwise
    execute max-Q_hat and log a target-miss.

    a_pol itself is deliberately **not** among the K candidates: with delta > 0
    it could never clear its own margin, and including it in the max-Q_hat
    fallback would convert the fallback into a no-op.  Executing a corrective
    action that does not actually clear the margin is the behaviour under test
    -- it is what produces the large-epsilon destabilisation regime.
    """

    name = "A_sampling"

    def __init__(self, *args, k_samples=64, **kwargs):
        super(SamplingProjectionFilter, self).__init__(*args, **kwargs)
        self.k_samples = int(k_samples)

    def apply(self, a_pol, s_feat, trigger_value):
        B, A = a_pol.shape
        info = StepInfo.empty(B, a_pol.device)
        trig = self._trigger(trigger_value, B, a_pol.device)
        info["triggered"] = trig
        idx = trig.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return a_pol, info

        with torch.no_grad():
            ap = a_pol[idx]
            sf = _index(s_feat, idx)
            n, K = idx.numel(), self.k_samples

            u = (torch.rand((n, K, A), generator=self.gen,
                            device=a_pol.device) * 2.0 - 1.0) * self.eps_raw
            cand = (ap.unsqueeze(1) + u).clamp(-self.clip_actions,
                                               self.clip_actions)

            q_cand = self.qsafe.q_from_features(sf, cand)          # [n, K]
            q_pol = self.qsafe.q_from_features(sf, ap)             # [n]

            dist = (cand - ap.unsqueeze(1)).norm(dim=-1)           # [n, K]
            ok = q_cand >= (q_pol.unsqueeze(1) + self.delta)
            miss = ~ok.any(dim=1)

            masked = torch.where(ok, dist, torch.full_like(dist, float("inf")))
            best = masked.argmin(dim=1)
            best = torch.where(miss, q_cand.argmax(dim=1), best)

            rows = torch.arange(n, device=a_pol.device)
            a_exec_sub = cand[rows, best]
            q_exec = q_cand[rows, best]

            a_exec = a_pol.clone()
            a_exec[idx] = a_exec_sub
            self._finish(info, idx, ap, a_exec_sub, q_pol, q_exec, miss)
        return a_exec, info


class GradientProjectionFilter(_QFilter):
    """E1-B.  m gradient-ascent steps on Q_hat, projected back into the
    eps-box after each one.

    `grad_normalize='inf'` (default) rescales the gradient to unit
    infinity-norm so each step moves exactly eta = eps/m -- the *charitable*
    reading, since a raw gradient whose magnitude happens to be tiny would make
    B a no-op and B exists precisely to rule out "sampling is a strawman".
    Pass `grad_normalize='none'` for the literal raw-gradient form.

    Target-miss for B is the same margin test applied to the outcome: the final
    action failed to reach Q_hat(o, a_pol) + delta.  This keeps the diagnostic
    comparable with A even though B has no explicit acceptance test.
    """

    name = "B_gradient"

    def __init__(self, *args, m_steps=3, grad_normalize="inf", **kwargs):
        super(GradientProjectionFilter, self).__init__(*args, **kwargs)
        self.m_steps = int(m_steps)
        self.grad_normalize = grad_normalize
        self.eta_raw = self.eps_raw / max(self.m_steps, 1)

    def apply(self, a_pol, s_feat, trigger_value):
        B, _ = a_pol.shape
        info = StepInfo.empty(B, a_pol.device)
        trig = self._trigger(trigger_value, B, a_pol.device)
        info["triggered"] = trig
        idx = trig.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return a_pol, info

        ap = a_pol[idx].detach()
        sf = _index(s_feat, idx)
        lo = (ap - self.eps_raw).clamp(-self.clip_actions, self.clip_actions)
        hi = (ap + self.eps_raw).clamp(-self.clip_actions, self.clip_actions)

        with torch.no_grad():
            q_pol = self.qsafe.q_from_features(sf, ap)

        a = ap.clone()
        with torch.enable_grad():
            for _ in range(self.m_steps):
                a_var = a.detach().requires_grad_(True)
                q = self.qsafe.q_from_features(sf, a_var).sum()
                g, = torch.autograd.grad(q, a_var)
                if self.grad_normalize == "inf":
                    g = g / (g.abs().amax(dim=-1, keepdim=True) + 1e-8)
                a = (a_var.detach() + self.eta_raw * g).clamp(min=lo, max=hi)

        with torch.no_grad():
            a = a.detach()
            q_exec = self.qsafe.q_from_features(sf, a)
            miss = q_exec < (q_pol + self.delta)
            a_exec = a_pol.clone()
            a_exec[idx] = a
            self._finish(info, idx, ap, a, q_pol, q_exec, miss)
        return a_exec, info


# ---------------------------------------------------------------------------

def build_filter(variant, qsafe, epsilon_rad, action_scale, clip_actions,
                 tau=-0.25, delta=None, always_on=False, k_samples=64,
                 m_steps=3, grad_normalize="inf", device="cuda:0", seed=0):
    """Factory used by run_e1.py; `variant='none'` is the unfiltered row."""
    if variant in ("none", "unfiltered", None):
        return NoFilter(tau=(tau if qsafe is not None else None))
    kw = dict(qsafe=qsafe, epsilon_rad=epsilon_rad, action_scale=action_scale,
              clip_actions=clip_actions, tau=tau, delta=delta,
              always_on=always_on, device=device, seed=seed)
    if variant in ("A", "A_sampling", "sampling"):
        return SamplingProjectionFilter(k_samples=k_samples, **kw)
    if variant in ("B", "B_gradient", "gradient"):
        return GradientProjectionFilter(m_steps=m_steps,
                                        grad_normalize=grad_normalize, **kw)
    raise ValueError("unknown filter variant: {}".format(variant))
