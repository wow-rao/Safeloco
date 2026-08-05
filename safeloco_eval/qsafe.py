"""Q_safe -- the learned joint-space action-value the E1 filter projects against.

Phase 1's structural premise (experiment_protocol_phased.md §"Structural
preface"): for collisions **no analytic joint-space barrier exists**, because
joint commands reach base clearance only through PD -> torques -> contact
forces -> base motion.  Every Phase-1 filter therefore has to target a
*learned* action-value.  This module is that learned substitute.

Construction follows viploco_codebase_addendum.md §3.2: clone the existing
`ActorCriticWMP.safety_critic` head (plus its own world-model feature
encoder), add the 12-D action to the input layer, and regress the existing
`safety_returns` targets.  No new target machinery.

Two implementation notes that matter:

1. **The first layer is factorised.**  `W @ [s; a] == W_s @ s + W_a @ a`, so
   the state pass runs once per env and the K=64 action samples only pay for
   `W_a @ a` plus the trunk.  This is exact, not an approximation, and it is
   what makes a 64-sample filter affordable at 40-50 Hz across hundreds of
   envs.

2. **A state-only head V_hat is trained alongside Q_hat on the same targets.**
   pi_nom is trained with `safety_coef = 0`, and `AMPPPO.update` gates the
   whole safety branch behind `if self.safety_coef > 0`, so the checkpoint's
   own `safety_critic` has *never received a gradient*.  Triggering on it
   would be triggering on noise.  V_hat here is the trigger signal.
"""

import torch
import torch.nn as nn


def _mlp(dims, activation, final_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


class SafetyActionValue(nn.Module):
    """Q_hat(o, a) and V_hat(o), sharing the observation encoders.

    Mirrors `ActorCriticWMP.safety_critic` / `safety_critic_wm_feature_encoder`
    / `grid_encoder_critic` in architecture and activation (tanh), including
    the `clamp(max=0.5)` on the output so the value range matches V_hat's.
    """

    def __init__(self,
                 num_critic_obs,
                 num_actions,
                 wm_feature_dim,
                 wm_encoder_hidden_dims=(64, 64),
                 wm_latent_dim=32,
                 critic_hidden_dims=(512, 256, 128),
                 grid_enabled=False,
                 grid_H=60,
                 grid_W=30,
                 grid_latent_dim=32,
                 value_clamp_max=0.5):
        super().__init__()
        act = nn.Tanh
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        self.wm_feature_dim = wm_feature_dim
        self.grid_enabled = grid_enabled
        self.value_clamp_max = value_clamp_max

        # --- world-model feature encoder (same shape as the safety critic's)
        wm_dims = [wm_feature_dim] + list(wm_encoder_hidden_dims) + [wm_latent_dim]
        self.wm_encoder = _mlp(wm_dims, act)

        # --- occupancy-grid encoder (same conv stack as grid_encoder_critic)
        if grid_enabled:
            def _conv_out(size, kernel, stride, padding):
                return (size + 2 * padding - kernel) // stride + 1

            h2 = _conv_out(_conv_out(grid_H, 5, 2, 2), 3, 2, 1)
            w2 = _conv_out(_conv_out(grid_W, 5, 2, 2), 3, 2, 1)
            self.grid_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
                nn.Flatten(),
                nn.Linear(32 * h2 * w2, 128), nn.ELU(),
                nn.Linear(128, grid_latent_dim),
            )
            self.grid_latent_dim = grid_latent_dim
        else:
            self.grid_latent_dim = 0
        self.grid_H, self.grid_W = grid_H, grid_W

        state_dim = num_critic_obs + wm_latent_dim + self.grid_latent_dim
        self.state_dim = state_dim
        h = list(critic_hidden_dims)

        # --- Q head, first layer factorised over (state, action)
        self.q_state_proj = nn.Linear(state_dim, h[0])
        self.q_action_proj = nn.Linear(num_actions, h[0], bias=False)
        self.q_trunk = nn.Sequential(act(), *_mlp(h + [1], act))

        # --- V head, state only
        self.v_head = _mlp([state_dim] + h + [1], act)

        # --- frozen input standardisation, filled by fit_normalizer()
        self.register_buffer("obs_mean", torch.zeros(num_critic_obs))
        self.register_buffer("obs_std", torch.ones(num_critic_obs))
        self.register_buffer("wm_mean", torch.zeros(wm_feature_dim))
        self.register_buffer("wm_std", torch.ones(wm_feature_dim))
        #: 0.05 * IQR of Q_hat over the training set -- the filter's delta
        #: margin (experiment_protocol_phased.md §3).  Set at training time so
        #: the deployed filter is fully determined by the checkpoint.
        self.register_buffer("q_iqr", torch.tensor(1.0))

    # ------------------------------------------------------------------
    def fit_normalizer(self, critic_obs, wm_feature, eps=1e-3):
        with torch.no_grad():
            self.obs_mean.copy_(critic_obs.mean(0))
            self.obs_std.copy_(critic_obs.std(0).clamp(min=eps))
            self.wm_mean.copy_(wm_feature.mean(0))
            self.wm_std.copy_(wm_feature.std(0).clamp(min=eps))

    def state_features(self, critic_obs, wm_feature, grid=None):
        """Encode everything that does not depend on the candidate action.

        Called **once** per control step; the K action samples reuse the
        result.  Shape [B, state_dim].
        """
        obs_n = (critic_obs - self.obs_mean) / self.obs_std
        wm_n = (wm_feature - self.wm_mean) / self.wm_std
        parts = [obs_n, self.wm_encoder(wm_n)]
        if self.grid_enabled and grid is not None:
            parts.append(self.grid_encoder(grid.unsqueeze(1)))
        elif self.grid_enabled:
            parts.append(torch.zeros(critic_obs.shape[0], self.grid_latent_dim,
                                     device=critic_obs.device))
        return torch.cat(parts, dim=-1)

    def q_from_features(self, s_feat, actions):
        """Q_hat for [B, A] or [B, K, A] actions against [B, state_dim].

        The state term is computed once and broadcast over K; this is an
        algebraic identity with a single un-factorised Linear, not an
        approximation.
        """
        s_term = self.q_state_proj(s_feat)
        if actions.dim() == 3:
            a_term = self.q_action_proj(actions)          # [B, K, H]
            pre = s_term.unsqueeze(1) + a_term
            q = self.q_trunk(pre).squeeze(-1)             # [B, K]
        else:
            q = self.q_trunk(s_term + self.q_action_proj(actions)).squeeze(-1)
        return q.clamp(max=self.value_clamp_max)

    def v_from_features(self, s_feat):
        return self.v_head(s_feat).squeeze(-1).clamp(max=self.value_clamp_max)

    def forward(self, critic_obs, wm_feature, actions, grid=None):
        return self.q_from_features(
            self.state_features(critic_obs, wm_feature, grid), actions)

    # ------------------------------------------------------------------
    @property
    def delta(self):
        """delta = 0.05 * IQR(Q_hat) -- the margin a sample must clear."""
        return 0.05 * float(self.q_iqr)

    def save(self, path, meta=None):
        torch.save({"state_dict": self.state_dict(),
                    "config": self.config(),
                    "meta": meta or {}}, path)

    def config(self):
        return dict(num_critic_obs=self.num_critic_obs,
                    num_actions=self.num_actions,
                    wm_feature_dim=self.wm_feature_dim,
                    wm_latent_dim=self.wm_encoder[-1].out_features,
                    grid_enabled=self.grid_enabled,
                    grid_H=self.grid_H,
                    grid_W=self.grid_W,
                    grid_latent_dim=self.grid_latent_dim or 32,
                    value_clamp_max=self.value_clamp_max)

    @staticmethod
    def load(path, device="cpu", **overrides):
        blob = torch.load(path, map_location=device)
        cfg = dict(blob["config"])
        cfg.update(overrides)
        model = SafetyActionValue(**cfg).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, blob.get("meta", {})


class QSafeEnsemble(nn.Module):
    """Min-aggregated ensemble -- the *control* for the primary single head.

    The protocol asks for a single head as the primary (what a naive
    practitioner builds) and a 3-head min-aggregated ensemble reported at one
    epsilon, so a reviewer cannot attribute E1's outcome to a wobbly critic.
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.grid_enabled = members[0].grid_enabled
        self.value_clamp_max = members[0].value_clamp_max

    def state_features(self, critic_obs, wm_feature, grid=None):
        return [m.state_features(critic_obs, wm_feature, grid) for m in self.members]

    def q_from_features(self, s_feats, actions):
        return torch.stack([m.q_from_features(f, actions)
                            for m, f in zip(self.members, s_feats)], 0).min(0).values

    def v_from_features(self, s_feats):
        return torch.stack([m.v_from_features(f)
                            for m, f in zip(self.members, s_feats)], 0).min(0).values

    @property
    def delta(self):
        return min(m.delta for m in self.members)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def compute_safety_returns(safety_values, dones, alpha, clamp_max=0.5,
                           bootstrap=None):
    """Backward recursion, byte-for-byte the one in
    `RolloutStorage.compute_safety_returns`:

        R(t) = alpha * min(sv(t), R(t+1) * not_terminal) + (1-alpha) * sv(t)

    clamped at `clamp_max`.  `safety_values` and `dones` are [T, B].

    NOTE on alpha: the shipped `rollout_storage.py` uses 0.7, its own
    docstring says 0.95/0.05, and Table 4 says 0.80.  That is
    viploco_codebase_addendum.md §2's open reconciliation item.  We take it as
    a parameter and record the value used in the Q-hat report, so E1 is
    reproducible whichever way the reconciliation lands.
    """
    T = safety_values.shape[0]
    out = torch.zeros_like(safety_values)
    nxt = (torch.zeros_like(safety_values[0]) if bootstrap is None
           else bootstrap.clamp(max=clamp_max))
    for t in reversed(range(T)):
        sv_t = safety_values[t]
        not_terminal = 1.0 - dones[t].float()
        effective_next = nxt * not_terminal
        out[t] = (alpha * torch.min(sv_t, effective_next)
                  + (1.0 - alpha) * sv_t).clamp(max=clamp_max)
        nxt = out[t]
    return out


def collision_within_horizon(min_cbf_h, dones, horizon, thresh):
    """Label [T, B]: does a collision (`min_cbf_h < thresh`) occur within
    `horizon` steps, without crossing an episode boundary?  The positive class
    for the Q-hat AUROC check.

    Implemented as a backward "steps until the next collision" recursion, with
    the count reset to infinity across a terminal step so labels never leak
    from one episode into the previous one.
    """
    T = min_cbf_h.shape[0]
    hit = (min_cbf_h < thresh)
    INF = float(horizon) + 1.0
    out = torch.zeros_like(hit)
    steps = torch.full_like(min_cbf_h[0], INF)   # steps-to-hit at t+1
    for t in reversed(range(T)):
        nxt = torch.where(dones[t].bool(), torch.full_like(steps, INF), steps)
        steps = torch.where(hit[t], torch.zeros_like(steps),
                            torch.clamp(nxt + 1.0, max=INF))
        out[t] = steps <= float(horizon)
    return out
