"""E5 -- biped fall-safety evaluation loop.

`EvalCollector` (eval_common.py) is corridor-specific: its headline metric is
the link-vs-cylinder collision test, its `safety_value < -0.05` fallback would
misread the fall margin as a collision, and it has no push injection.  This
module keeps everything that is task-agnostic -- `PolicyRuntime` (the
world-model / history bookkeeping, reproduced step-for-step from the training
loop), the chunked seed protocol, the record/manifest writers -- and swaps the
metric core for fall accounting plus directed push perturbations.

Timing convention (mirrors eval_common's, stated in its module docstring):
per-step *state* quantities (V_safe, the fall margin, velocities) are read
from the state the policy acts on, i.e. before `env.step`; the terminal
margin of a fall is additionally folded in from the post-step `infos`
(pre-reset values).  V_safe and the margin are therefore evaluated on the
same state, which is what makes their zero-crossing lead times comparable.

IMPORTANT: `import isaacgym` must happen before torch is imported.  Callers
(run_e5.py) do that; this module only assumes it has already happened at
env-build time.  `PushSchedule`'s timing/direction math is pure torch so the
offline tests can exercise it without isaacgym.
"""

import os

import torch

from . import metrics as M
from . import seeds as S
from . import eval_common as EC


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def apply_biped_eval_overrides(env_cfg, train_cfg, terrain, num_envs,
                               cmd_vx, dr_mode="off"):
    """eval_common.apply_eval_overrides plus the two biped-specific settings.

    * `fall_safety.disable_vel_violate = True`: the base env resets any env
      whose forward velocity lags its command by 1.5 m/s, which would censor
      exactly the episodes a push evaluation is about (pushed hard backward,
      still recoverable).  Falls still terminate through the contact / tilt /
      timeout paths.
    * `dr_mode='off'` already sets `push_robots = False`, so the only pushes
      an eval episode sees are the scheduled, directed ones.
    """
    EC.apply_eval_overrides(env_cfg, train_cfg, terrain, num_envs,
                            dr_mode=dr_mode, cmd_vel_x=cmd_vx)
    if hasattr(env_cfg, "fall_safety"):
        env_cfg.fall_safety.disable_vel_violate = True
    return env_cfg, train_cfg


def build_biped_env_and_policy(args, terrain, num_envs, cmd_vx, dr_mode="off",
                               load_run=None, checkpoint=None):
    """eval_common.build_env_and_policy with the biped override function.

    (That function hardcodes `apply_eval_overrides`, so the ~25 lines are
    reproduced here rather than modifying the shared invariant file.)
    """
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_load_path
    from legged_gym import LEGGED_GYM_ROOT_DIR

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    apply_biped_eval_overrides(env_cfg, train_cfg, terrain, num_envs,
                               cmd_vx, dr_mode)
    args.num_envs = num_envs

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    train_cfg.runner.resume = True
    if load_run is not None:
        train_cfg.runner.load_run = load_run
    if checkpoint is not None:
        train_cfg.runner.checkpoint = checkpoint

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name)
    EC.check_run_exists(log_root, train_cfg.runner.load_run)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                checkpoint=train_cfg.runner.checkpoint)

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    return env, runner, policy, env_cfg, train_cfg, resume_path


# ---------------------------------------------------------------------------
# Directed push schedule
# ---------------------------------------------------------------------------

class PushSchedule(object):
    """Directed base-velocity impulses on a fixed step schedule.

    First push at step `p0`, then every `dp` steps.  Directions cycle through
    +x, -x, +y, -y (world frame) and a random heading, with a per-env phase
    offset so simultaneous pushes point different ways across envs.  The
    timing/direction math is deterministic in (seed, push ordinal, env index)
    and torch-only; only `apply` touches the simulator.

    mode='add' adds the impulse to the current base velocity (a shove -- a
    push against the direction of travel actually removes momentum);
    mode='set' overwrites it like the training-time `_push_robots` does.
    """

    CARDINALS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))

    def __init__(self, magnitude, mode="add", dirs="all",
                 p0=200, dp=150, seed=0):
        if mode not in ("add", "set"):
            raise ValueError("mode must be 'add' or 'set'")
        if dirs not in ("all", "cardinal", "random"):
            raise ValueError("dirs must be 'all', 'cardinal' or 'random'")
        self.magnitude = float(magnitude)
        self.mode = mode
        self.dirs = dirs
        self.p0 = int(p0)
        self.dp = int(dp)
        self.seed = int(seed)

    def is_push_step(self, step_idx):
        if self.magnitude <= 0.0:
            return False
        return step_idx >= self.p0 and (step_idx - self.p0) % self.dp == 0

    def ordinal(self, step_idx):
        return (step_idx - self.p0) // self.dp

    def directions(self, k, num_envs, device="cpu"):
        """[E, 2] unit vectors for push ordinal k."""
        n_card = len(self.CARDINALS)
        if self.dirs == "random":
            choice = torch.full((num_envs,), n_card, device=device,
                                dtype=torch.long)
        else:
            n_opts = n_card + (1 if self.dirs == "all" else 0)
            choice = (torch.arange(num_envs, device=device) + k) % n_opts
        card = torch.tensor(self.CARDINALS, device=device)
        out = torch.zeros(num_envs, 2, device=device)
        is_card = choice < n_card
        out[is_card] = card[choice[is_card].clamp(max=n_card - 1)]
        n_rand = int((~is_card).sum())
        if n_rand > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 1000003 + k)
            theta = torch.rand(num_envs, generator=gen) * 2 * torch.pi
            theta = theta.to(device)
            rand_vec = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
            out[~is_card] = rand_vec[~is_card]
        return out

    def maybe_push(self, env, step_idx):
        """Apply the scheduled impulse, if any.  Returns [E] bool pushed mask."""
        num_envs = env.num_envs
        device = env.device
        if not self.is_push_step(step_idx):
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        from isaacgym import gymtorch
        d = self.directions(self.ordinal(step_idx), num_envs, device)
        dv = self.magnitude * d
        if self.mode == "add":
            env.root_states[:, 7:9] += dv
        else:
            env.root_states[:, 7:9] = dv
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.root_states))
        return torch.ones(num_envs, dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Episode accumulation
# ---------------------------------------------------------------------------

_SUM_FIELDS = ["n_steps", "return", "vel_err_sum", "lat_vel_sum",
               "fwd_vel_sum", "fall_margin_sum", "vsafe_sum", "n_pushes"]
_MIN_FIELDS = ["fall_margin_min", "vsafe_min"]
_STEP_FIELDS = ["margin_first_neg", "vsafe_first_neg", "last_push_step"]


class BipedAccumulator(object):
    """Vectorised per-env accumulators for the E5 record fields."""

    def __init__(self, num_envs, device):
        self.n = num_envs
        self.device = device
        self.reset_all()

    def _fresh(self, fill):
        return torch.full((self.n,), fill, device=self.device)

    def reset_all(self):
        self.acc = {}
        for k in _SUM_FIELDS:
            self.acc[k] = self._fresh(0.0)
        for k in _MIN_FIELDS:
            self.acc[k] = self._fresh(float("inf"))
        for k in _STEP_FIELDS:
            self.acc[k] = self._fresh(-1.0)

    def reset_envs(self, mask):
        for k in _SUM_FIELDS:
            self.acc[k] = torch.where(mask, torch.zeros_like(self.acc[k]),
                                      self.acc[k])
        for k in _MIN_FIELDS:
            self.acc[k] = torch.where(
                mask, torch.full_like(self.acc[k], float("inf")), self.acc[k])
        for k in _STEP_FIELDS:
            self.acc[k] = torch.where(
                mask, torch.full_like(self.acc[k], -1.0), self.acc[k])

    def add(self, key, value, live):
        self.acc[key] = self.acc[key] + value.float() * live

    def take_min(self, key, value, live):
        cand = torch.where(live.bool(), value.float(), self.acc[key])
        self.acc[key] = torch.minimum(self.acc[key], cand)

    def first_step(self, key, cond, step_idx, live):
        """Record step_idx the first time cond holds (per env)."""
        hit = cond & live.bool() & (self.acc[key] < 0)
        self.acc[key] = torch.where(
            hit, torch.full_like(self.acc[key], float(step_idx)), self.acc[key])

    def set_step(self, key, mask, step_idx, live):
        upd = mask & live.bool()
        self.acc[key] = torch.where(
            upd, torch.full_like(self.acc[key], float(step_idx)), self.acc[key])


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class BipedEvalCollector(object):
    """Steps the env under a push schedule and harvests fall records.

    Also logs per-step (V_safe, fall margin) series per chunk for the
    fall-prediction calibration analysis; `save_calibration` writes them as a
    compressed .npz keyed `c{chunk}_{name}`.
    """

    def __init__(self, env, runner, policy, push, run_meta=None):
        self.env = env
        self.runner = runner
        self.policy = policy
        self.push = push
        self.meta = run_meta or {}
        self.rt = EC.PolicyRuntime(env, runner)
        self.acc = BipedAccumulator(env.num_envs, env.device)
        self.records = []
        self.calib = {}          # chunk_idx -> dict of stacked [T, E] arrays
        self.clip_actions = float(env.cfg.normalization.clip_actions)
        self.dt = float(env.dt)
        self.push_window_steps = int(round(M.PUSH_FALL_WINDOW_S / self.dt))

    def _vsafe(self, critic_obs):
        with torch.no_grad():
            v = self.runner.alg.actor_critic.evaluate_safety(
                critic_obs.detach(), self.rt.wm_feature.detach(),
                grid=self.rt.occ_grid)
        return v.squeeze(-1)

    def run_chunk(self, chunk_idx, chunk_seed, keep, max_steps=None):
        from legged_gym.utils.helpers import set_seed
        env = self.env

        set_seed(int(chunk_seed))
        obs, privileged_obs, _ = env.reset()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.rt.reset(obs)
        self.acc.reset_all()

        live = torch.ones(env.num_envs, device=env.device)
        finished = torch.zeros(env.num_envs, dtype=torch.bool,
                               device=env.device)
        max_steps = max_steps or int(env.max_episode_length) + 3
        kept = []
        series = {k: [] for k in
                  ("vsafe", "margin", "live", "pushed", "done")}

        for step_idx in range(max_steps):
            self.rt.maybe_update_world_model()

            # ---- pre-step state metrics (state the policy acts on) -------
            # env.safety_values / env.fall_margin_min were computed in the
            # previous post_physics_step, i.e. they describe the current
            # state -- the same state evaluate_safety sees.
            vsafe = self._vsafe(critic_obs)
            margin = env.fall_margin_min.to(env.device)
            self.acc.add("vsafe_sum", vsafe, live)
            self.acc.take_min("vsafe_min", vsafe, live)
            self.acc.add("fall_margin_sum", margin, live)
            self.acc.take_min("fall_margin_min", margin, live)
            self.acc.first_step("vsafe_first_neg", vsafe < 0, step_idx, live)
            self.acc.first_step("margin_first_neg", margin < 0, step_idx, live)

            v_err = (env.base_lin_vel[:, 0] - env.commands[:, 0]).abs()
            self.acc.add("vel_err_sum", v_err, live)
            self.acc.add("lat_vel_sum", env.base_lin_vel[:, 1].abs(), live)
            self.acc.add("fwd_vel_sum", env.base_lin_vel[:, 0], live)

            # ---- policy ---------------------------------------------------
            with torch.no_grad():
                a_pol = self.policy(obs.detach(), self.rt.history().detach(),
                                    self.rt.wm_feature.detach(),
                                    grid=self.rt.occ_grid)
                a_pol = a_pol.clamp(-self.clip_actions, self.clip_actions)

            # ---- scheduled push, integrated by the upcoming sim step -----
            pushed = self.push.maybe_push(env, step_idx)
            self.acc.add("n_pushes", pushed, live)
            self.acc.set_step("last_push_step", pushed, step_idx, live)

            series["vsafe"].append(vsafe.detach().cpu())
            series["margin"].append(margin.detach().cpu())
            series["live"].append(live.detach().cpu())
            series["pushed"].append(pushed.detach().cpu())

            # ---- step -----------------------------------------------------
            obs, privileged_obs, rews, dones, infos, reset_ids, _ = env.step(
                a_pol.detach())
            critic_obs = privileged_obs if privileged_obs is not None else obs

            # ---- post-step transition metrics (pre-reset values) ---------
            self.acc.add("n_steps", torch.ones_like(live), live)
            self.acc.add("return", rews.to(env.device), live)
            # The done step's post-step margin is the terminal state's --
            # fold it into the min so a fall's depth is captured even though
            # the pre-step read never sees the post-impact pose.
            term_margin = infos["fall_margin_min"].to(env.device)
            self.acc.take_min("fall_margin_min", term_margin,
                              live * dones.float())
            series["done"].append(dones.detach().bool().cpu())

            self.rt.post_step(obs, infos, dones, reset_ids)

            just_done = dones.bool() & (~finished)
            if just_done.any():
                kept.extend(self._harvest(just_done, chunk_idx, chunk_seed))
                finished |= just_done
                live = (~finished).float()
            if bool(finished.all()):
                break

        self._store_series(chunk_idx, series)

        kept.sort(key=lambda r: r["env_idx"])
        if keep < len(kept):
            stride = len(kept) / float(keep)
            kept = [kept[int(i * stride)] for i in range(keep)]
        self.records.extend(kept)
        return len(kept)

    def _store_series(self, chunk_idx, series):
        stacked = {}
        for k, rows in series.items():
            if rows:
                stacked[k] = torch.stack(rows, dim=0).numpy()
        self.calib[chunk_idx] = stacked

    def _harvest(self, mask, chunk_idx, chunk_seed):
        env = self.env
        idx = mask.nonzero(as_tuple=False).flatten()
        a = {k: v[idx].cpu().tolist() for k, v in self.acc.acc.items()}
        term = {
            "timeout": env.time_out_buf[idx].cpu().tolist(),
            "fall": env.fall[idx].cpu().tolist(),
            "vel_violate": env.vel_violate[idx].cpu().tolist(),
            "contact": EC._get(env, "term_contact", idx),
            "proj_grav_z": EC._get(env, "term_proj_grav_z", idx, default=-1.0),
            "base_z_rel": EC._get(env, "term_base_z_rel", idx, default=1.0),
            "base_vel_z": EC._get(env, "term_base_vel_z", idx, default=0.0),
        }
        levels = env.terrain_levels[idx].cpu().tolist()
        out = []
        for j, env_i in enumerate(idx.cpu().tolist()):
            n = max(1.0, a["n_steps"][j])
            rec = M.blank_record()
            rec.update(self.meta)
            rec.update({
                "episode_idx": -1,
                "chunk_idx": chunk_idx, "seed": int(chunk_seed),
                "env_idx": env_i, "terrain_level": int(levels[j]),
                "n_steps": int(a["n_steps"][j]),
                "return": a["return"][j],
                "vel_err": a["vel_err_sum"][j] / n,
                "mean_lat_vel": a["lat_vel_sum"][j] / n,
                "mean_fwd_vel": a["fwd_vel_sum"][j] / n,
                "n_pushes": int(a["n_pushes"][j]),
                "mean_fall_margin": a["fall_margin_sum"][j] / n,
                "min_fall_margin": a["fall_margin_min"][j],
                "mean_vsafe": a["vsafe_sum"][j] / n,
                "min_vsafe": a["vsafe_min"][j],
                "vsafe_first_neg_step": int(a["vsafe_first_neg"][j]),
                "margin_first_neg_step": int(a["margin_first_neg"][j]),
                "term_timeout": int(bool(term["timeout"][j])),
                "term_fall_flag": int(bool(term["fall"][j])),
                "term_vel_violate": int(bool(term["vel_violate"][j])),
                "term_contact": int(bool(term["contact"][j])),
                "term_proj_grav_z": term["proj_grav_z"][j],
                "term_base_z_rel": term["base_z_rel"][j],
                "term_base_vel_z": term["base_vel_z"][j],
            })
            rec["fell"] = int(M.is_fall(rec))
            steps = int(a["n_steps"][j])
            rec["steps_to_fall"] = steps if rec["fell"] else 0
            last_push = a["last_push_step"][j]
            rec["fell_within_push_window"] = int(
                rec["fell"] and last_push >= 0
                and (steps - 1 - last_push) <= self.push_window_steps)
            out.append(rec)
        self.acc.reset_envs(mask)
        return out

    # -- driver ----------------------------------------------------------
    def run(self, n_episodes, seed_list, verbose=True):
        import time as _time
        plan = S.chunk_plan(seed_list, self.env.num_envs, n_episodes)
        t0 = _time.time()
        for chunk_idx, chunk_seed, keep in plan:
            got = self.run_chunk(chunk_idx, chunk_seed, keep)
            if verbose:
                print("[e5] chunk {}/{} seed={} kept={} total={} "
                      "({:.0f}s elapsed)".format(
                          chunk_idx + 1, len(plan), chunk_seed, got,
                          len(self.records), _time.time() - t0))
        for i, r in enumerate(self.records):
            r["episode_idx"] = i
        return self.records

    def save_calibration(self, path):
        """Write the per-step series as a compressed npz.

        Keys: c{chunk}_vsafe, c{chunk}_margin  [T, E] float32
              c{chunk}_live, c{chunk}_pushed, c{chunk}_done  [T, E] bool
        The analyzer derives fall labels from `done` at steps where `live`
        was still set (each env's first episode only, matching the records).
        """
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        blob = {}
        for ci, arrs in self.calib.items():
            for k, v in arrs.items():
                blob["c{}_{}".format(ci, k)] = v
        np.savez_compressed(path, **blob)
        return path
