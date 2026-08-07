"""The shared evaluation loop, termination definition and metric computation.

experiment_protocol_phased.md §1.1 makes this file a cross-phase invariant:

    Shared eval/metrics module: factor the evaluation loop, termination
    definition, and metric computations into one file and copy it verbatim
    across branches.  Divergent metric code is the classic silent killer here.

So: **copy this file (and metrics.py, stats.py, filters.py, seeds.py,
eval_seeds.json) verbatim to the joint-limit branch.**  J1 supplies a different
`ActionFilter`; nothing else changes.

The loop itself is `legged_gym/scripts/play.py` refactored, with the world
model / history / occupancy-grid bookkeeping reproduced step-for-step so the
unfiltered row is bit-identical to the deployment path.

IMPORTANT: `import isaacgym` must happen before torch is imported.  Callers
(run_e1.py) do that; this module only assumes it has already happened.

--------------------------------------------------------------------------
Timing convention, stated once and applied everywhere
--------------------------------------------------------------------------
Per-step *state* quantities (joint-limit margins, base velocities, yaw rate,
contact phase, command jerk) are read from the state the policy acted on, i.e.
immediately **before** `env.step`.  Per-transition quantities (reward,
`min_cbf_h`) are read immediately **after** `env.step`, from values the env
computes before `reset_idx` runs.  No post-reset value ever enters a metric.
Terminal-state diagnostics for the fall definition are snapshotted inside
`check_termination`, also before `reset_idx`.

Post-fall timesteps cannot exist here because an episode ends at its fall, so
the §1 rule about excluding them from rate denominators is satisfied by
construction; state it in table captions anyway.
"""

import hashlib
import json
import os
import subprocess
import time

import torch

from . import metrics as M
from . import seeds as S

# Frozen eval-time settings (§1.1).  Changing any of these invalidates
# comparisons against the paper's existing rows -- re-run the reference rows
# instead of comparing across configs.
TERRAIN_SEED = 1                 # seeds terrain generation at env creation
EVAL_CMD_LIN_VEL_X = 0.6         # fixed forward command, as in play.py
EVAL_TERRAIN_NUM_COLS = 1
CONTACT_FORCE_THRESH = 1.0       # foot considered in contact above this (N)

TERRAIN_PROPORTIONS = {
    # 'flat' is used by the command-response calibration, which needs the
    # command interface measured without terrain or obstacles confounding it.
    #
    # Proportion index 9 is the flattest thing this generator makes:
    # random_uniform_terrain(+-0.05 m) and nothing else -- the same base the
    # corridor is built on, minus the obstacles.  Index 0 is NOT flat, it is
    # wave_terrain(num_waves=5); putting the calibration there had the robot
    # crawling at 0.05 m/s against a 0.6 m/s command, which makes every
    # tracking gain meaningless.
    #
    # Index 9 also puts roughflat_start_idx at 0, so the heading controller
    # covers no env -- the same as the corridor, which is what the
    # calibration is supposed to be informing.  Index 0 covered every env and
    # overwrote the yaw command outright.
    'flat':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
    'wave':     [1.0, 0, 0, 0, 0, 0, 0, 0, 0],
    'slope':    [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
    'stair':    [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
    'gap':      [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
    'climb':    [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
    'tilt':     [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
    'crawl':    [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
    'corridor': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
}


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def apply_eval_overrides(env_cfg, train_cfg, terrain, num_envs, dr_mode="off",
                         cmd_vel_x=EVAL_CMD_LIN_VEL_X):
    """The one place eval-time config is decided (§1.1 invariant #4).

    `dr_mode='off'` neutralises domain randomisation by collapsing its
    *ranges*.  It deliberately does **not** flip the `randomize_*` booleans
    that gate blocks of `privileged_obs_buf`: those change the observation
    width and would silently break the checkpoint.  Only the flags that add no
    observation (link mass, motor strength, action latency, pushes) are turned
    off outright.
    """
    if terrain not in TERRAIN_PROPORTIONS:
        raise ValueError("terrain must be one of {}".format(
            sorted(TERRAIN_PROPORTIONS)))

    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_cols = EVAL_TERRAIN_NUM_COLS
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_proportions = TERRAIN_PROPORTIONS[terrain]
    env_cfg.noise.add_noise = False
    env_cfg.depth.use_camera = True

    if env_cfg.terrain.mesh_type == 'plane':
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0

    for attr, val in (("lin_vel_x", [cmd_vel_x, cmd_vel_x]),
                      ("lin_vel_y", [0.0, 0.0]),
                      ("ang_vel_yaw", [0.0, 0.0]),
                      ("heading", [0.0, 0.0]),
                      ("flat_lin_vel_x", [cmd_vel_x, cmd_vel_x]),
                      ("flat_lin_vel_y", [0.0, 0.0]),
                      ("flat_ang_vel_yaw", [0.0, 0.0])):
        if hasattr(env_cfg.commands.ranges, attr):
            setattr(env_cfg.commands.ranges, attr, val)

    if dr_mode == "off":
        dr = env_cfg.domain_rand
        dr.friction_range = [1.0, 1.0]
        dr.restitution_range = [0.0, 0.0]
        dr.added_mass_range = [0.0, 0.0]
        dr.com_x_pos_range = [0.0, 0.0]
        dr.com_y_pos_range = [0.0, 0.0]
        dr.com_z_pos_range = [0.0, 0.0]
        dr.link_mass_range = [1.0, 1.0]
        dr.stiffness_multiplier_range = [1.0, 1.0]
        dr.damping_multiplier_range = [1.0, 1.0]
        dr.motor_strength_range = [1.0, 1.0]
        # These add no privileged-observation block, so they can be disabled.
        dr.randomize_link_mass = False
        dr.randomize_motor_strength = False
        dr.randomize_action_latency = False
        dr.push_robots = False
    elif dr_mode != "on":
        raise ValueError("dr_mode must be 'off' or 'on'")

    train_cfg.runner.amp_num_preload_transitions = 1
    env_cfg.seed = TERRAIN_SEED
    train_cfg.seed = TERRAIN_SEED
    return env_cfg, train_cfg


def heading_control_coverage(env):
    """How many envs have their yaw-rate command overwritten by the env.

    With `commands.heading_command = True`, `_post_physics_step_callback`
    recomputes `commands[:, 2]` every step from the heading error, for envs
    `[:roughflat_start_idx]`.  Anything written to that column for those envs
    is discarded, silently.

    The slice bound is `ceil(num_envs * sum(terrain_proportions[:9]))`, so it
    depends on the terrain:

        flat      proportions[9]  = 1  ->  bound 0, nobody is covered
        corridor  proportions[10] = 1  ->  bound 0, nobody is covered
        wave      proportions[0]  = 1  ->  bound N, everybody is covered

    Calibration and sweep therefore see the same command interface, which is
    the point.  The calibration originally ran on `wave` by mistake -- it was
    the entry named 'flat' -- where the heading controller overwrote the yaw
    command outright and the measured tracking gain was zero regardless of
    what the policy could do.  Returns (n_covered, n_envs).
    """
    if not bool(getattr(env.cfg.commands, "heading_command", False)):
        return 0, int(env.num_envs)
    return int(getattr(env, "roughflat_start_idx", 0)), int(env.num_envs)


def disable_heading_command(env):
    """Hand the yaw-rate command back to the caller.

    Open-loop yaw calibration needs the commanded yaw rate to actually reach
    the policy; with heading mode on it never does.  Returns the previous
    setting so a caller can record what it changed.
    """
    prev = bool(getattr(env.cfg.commands, "heading_command", False))
    env.cfg.commands.heading_command = False
    return prev


def check_run_exists(log_root, load_run):
    """Fail early, and say what *is* there.

    `get_load_path` resolves a missing run into `os.listdir` on a path that
    does not exist, so a mistyped or absent checkpoint surfaces as a bare
    FileNotFoundError from inside legged_gym with no indication of which run
    was wanted or what the alternatives are.  Every one of these scripts
    takes a `--load_run`, so that is a cheap thing to get right.
    """
    if load_run in (None, -1, "-1"):
        return
    path = os.path.join(log_root, str(load_run))
    if os.path.isdir(path):
        return
    try:
        available = sorted(d for d in os.listdir(log_root)
                           if os.path.isdir(os.path.join(log_root, d)))
    except OSError:
        available = []
    raise SystemExit(
        "no such run: {}\n"
        "  looked in : {}\n"
        "  available : {}\n"
        "Pass --load_run with one of the above, or point the sweep at it "
        "(e.g. PI_OURS_RUN=... for the reference sweep).".format(
            load_run, log_root,
            ", ".join(available) if available else "(none)"))


def build_env_and_policy(args, terrain, num_envs, dr_mode="off",
                         load_run=None, checkpoint=None):
    """Create the eval env and load the pinned checkpoint.

    Returns `(env, runner, policy, env_cfg, train_cfg, resume_path)`.
    """
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_load_path
    from legged_gym import LEGGED_GYM_ROOT_DIR

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    apply_eval_overrides(env_cfg, train_cfg, terrain, num_envs, dr_mode)
    # `update_cfg_from_args` re-applies args.num_envs inside make_env; keep the
    # two in sync so the chunk plan matches the env that is actually built.
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
    check_run_exists(log_root, train_cfg.runner.load_run)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                checkpoint=train_cfg.runner.checkpoint)

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    return env, runner, policy, env_cfg, train_cfg, resume_path


# ---------------------------------------------------------------------------
# World-model / history bookkeeping, mirroring play.py exactly
# ---------------------------------------------------------------------------

class PolicyRuntime(object):
    """Owns the trajectory history, world-model state and occupancy grid.

    Reset per chunk so every method starts each chunk from the same state.
    """

    def __init__(self, env, runner, history_length=5):
        self.env = env
        self.runner = runner
        self.world_model = runner._world_model.to(env.device)
        self.wm_feature_dim = runner.wm_feature_dim
        self.history_length = history_length
        self.wm_update_interval = env.cfg.depth.update_interval
        occ_cfg = getattr(env.cfg, 'occupancy_grid', None)
        self.grid_enabled = occ_cfg is not None and occ_cfg.enabled
        self.grid_shape = ((occ_cfg.grid_H, occ_cfg.grid_W)
                           if self.grid_enabled else None)
        self._history_reads = 0
        self._warned_post_step = False

    def reset(self, obs):
        env = self.env
        n = env.num_envs
        self._history_reads = 0
        self._warned_post_step = False
        self.trajectory_history = torch.zeros(
            (n, self.history_length,
             env.num_obs - env.privileged_dim - env.height_dim - 3),
            device=env.device)
        self._push_history(obs)

        self.wm_is_first = torch.ones(n, device=self.world_model.device)
        self.wm_obs = {
            "prop": obs[:, env.privileged_dim:
                        env.privileged_dim + env.cfg.env.prop_dim
                        ].to(self.world_model.device),
            "is_first": self.wm_is_first,
            "wm_state": None,
        }
        if env.cfg.depth.use_camera:
            self.wm_obs["image"] = torch.zeros(
                ((n,) + env.cfg.depth.resized + (1,)),
                device=self.world_model.device)
        self.wm_feature = torch.zeros((n, self.wm_feature_dim),
                                      device=env.device)
        self.occ_grid = (torch.zeros(n, *self.grid_shape, device=env.device)
                         if self.grid_enabled else None)

    def _push_history(self, obs):
        env = self.env
        obs_wo_cmd = torch.cat(
            (obs[:, env.privileged_dim:env.privileged_dim + 6],
             obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
        self.trajectory_history = torch.cat(
            (self.trajectory_history[:, 1:], obs_wo_cmd.unsqueeze(1)), dim=1)

    def maybe_update_world_model(self):
        """Runs at the top of the loop, exactly as in play.py.

        Wrapped in `no_grad` -- unlike play.py, which leaves it ungated. The
        RSSM state is carried across steps, so a tracked graph would chain
        every world-model update in the episode together and grow without
        bound. `no_grad` (not `inference_mode`) because variant B still needs
        autograd through Q_hat w.r.t. the action.
        """
        if self.env.global_counter % self.wm_update_interval != 0:
            return
        with torch.no_grad():
            preds, hidden, state = self.world_model(self.wm_obs)
            self.wm_feature = torch.cat([hidden, preds], dim=-1).to(
                self.world_model.device)
            self.wm_obs["wm_state"] = state
            self.wm_is_first[:] = 0

    def history(self):
        # A loop that drives the policy but never calls post_step gets a
        # frozen history, a world model re-encoding the same proprioception,
        # and an all-zero depth image and occupancy grid.  The policy then
        # produces actions that do not walk, which reads as a broken
        # checkpoint or an untrackable command rather than as a broken loop.
        # It cost a calibration run to find, so say so instead.
        self._history_reads += 1
        if self._history_reads > 10 and not self._warned_post_step:
            self._warned_post_step = True
            print("[PolicyRuntime] WARNING: history() has been read {} times "
                  "with no intervening post_step(). The trajectory history "
                  "and world-model inputs are frozen at reset, so the policy "
                  "is acting on a stale snapshot. Call post_step(obs, infos, "
                  "dones, reset_ids) after every env.step().".format(
                      self._history_reads))
        return self.trajectory_history.flatten(1).to(self.env.device)

    def post_step(self, obs, infos, dones, reset_env_ids):
        env = self.env
        self._history_reads = 0
        if self.grid_enabled:
            new_grid = infos.get("occupancy_grid", None)
            if new_grid is not None:
                self.occ_grid = new_grid.to(env.device)

        self.wm_obs["prop"] = obs[
            :, env.privileged_dim:env.privileged_dim + env.cfg.env.prop_dim
        ].to(self.world_model.device)
        self.wm_obs["is_first"] = self.wm_is_first

        # "because after step": global_counter already advanced, so the image
        # captured here is the one the next world-model update consumes.
        if env.global_counter % self.wm_update_interval == 0:
            if env.cfg.depth.use_camera and infos.get("depth") is not None:
                self.wm_obs["image"] = infos["depth"].unsqueeze(-1).to(
                    self.world_model.device)

        ids = reset_env_ids.cpu().numpy() if torch.is_tensor(reset_env_ids) \
            else reset_env_ids
        if len(ids) > 0:
            self.wm_is_first[ids] = 1

        self.trajectory_history[dones.nonzero(as_tuple=False).flatten()] = 0
        self._push_history(obs)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def link_obstacle_collision(body_pos, obstacle_pos, obstacle_radii,
                            obstacle_mask, has_obstacles,
                            cylinder_height=M.CYLINDER_HEIGHT,
                            tol=M.BODY_COLLISION_TOL):
    """[E] bool -- does any robot link intersect any obstacle cylinder?

    Transcribed from `legged_gym/scripts/play_plan.py`, which is how the
    paper's own collision numbers were computed.  Keeping it identical is what
    makes E1's rows comparable to Table 2 rather than merely internally
    consistent.

    A link collides when it is within `r_obs + tol` of the obstacle axis in XY
    *and* sits at or below the cylinder top (`obstacle_z + cylinder_height`)
    within the same tolerance.  Every rigid body counts, so this is real
    leg-environment contact -- not the base-only margin `h`.

    Shapes: body_pos [E, B, 3], obstacle_pos [E, O, 3], obstacle_radii [E],
    obstacle_mask [E, O], has_obstacles [E].
    """
    body_xy = body_pos[:, :, :2].unsqueeze(2)            # [E, B, 1, 2]
    obs_xy = obstacle_pos[:, :, :2].unsqueeze(1)         # [E, 1, O, 2]
    xy_dist = torch.norm(body_xy - obs_xy, dim=-1)       # [E, B, O]

    obs_top_z = obstacle_pos[:, :, 2] + cylinder_height  # [E, O]
    z_ok = body_pos[:, :, 2].unsqueeze(2) <= (obs_top_z.unsqueeze(1) + tol)

    touching = (
        (xy_dist < obstacle_radii.unsqueeze(1).unsqueeze(2) + tol)
        & z_ok
        & obstacle_mask.unsqueeze(1)
        & has_obstacles.unsqueeze(1).unsqueeze(2)
    )
    return touching.any(dim=1).any(dim=1)


def env_has_collision_geometry(env):
    """play_plan.py's own precondition check, kept verbatim in spirit."""
    return all(hasattr(env, a) for a in
               ("obstacle_positions", "obstacle_radii", "has_obstacles",
                "obstacle_mask", "rigid_body_pos"))


# ---------------------------------------------------------------------------
# Per-episode accumulation
# ---------------------------------------------------------------------------

_ACC_FIELDS = [
    "n_steps", "n_collision_steps", "n_proximity_steps",
    "n_viol_steps", "return", "vel_err_sum",
    "lat_vel_sum", "fwd_vel_sum", "h_sum", "h_min", "activation_steps", "target_miss_steps",
    "sum_dnorm_v", "sum_dnorm_omega",
    "sum_yaw_cmd_abs", "sum_yaw_realised_abs", "sum_yaw_signed",
    "yaw_cmd_steps",
    "trigger_steps", "sum_dnorm_inf", "max_dnorm_inf", "sum_dnorm_l2",
    "q_gap_sum", "q_gap_n", "peak_yaw", "peak_jerk", "handoffs",
    "steps_flight", "steps_partial", "steps_stance",
    "dnorm_flight", "dnorm_partial", "dnorm_stance",
    "infeasible_steps",
]


class EpisodeAccumulator(object):
    """Vectorised per-env accumulators, harvested into records on `done`."""

    def __init__(self, num_envs, device):
        self.device = device
        self.n = num_envs
        self.reset_all()

    def reset_all(self):
        z = torch.zeros(self.n, device=self.device)
        self.acc = {k: z.clone() for k in _ACC_FIELDS}
        self.acc["h_min"] = torch.full((self.n,), float("inf"),
                                       device=self.device)

    def reset_envs(self, mask):
        for k in _ACC_FIELDS:
            fill = float("inf") if k == "h_min" else 0.0
            self.acc[k] = torch.where(mask, torch.full_like(self.acc[k], fill),
                                      self.acc[k])

    def add(self, key, value, live):
        self.acc[key] = self.acc[key] + value.float() * live

    def add_max(self, key, value, live):
        cand = torch.where(live.bool(), value.float(), self.acc[key])
        self.acc[key] = torch.maximum(self.acc[key], cand)

    def add_min(self, key, value, live):
        cand = torch.where(live.bool(), value.float(), self.acc[key])
        self.acc[key] = torch.minimum(self.acc[key], cand)


class EvalCollector(object):
    """Steps the env under a filter and harvests per-episode records."""

    def __init__(self, env, runner, policy, filt, qsafe=None,
                 trigger_source="vhat", run_meta=None, cmd_filt=None):
        self.env = env
        self.runner = runner
        self.policy = policy
        self.filt = filt
        # E4: optional command-space filter, applied before the policy runs.
        # None for every other experiment, which leaves this path inert.
        self.cmd_filt = cmd_filt
        self.qsafe = qsafe
        self.trigger_source = trigger_source
        self.meta = run_meta or {}
        self.rt = PolicyRuntime(env, runner)
        self.acc = EpisodeAccumulator(env.num_envs, env.device)
        self.records = []

        self.clip_actions = float(env.cfg.normalization.clip_actions)
        self.action_scale = float(env.cfg.control.action_scale)
        self.dt = float(env.dt)

        self.collision_geometry = env_has_collision_geometry(env)
        if not self.collision_geometry:
            print("[eval] WARNING: env is missing obstacle_positions / "
                  "obstacle_radii / has_obstacles / obstacle_mask / "
                  "rigid_body_pos. Falling back to `safety_value < -0.05` as a "
                  "collision proxy -- these numbers are NOT comparable with "
                  "the paper's. Recorded as collision_metric in the manifest.")

        # Hard joint limits, recovered from the softened ones the env stores.
        soft = float(getattr(env.cfg.rewards, "soft_dof_pos_limit", 1.0))
        lo, hi = M.hard_dof_limits(env.dof_pos_limits[:, 0],
                                   env.dof_pos_limits[:, 1], soft)
        self.dof_lo, self.dof_hi = lo, hi

    # -- per-step helpers ------------------------------------------------
    def _joint_violation(self):
        q = self.env.dof_pos
        near = ((q - self.dof_lo.unsqueeze(0)) < M.JOINT_VIOL_MARGIN) | \
               ((self.dof_hi.unsqueeze(0) - q) < M.JOINT_VIOL_MARGIN)
        return near.any(dim=1)

    def _collision(self):
        """Geometric link-vs-cylinder contact, play_plan.py's definition."""
        if not self.collision_geometry:
            # play_plan.py's fallback. Recorded in the manifest as
            # `collision_metric` so a CSV produced this way can never be
            # mistaken for the geometric one.
            return self.env.safety_values.to(self.env.device) < \
                M.COLLISION_H_THRESH
        env = self.env
        return link_obstacle_collision(
            env.rigid_body_pos, env.obstacle_positions, env.obstacle_radii,
            env.obstacle_mask, env.has_obstacles)

    def _contact_bucket(self):
        f = self.env.contact_forces[:, self.env.feet_indices, 2]
        n_contact = (f.abs() > CONTACT_FORCE_THRESH).sum(dim=1)
        return (n_contact == 0), ((n_contact >= 1) & (n_contact <= 2)), \
            (n_contact >= 3)

    def _trigger_value(self, s_feat, critic_obs):
        """V_hat(o) for the trigger.

        Default `vhat`: the state-only head trained alongside Q_hat.  The
        checkpoint's own `safety_critic` is **not** usable for pi_nom -- with
        `safety_coef = 0`, `AMPPPO.update` skips the whole safety branch, so
        that head never received a gradient.  `qpol` (= Q_hat at the policy
        action) and `sv_oracle` (the env's privileged margin) exist as
        controls.
        """
        if self.trigger_source == "vhat":
            return self.qsafe.v_from_features(s_feat)
        if self.trigger_source == "qpol":
            return self.qsafe.q_from_features(s_feat, self._last_a_pol)
        if self.trigger_source == "sv_oracle":
            return self.env.safety_values.to(critic_obs.device)
        raise ValueError("unknown trigger source " + str(self.trigger_source))

    # -- main loop -------------------------------------------------------
    def run_chunk(self, chunk_idx, chunk_seed, keep, max_steps=None):
        from legged_gym.utils.helpers import set_seed
        env = self.env

        set_seed(int(chunk_seed))
        obs, privileged_obs, _ = env.reset()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.rt.reset(obs)
        self.acc.reset_all()

        # The nominal task command, captured before any filtering.  A command
        # filter overwrites env.commands so the policy tracks the filtered
        # value; without this snapshot the next step would filter its own
        # previous output, ratcheting the command down monotonically instead
        # of re-deciding it from the task command each step.  The eval config
        # pins the command ranges to constants, so one snapshot per chunk is
        # exact -- asserted rather than assumed.
        self._nominal_cmd = None
        if self.cmd_filt is not None:
            self._nominal_cmd = env.commands.clone()
            rng = env.cfg.commands.ranges
            for attr in ("lin_vel_x", "lin_vel_y"):
                lo, hi = getattr(rng, attr, [0.0, 0.0])
                if abs(hi - lo) > 1e-9:
                    raise RuntimeError(
                        "command filter assumes a pinned eval command, but "
                        "{} is [{}, {}]; snapshotting the nominal once per "
                        "chunk would be wrong.".format(attr, lo, hi))

        live = torch.ones(env.num_envs, device=env.device)   # first episode?
        finished = torch.zeros(env.num_envs, dtype=torch.bool,
                               device=env.device)
        q_cmd_prev = q_cmd_prev2 = None
        max_steps = max_steps or int(env.max_episode_length) + 3
        kept = []

        for _ in range(max_steps):
            self.rt.maybe_update_world_model()

            # ---- pre-step state metrics (state the policy acts on) -------
            self.acc.add("n_viol_steps", self._joint_violation(), live)
            # Referenced to the *nominal* task command.  Measuring against
            # env.commands would let a command filter drive its own error to
            # zero by lowering the command it is scored against.
            cmd_ref = (self._nominal_cmd[:, 0] if self._nominal_cmd is not None
                       else env.commands[:, 0])
            v_err = (env.base_lin_vel[:, 0] - cmd_ref).abs()
            self.acc.add("vel_err_sum", v_err, live)
            self.acc.add("lat_vel_sum", env.base_lin_vel[:, 1].abs(), live)
            # Actual forward speed: immune to the command being rewritten, so
            # this is what exposes a filter that simply stops the robot.
            self.acc.add("fwd_vel_sum", env.base_lin_vel[:, 0], live)
            self.acc.add_max("peak_yaw", env.base_ang_vel[:, 2].abs(), live)
            flight, partial, stance = self._contact_bucket()

            # ---- command-space filter (E4) -------------------------------
            # Runs *before* the policy, because the command is part of the
            # observation the policy tracks.  Writing env.commands and
            # refreshing the observation is what a navigation-layer filter
            # does in deployment; the refresh is deterministic here because
            # apply_eval_overrides turns observation noise off, so re-running
            # compute_observations cannot resample anything.
            if self.cmd_filt is not None:
                cmd_new, cinfo = self.cmd_filt.apply(self._nominal_cmd, env)
                env.commands[:, :3] = cmd_new[:, :3]
                env.compute_observations()
                obs = env.obs_buf
                privileged_obs = env.privileged_obs_buf
                critic_obs = privileged_obs if privileged_obs is not None else obs
                self.acc.add("trigger_steps", cinfo["triggered"], live)
                self.acc.add("activation_steps", cinfo["activated"], live)
                self.acc.add("sum_dnorm_inf", cinfo["dnorm"], live)
                self.acc.add_max("max_dnorm_inf", cinfo["dnorm"], live)
                self.acc.add("infeasible_steps", cinfo["infeasible"], live)
                if "dnorm_v" in cinfo:
                    self.acc.add("sum_dnorm_v", cinfo["dnorm_v"], live)
                    self.acc.add("sum_dnorm_omega", cinfo["dnorm_omega"], live)
                    # Was the yaw the filter asked for actually delivered?
                    # Accumulated only over steps where yaw was commanded, so
                    # the ratio is a tracking gain and not diluted by the
                    # steps where the filter asked for nothing.
                    yaw_cmd = cmd_new[:, 2]
                    asked = yaw_cmd.abs() > 1e-3
                    self.acc.add("yaw_cmd_steps", asked, live)
                    self.acc.add("sum_yaw_cmd_abs", yaw_cmd.abs() * asked, live)
                    self.acc.add("sum_yaw_realised_abs",
                                 env.base_ang_vel[:, 2].abs() * asked, live)
                    # The one that answers the question.  |realised| cannot
                    # tell turning from gait wobble, and the wobble is an
                    # order of magnitude larger: peak yaw rate is 1.4-2.0
                    # rad/s in every row including unfiltered, against a
                    # commanded 0.1-0.8.  Projecting the realised yaw onto
                    # the commanded direction averages to ~0 under symmetric
                    # wobble and to |cmd| under real tracking.
                    self.acc.add("sum_yaw_signed",
                                 torch.sign(yaw_cmd) * env.base_ang_vel[:, 2]
                                 * asked, live)

            # ---- policy, then filter -------------------------------------
            with torch.no_grad():
                a_pol = self.policy(obs.detach(), self.rt.history().detach(),
                                    self.rt.wm_feature.detach(),
                                    grid=self.rt.occ_grid)
                a_pol = a_pol.clamp(-self.clip_actions, self.clip_actions)
                self._last_a_pol = a_pol
                if self.qsafe is not None:
                    # The world model may live on a different device
                    # (--wm_device); Q_hat lives with the env.
                    s_feat = self.qsafe.state_features(
                        critic_obs.detach(),
                        self.rt.wm_feature.detach().to(env.device),
                        self.rt.occ_grid)
                    trig_val = self._trigger_value(s_feat, critic_obs)
                else:
                    s_feat, trig_val = None, None

            a_exec, finfo = self.filt.apply(a_pol, s_feat, trig_val)

            # ---- filter diagnostics --------------------------------------
            self.acc.add("trigger_steps", finfo["triggered"], live)
            self.acc.add("activation_steps", finfo["activated"], live)
            self.acc.add("target_miss_steps",
                         finfo["target_miss"] & finfo["activated"], live)
            self.acc.add("sum_dnorm_inf", finfo["dnorm_inf"], live)
            self.acc.add("sum_dnorm_l2", finfo["dnorm_l2"], live)
            self.acc.add_max("max_dnorm_inf", finfo["dnorm_inf"], live)
            self.acc.add("q_gap_sum", finfo["q_gap"], live)
            self.acc.add("q_gap_n", finfo["q_gap_valid"], live)
            self.acc.add("steps_flight", flight, live)
            self.acc.add("steps_partial", partial, live)
            self.acc.add("steps_stance", stance, live)
            self.acc.add("dnorm_flight", finfo["dnorm_inf"] * flight, live)
            self.acc.add("dnorm_partial", finfo["dnorm_inf"] * partial, live)
            self.acc.add("dnorm_stance", finfo["dnorm_inf"] * stance, live)

            # ---- command jerk from the executed action -------------------
            q_cmd = env.default_dof_pos + self.action_scale * a_exec
            if q_cmd_prev2 is not None:
                jerk = ((q_cmd - 2 * q_cmd_prev + q_cmd_prev2).abs().amax(-1)
                        / (self.dt ** 2))
                self.acc.add_max("peak_jerk", jerk, live)
            q_cmd_prev2, q_cmd_prev = q_cmd_prev, q_cmd

            # ---- step -----------------------------------------------------
            obs, privileged_obs, rews, dones, infos, reset_ids, _ = env.step(
                a_exec.detach())
            critic_obs = privileged_obs if privileged_obs is not None else obs

            # ---- post-step transition metrics (pre-reset values) ---------
            # `rigid_body_pos` is a view on the sim's rigid-body tensor, which
            # reset_idx does not rewrite (only root_states and dof_state are),
            # so like `cbf_h_min` it still holds the pre-reset pose here.
            h = infos["cbf_h_min"].to(env.device)
            self.acc.add("n_steps", torch.ones_like(live), live)
            self.acc.add("n_collision_steps", self._collision(), live)
            self.acc.add("n_proximity_steps", h < M.COLLISION_H_THRESH, live)
            self.acc.add("return", rews.to(env.device), live)
            self.acc.add("h_sum", h, live)
            self.acc.add_min("h_min", h, live)

            self.rt.post_step(obs, infos, dones, reset_ids)

            # ---- harvest --------------------------------------------------
            just_done = dones.bool() & (~finished)
            if just_done.any():
                kept.extend(self._harvest(just_done, chunk_idx, chunk_seed))
                finished |= just_done
                live = (~finished).float()
            if bool(finished.all()):
                break

        # Deterministic truncation.  Terrain level is `env_idx % num_rows`, so
        # a prefix would bias the last chunk toward easy levels; take an evenly
        # strided subset instead.
        kept.sort(key=lambda r: r["env_idx"])
        if keep < len(kept):
            stride = len(kept) / float(keep)
            kept = [kept[int(i * stride)] for i in range(keep)]
        self.records.extend(kept)
        return len(kept)

    def _harvest(self, mask, chunk_idx, chunk_seed):
        env = self.env
        idx = mask.nonzero(as_tuple=False).flatten()
        a = {k: v[idx].cpu().tolist() for k, v in self.acc.acc.items()}
        term = {
            "timeout": env.time_out_buf[idx].cpu().tolist(),
            "fall": env.fall[idx].cpu().tolist(),
            "vel_violate": env.vel_violate[idx].cpu().tolist(),
            "contact": _get(env, "term_contact", idx),
            "proj_grav_z": _get(env, "term_proj_grav_z", idx, default=-1.0),
            "base_z_rel": _get(env, "term_base_z_rel", idx, default=1.0),
            "base_vel_z": _get(env, "term_base_vel_z", idx, default=0.0),
        }
        levels = env.terrain_levels[idx].cpu().tolist()
        out = []
        for j, env_i in enumerate(idx.cpu().tolist()):
            n = max(1.0, a["n_steps"][j])
            rec = M.blank_record()
            rec.update(self.meta)
            rec.update({
                "episode_idx": -1,          # assigned by the caller
                "chunk_idx": chunk_idx, "seed": int(chunk_seed),
                "env_idx": env_i, "terrain_level": int(levels[j]),
                "n_steps": int(a["n_steps"][j]),
                "n_collision_steps": int(a["n_collision_steps"][j]),
                "collided": int(a["n_collision_steps"][j] > 0),
                "n_proximity_steps": int(a["n_proximity_steps"][j]),
                "n_viol_steps": int(a["n_viol_steps"][j]),
                "return": a["return"][j],
                "vel_err": a["vel_err_sum"][j] / n,
                "mean_lat_vel": a["lat_vel_sum"][j] / n,
                "mean_fwd_vel": a["fwd_vel_sum"][j] / n,
                "mean_h": a["h_sum"][j] / n,
                "min_h": a["h_min"][j],
                "activation_steps": int(a["activation_steps"][j]),
                "infeasible_steps": int(a["infeasible_steps"][j]),
                "sum_dnorm_v": a["sum_dnorm_v"][j],
                "sum_dnorm_omega": a["sum_dnorm_omega"][j],
                "sum_yaw_cmd_abs": a["sum_yaw_cmd_abs"][j],
                "sum_yaw_realised_abs": a["sum_yaw_realised_abs"][j],
                "sum_yaw_signed": a["sum_yaw_signed"][j],
                "yaw_cmd_steps": a["yaw_cmd_steps"][j],
                "target_miss_steps": int(a["target_miss_steps"][j]),
                "trigger_steps": int(a["trigger_steps"][j]),
                "sum_dnorm_inf": a["sum_dnorm_inf"][j],
                "max_dnorm_inf": a["max_dnorm_inf"][j],
                "sum_dnorm_l2": a["sum_dnorm_l2"][j],
                "mean_q_gap": (a["q_gap_sum"][j] / a["q_gap_n"][j]
                               if a["q_gap_n"][j] > 0 else 0.0),
                "n_q_gap": int(a["q_gap_n"][j]),
                "handoffs": int(a["handoffs"][j]),
                "peak_yaw": a["peak_yaw"][j],
                "peak_jerk": a["peak_jerk"][j],
                "steps_flight": int(a["steps_flight"][j]),
                "steps_partial": int(a["steps_partial"][j]),
                "steps_stance": int(a["steps_stance"][j]),
                "dnorm_flight": a["dnorm_flight"][j],
                "dnorm_partial": a["dnorm_partial"][j],
                "dnorm_stance": a["dnorm_stance"][j],
                "term_timeout": int(bool(term["timeout"][j])),
                "term_fall_flag": int(bool(term["fall"][j])),
                "term_vel_violate": int(bool(term["vel_violate"][j])),
                "term_contact": int(bool(term["contact"][j])),
                "term_proj_grav_z": term["proj_grav_z"][j],
                "term_base_z_rel": term["base_z_rel"][j],
                "term_base_vel_z": term["base_vel_z"][j],
            })
            rec["fell"] = int(M.is_fall(rec))
            out.append(rec)
        self.acc.reset_envs(mask)
        return out

    # -- driver ----------------------------------------------------------
    def run(self, n_episodes, seed_list, verbose=True):
        plan = S.chunk_plan(seed_list, self.env.num_envs, n_episodes)
        t0 = time.time()
        for chunk_idx, chunk_seed, keep in plan:
            got = self.run_chunk(chunk_idx, chunk_seed, keep)
            if verbose:
                print("[eval] chunk {}/{} seed={} kept={} total={} "
                      "({:.0f}s elapsed)".format(
                          chunk_idx + 1, len(plan), chunk_seed, got,
                          len(self.records), time.time() - t0))
        for i, r in enumerate(self.records):
            r["episode_idx"] = i
        return self.records


def _get(env, attr, idx, default=None):
    t = getattr(env, attr, None)
    if t is None:
        return [default] * int(idx.numel())
    return t[idx].cpu().tolist()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_records(path, records):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=M.EPISODE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    return path


def read_records(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def sha256_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def write_manifest(path, **fields):
    """Provenance for §2's "verify before comparing" rule.

    A comparison against the paper's rows is only valid if the pinned
    checkpoint, seed list, eval module and DR setting all match; this file is
    what makes that checkable after the fact.
    """
    blob = dict(fields)
    blob.setdefault("git_commit", git_commit())
    blob.setdefault("torch_version", torch.__version__)
    blob.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    blob.setdefault("eval_module_sha", sha256_file(os.path.abspath(__file__)))
    blob.setdefault("metrics_module_sha", sha256_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.py")))
    blob.setdefault("seed_file_sha", S.file_hash())
    blob.setdefault("thresholds", {
        "collision": "geometric link-vs-cylinder (play_plan.py)",
        "cylinder_height": M.CYLINDER_HEIGHT,
        "body_collision_tol": M.BODY_COLLISION_TOL,
        "proximity_h_thresh": M.COLLISION_H_THRESH,
        "joint_viol_margin": M.JOINT_VIOL_MARGIN,
        "fall_tilt_proj_grav_z": M.FALL_TILT_PROJ_GRAV_Z,
        "fall_height": M.FALL_HEIGHT,
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return path
