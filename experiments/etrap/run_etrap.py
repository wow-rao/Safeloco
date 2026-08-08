"""ETRAP -- one row of the step-trap study.

The claim being measured
------------------------
A body-level (command-space) safety controller reasons about obstacles as
regions of the plane.  On the step-trap terrain the robot spawns inside a
closed pen whose only crossable side is a low sill ahead: projected to 2-D
the pen is a closed ring, so for *any* planar controller -- braking-only or
with full steering authority -- the free space inside is disconnected from
the outside and the safe thing to do is park.  A policy trained under the
gradient-projection method with the volumetric l-value
(`legged_gym/utils/box_sdf.py`) has the space above the sill inside its safe
set, and can exit by lifting its swing legs higher than usual.

This script produces one row of that comparison: a policy checkpoint run
either unfiltered or under the E4-Y unicycle CBF command filter (the
steel-man body-level controller, given steering authority *beyond* what the
policy was trained to track, on purpose -- the point is that authority does
not help when the 2-D free space has no exit).

Per-episode outputs (schema below): whether the robot exited the pen, when,
how far it progressed beyond the gate, whether it parked, how often any link
actually intersected a wall volume (a crossing with clearance is NOT contact
-- that distinction is what the volumetric test exists for), and the swing
apex over the gate (the mechanism evidence).

Usage:
    # the body-level controller arm (expected: exit rate 0, parked)
    python experiments/etrap/run_etrap.py --task go1_amp --headless \
        --load_run pi_nom --policy_tag pi_nom --controller body

    # the gradient-projection arm (expected: exits by stepping over)
    python experiments/etrap/run_etrap.py --task go1_amp --headless \
        --load_run pi_trap_ours --policy_tag pi_trap_ours --controller none
"""

import argparse
import csv
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--controller", type=str, default="none",
                   choices=["none", "body", "body_noyaw"],
                   help="'body' = unicycle CBF command filter with steering "
                        "(E4-Y steel-man); 'body_noyaw' = same filter, "
                        "braking only; 'none' = raw policy")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="CBF class-K gain (small = conservative)")
    p.add_argument("--lookahead", type=float, default=0.35)
    p.add_argument("--w_v", type=float, default=4.0)
    p.add_argument("--w_omega", type=float, default=1.0)
    p.add_argument("--omega_limit", type=float, default=1.0,
                   help="steering authority granted to the filter. The "
                        "default deliberately EXCEEDS pi_nom's trained yaw "
                        "range: the study's point is that steering authority "
                        "cannot open an exit that does not exist in 2-D.")
    p.add_argument("--barrier", type=str, default="geometric",
                   choices=["geometric", "sv"])
    p.add_argument("--body_extent", type=float, default=None)
    p.add_argument("--max_passes", type=int, default=12)
    p.add_argument("--policy_tag", type=str, default="pi_nom",
                   help="row label for the policy; pass alongside --load_run")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--eval_envs", type=int, default=250)
    p.add_argument("--eval_seed_file", type=str, default=None)
    p.add_argument("--dr", type=str, default="off")
    p.add_argument("--gate_difficulty", type=float, default=None,
                   help="pin every terrain row to this difficulty in [0,1] "
                        "(one sill height). Default: rows spread 0..1, i.e. "
                        "sills 0.08..0.16 m across envs.")
    p.add_argument("--exit_margin", type=float, default=0.25,
                   help="metres past the gate's outer face that count as out")
    p.add_argument("--wall_contact_tol", type=float, default=None,
                   help="SDF threshold for wall contact; defaults to the "
                        "collision tolerance the corridor studies use (0.03)")
    p.add_argument("--out_dir", type=str, default="logs/etrap/sweep")
    p.add_argument("--run_id", type=str, default=None)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
import torch  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils import box_sdf  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402
from safeloco_eval import seeds as S  # noqa: E402
from safeloco_eval import stats as ST  # noqa: E402
from safeloco_eval.command_filters import (  # noqa: E402
    build_command_filter, BODY_COLLISION_TOL)


# Own schema: the step-trap questions (exit, progress, lift) are not the
# corridor questions, so this experiment writes its own CSV rather than
# forcing them into the frozen EPISODE_FIELDS. eval_common stays untouched.
TRAP_FIELDS = [
    "run_id", "method", "variant", "policy_tag", "controller",
    "epsilon_or_alpha", "seed", "episode_idx", "chunk_idx", "env_idx",
    "terrain_level", "gate_height", "n_steps",
    # the headline
    "exited", "exit_step", "progress_x", "progress_beyond_gate", "parked",
    # realised motion
    "mean_fwd_vel", "vel_err", "mean_abs_yaw_rate",
    # wall interaction (volumetric: passing over with clearance is NOT contact)
    "wall_contact_steps", "contacted_wall",
    # mechanism evidence: swing apex over the gate vs on open pen floor
    "gate_foot_apex", "interior_foot_apex",
    # the l-value along the way
    "min_l", "mean_l",
    # command-filter diagnostics
    "trigger_steps", "activation_steps", "infeasible_steps",
    "sum_dnorm_v", "sum_dnorm_omega",
    # termination diagnostics (fall definition recomputable post hoc)
    "term_timeout", "term_fall_flag", "term_vel_violate", "term_contact",
    "term_proj_grav_z", "term_base_z_rel", "fell",
]

FALL_TILT_PROJ_GRAV_Z = -0.5   # matches safeloco_eval.metrics
FALL_HEIGHT = 0.15


def blank_record():
    return {k: 0 for k in TRAP_FIELDS}


class TrapCollector(object):
    """Steps the env (optionally under a command filter) and harvests
    step-trap episode records.

    Timing convention matches eval_common.EvalCollector: per-step state
    quantities (position, foot heights, wall contact, velocities) are read
    from the state the policy acts on, immediately BEFORE env.step -- which
    also sidesteps the fact that reset_idx overwrites root_states of done
    envs before step() returns.  Per-transition quantities (reward, the
    l-value `cbf_h_min`) are read after env.step from pre-reset values.
    """

    def __init__(self, env, runner, policy, cmd_filt, exit_margin,
                 wall_tol, run_meta):
        self.env = env
        self.runner = runner
        self.policy = policy
        self.cmd_filt = cmd_filt
        self.exit_margin = float(exit_margin)
        self.wall_tol = float(wall_tol)
        self.meta = dict(run_meta or {})
        self.rt = EC.PolicyRuntime(env, runner)
        self.clip_actions = float(env.cfg.normalization.clip_actions)
        self.records = []

        if not bool(env.has_box_obstacles.any()):
            raise RuntimeError(
                "no env publishes box obstacles -- this is not the step-trap "
                "terrain. Check --eval_terrain wiring (should be step_trap).")

        self._refresh_gate_params()

    # -- per-env gate geometry, from the terrain's published info ---------
    def _refresh_gate_params(self):
        env = self.env
        E = env.num_envs
        gx_outer = torch.full((E,), 1e6, device=env.device)
        gx_inner = torch.full((E,), 1e6, device=env.device)
        gy_half = torch.zeros(E, device=env.device)
        gh = torch.zeros(E, device=env.device)
        for e in range(E):
            key = (int(env.terrain_levels[e].item()),
                   int(env.terrain_types[e].item()))
            info = env.corridor_obstacles.get(key)
            if not info or "gate" not in info:
                continue
            gx_outer[e] = float(info["gate"]["x_outer"])
            gx_inner[e] = float(info["gate"]["x_inner"])
            gy_half[e] = float(info["gate"]["y_half"])
            gh[e] = float(info["gate"]["height"])
        self.gate_x_outer = gx_outer
        self.gate_x_inner = gx_inner
        self.gate_y_half = gy_half
        self.gate_height = gh

    # -- main loop --------------------------------------------------------
    def run_chunk(self, chunk_idx, chunk_seed, keep, max_steps=None):
        from legged_gym.utils.helpers import set_seed
        env = self.env
        E = env.num_envs
        dev = env.device

        set_seed(int(chunk_seed))
        obs, privileged_obs, _ = env.reset()
        self.rt.reset(obs)
        self._refresh_gate_params()

        nominal_cmd = None
        if self.cmd_filt is not None:
            nominal_cmd = env.commands.clone()
            rng = env.cfg.commands.ranges
            for attr in ("lin_vel_x", "lin_vel_y"):
                lo, hi = getattr(rng, attr, [0.0, 0.0])
                if abs(hi - lo) > 1e-9:
                    raise RuntimeError(
                        "command filter assumes a pinned eval command, but "
                        "{} is [{}, {}].".format(attr, lo, hi))
        cmd_vx = float(env.commands[0, 0].item())

        z = lambda dtype=torch.float: torch.zeros(E, device=dev, dtype=dtype)
        acc = {
            "n_steps": z(), "exited": z(torch.bool),
            "exit_step": torch.full((E,), -1.0, device=dev),
            "x_start": z(), "x_last": z(),
            "x_half": torch.full((E,), float("nan"), device=dev),
            "fwd_vel_sum": z(), "vel_err_sum": z(), "yaw_abs_sum": z(),
            "wall_contact_steps": z(),
            "gate_foot_apex": z(), "interior_foot_apex": z(),
            "min_l": torch.full((E,), float("inf"), device=dev),
            "l_sum": z(), "return": z(),
            "trigger_steps": z(), "activation_steps": z(),
            "infeasible_steps": z(),
            "sum_dnorm_v": z(), "sum_dnorm_omega": z(),
        }

        x_local0 = env.root_states[:, 0] - env.env_origins[:, 0]
        acc["x_start"] = x_local0.clone()
        acc["x_last"] = x_local0.clone()

        live = torch.ones(E, device=dev)
        finished = torch.zeros(E, dtype=torch.bool, device=dev)
        max_steps = max_steps or int(env.max_episode_length) + 3
        half_step = int(env.max_episode_length) // 2
        kept = []

        for it in range(max_steps):
            self.rt.maybe_update_world_model()

            # ---- pre-step state metrics ---------------------------------
            x_local = env.root_states[:, 0] - env.env_origins[:, 0]
            acc["x_last"] = torch.where(live.bool(), x_local, acc["x_last"])
            out_now = x_local > (self.gate_x_outer + self.exit_margin)
            newly_out = out_now & (~acc["exited"]) & live.bool()
            acc["exit_step"] = torch.where(
                newly_out, torch.full_like(acc["exit_step"], float(it)),
                acc["exit_step"])
            acc["exited"] = acc["exited"] | newly_out
            if it == half_step:
                acc["x_half"] = torch.where(
                    live.bool(), x_local, acc["x_half"])

            acc["fwd_vel_sum"] += env.base_lin_vel[:, 0] * live
            acc["vel_err_sum"] += (env.base_lin_vel[:, 0] - cmd_vx).abs() * live
            acc["yaw_abs_sum"] += env.base_ang_vel[:, 2].abs() * live

            feet_pos = env.rigid_body_pos[:, env.feet_indices]      # [E,4,3]
            feet_x = feet_pos[..., 0] - env.env_origins[:, 0:1]
            feet_y = feet_pos[..., 1] - env.env_origins[:, 1:2]
            feet_z = feet_pos[..., 2] - env.env_origins[:, 2:3]
            gate_mid = 0.5 * (self.gate_x_inner + self.gate_x_outer)
            gate_halfw = 0.5 * (self.gate_x_outer - self.gate_x_inner) + 0.15
            in_gate = ((feet_x - gate_mid.unsqueeze(1)).abs()
                       <= gate_halfw.unsqueeze(1)) \
                & (feet_y.abs() <= self.gate_y_half.unsqueeze(1))
            gate_apex = (feet_z * in_gate).amax(dim=1)
            acc["gate_foot_apex"] = torch.maximum(
                acc["gate_foot_apex"], gate_apex * live)
            interior = (feet_x.abs() <= 1.2) & (feet_y.abs() <= 1.2)
            int_apex = (feet_z * interior).amax(dim=1)
            acc["interior_foot_apex"] = torch.maximum(
                acc["interior_foot_apex"], int_apex * live)

            contact = box_sdf.any_point_in_boxes(
                env.rigid_body_pos, env.box_obstacles,
                env.box_obstacle_mask, tol=self.wall_tol)
            acc["wall_contact_steps"] += contact.float() * live

            # ---- command filter (the body-level controller) --------------
            if self.cmd_filt is not None:
                cmd_new, cinfo = self.cmd_filt.apply(nominal_cmd, env)
                env.commands[:, :3] = cmd_new[:, :3]
                env.compute_observations()
                obs = env.obs_buf
                acc["trigger_steps"] += cinfo["triggered"].float() * live
                acc["activation_steps"] += cinfo["activated"].float() * live
                acc["infeasible_steps"] += cinfo["infeasible"].float() * live
                if "dnorm_v" in cinfo:
                    acc["sum_dnorm_v"] += cinfo["dnorm_v"] * live
                    acc["sum_dnorm_omega"] += cinfo["dnorm_omega"] * live

            # ---- policy --------------------------------------------------
            with torch.no_grad():
                a_pol = self.policy(obs.detach(), self.rt.history().detach(),
                                    self.rt.wm_feature.detach(),
                                    grid=self.rt.occ_grid)
                a_pol = a_pol.clamp(-self.clip_actions, self.clip_actions)

            obs, privileged_obs, rews, dones, infos, reset_ids, _ = env.step(
                a_pol.detach())

            # ---- post-step transition metrics (pre-reset values) ---------
            l_now = infos["cbf_h_min"].to(dev)
            acc["n_steps"] += live
            acc["l_sum"] += l_now * live
            acc["min_l"] = torch.minimum(
                acc["min_l"], torch.where(live.bool(), l_now, acc["min_l"]))
            acc["return"] += rews.to(dev) * live

            self.rt.post_step(obs, infos, dones, reset_ids)

            just_done = dones.bool() & (~finished)
            if just_done.any():
                kept.extend(self._harvest(acc, just_done, chunk_idx,
                                          chunk_seed))
                finished |= just_done
                live = (~finished).float()
            if bool(finished.all()):
                break

        kept.sort(key=lambda r: r["env_idx"])
        if keep < len(kept):
            stride = len(kept) / float(keep)
            kept = [kept[int(i * stride)] for i in range(keep)]
        self.records.extend(kept)
        return len(kept)

    def _harvest(self, acc, mask, chunk_idx, chunk_seed):
        env = self.env
        idx = mask.nonzero(as_tuple=False).flatten()
        out = []
        for j in idx.cpu().tolist():
            n = max(1.0, float(acc["n_steps"][j].item()))
            x_last = float(acc["x_last"][j].item())
            x_half = float(acc["x_half"][j].item())
            timeout = bool(env.time_out_buf[j].item())
            exited = bool(acc["exited"][j].item())
            # Parked: ran to timeout inside the pen with next to no progress
            # over the whole second half of the episode.
            parked = int(timeout and not exited
                         and (x_half == x_half)          # not NaN
                         and (x_last - x_half) < 0.25)
            gx_out = float(self.gate_x_outer[j].item())
            rec = blank_record()
            rec.update(self.meta)
            rec.update({
                "seed": int(chunk_seed), "episode_idx": -1,
                "chunk_idx": chunk_idx, "env_idx": int(j),
                "terrain_level": int(env.terrain_levels[j].item()),
                "gate_height": float(self.gate_height[j].item()),
                "n_steps": int(n),
                "exited": int(exited),
                "exit_step": int(acc["exit_step"][j].item()),
                "progress_x": x_last - float(acc["x_start"][j].item()),
                "progress_beyond_gate": max(0.0, x_last - gx_out),
                "parked": parked,
                "mean_fwd_vel": float(acc["fwd_vel_sum"][j].item()) / n,
                "vel_err": float(acc["vel_err_sum"][j].item()) / n,
                "mean_abs_yaw_rate": float(acc["yaw_abs_sum"][j].item()) / n,
                "wall_contact_steps": int(acc["wall_contact_steps"][j].item()),
                "contacted_wall": int(acc["wall_contact_steps"][j].item() > 0),
                "gate_foot_apex": float(acc["gate_foot_apex"][j].item()),
                "interior_foot_apex": float(
                    acc["interior_foot_apex"][j].item()),
                "min_l": float(acc["min_l"][j].item()),
                "mean_l": float(acc["l_sum"][j].item()) / n,
                "trigger_steps": int(acc["trigger_steps"][j].item()),
                "activation_steps": int(acc["activation_steps"][j].item()),
                "infeasible_steps": int(acc["infeasible_steps"][j].item()),
                "sum_dnorm_v": float(acc["sum_dnorm_v"][j].item()),
                "sum_dnorm_omega": float(acc["sum_dnorm_omega"][j].item()),
                "term_timeout": int(timeout),
                "term_fall_flag": int(env.fall[j].item()),
                "term_vel_violate": int(env.vel_violate[j].item()),
                "term_contact": int(getattr(env, "term_contact")[j].item()),
                "term_proj_grav_z": float(
                    getattr(env, "term_proj_grav_z")[j].item()),
                "term_base_z_rel": float(
                    getattr(env, "term_base_z_rel")[j].item()),
            })
            rec["fell"] = int(
                rec["term_proj_grav_z"] > FALL_TILT_PROJ_GRAV_Z
                or rec["term_base_z_rel"] < FALL_HEIGHT
                or rec["term_fall_flag"])
            out.append(rec)
        # zero the harvested envs' accumulators so a stray later write
        # (there is none, but cheap insurance) cannot leak across episodes
        for k, v in acc.items():
            if v.dtype == torch.bool:
                continue
            v[idx] = 0.0 if k not in ("min_l",) else float("inf")
        return out

    def run(self, n_episodes, seed_list, verbose=True):
        import time
        plan = S.chunk_plan(seed_list, self.env.num_envs, n_episodes)
        t0 = time.time()
        for chunk_idx, chunk_seed, keep in plan:
            got = self.run_chunk(chunk_idx, chunk_seed, keep)
            if verbose:
                print("[etrap] chunk {}/{} seed={} kept={} total={} "
                      "({:.0f}s elapsed)".format(
                          chunk_idx + 1, len(plan), chunk_seed, got,
                          len(self.records), time.time() - t0))
        for i, r in enumerate(self.records):
            r["episode_idx"] = i
        return self.records


def write_trap_records(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAP_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    return path


def default_run_id():
    if ARGS.controller == "none":
        return "{}_unfiltered".format(ARGS.policy_tag)
    return "{}_{}_a{:g}_L{:g}_w{:g}".format(
        ARGS.policy_tag, ARGS.controller, ARGS.alpha, ARGS.lookahead,
        ARGS.omega_limit)


def main():
    args = get_args()
    args.rl_device = args.sim_device
    run_id = ARGS.run_id or default_run_id()
    os.makedirs(ARGS.out_dir, exist_ok=True)

    # Trained ranges, read before the eval overrides collapse them; also the
    # place to pin the gate difficulty, since get_cfgs returns the shared
    # registered instances that build_env_and_policy will re-fetch.
    tcfg, _ = task_registry.get_cfgs(name=args.task)
    rng = tcfg.commands.ranges
    train_vx = tuple(getattr(rng, "lin_vel_x", [0.0, 0.8]))
    train_w = tuple(getattr(rng, "ang_vel_yaw", [0.0, 0.0]))
    if ARGS.gate_difficulty is not None:
        d = max(0.0, min(1.0, float(ARGS.gate_difficulty)))
        tcfg.terrain.min_difficulty = d
        tcfg.terrain.max_difficulty = d

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, "step_trap", ARGS.eval_envs,
                                dr_mode=ARGS.dr,
                                load_run=getattr(args, "load_run", None))

    if ARGS.controller == "none":
        cmd_filt = None
        omega_range = (0.0, 0.0)
    else:
        if ARGS.controller == "body":
            omega_range = (-abs(ARGS.omega_limit), abs(ARGS.omega_limit))
        else:
            omega_range = (0.0, 0.0)
        body_margin = None
        if ARGS.body_extent is not None:
            body_margin = BODY_COLLISION_TOL + ARGS.body_extent
        cmd_filt = build_command_filter(
            "unicycle", alpha=ARGS.alpha, vx_range=train_vx,
            omega_range=omega_range, max_passes=ARGS.max_passes,
            barrier=ARGS.barrier, body_margin=body_margin,
            lookahead=ARGS.lookahead, w_v=ARGS.w_v, w_omega=ARGS.w_omega)
        print("[etrap] body-level controller: alpha={:g} lookahead={:.2f} "
              "omega range {} (trained {}) -- authority granted beyond the "
              "trained range on purpose; the pen has no 2-D exit for it to "
              "use.".format(ARGS.alpha, ARGS.lookahead, omega_range, train_w))

    wall_tol = (ARGS.wall_contact_tol if ARGS.wall_contact_tol is not None
                else BODY_COLLISION_TOL)
    meta = {"run_id": run_id, "method": "ETRAP",
            "variant": ARGS.controller, "policy_tag": ARGS.policy_tag,
            "controller": ARGS.controller,
            "epsilon_or_alpha": (ARGS.alpha if cmd_filt is not None else "")}

    collector = TrapCollector(env, runner, policy, cmd_filt,
                              exit_margin=ARGS.exit_margin,
                              wall_tol=wall_tol, run_meta=meta)
    seed_list, _ = S.load(ARGS.eval_seed_file)
    records = collector.run(ARGS.n_episodes, seed_list)

    csv_path = os.path.join(ARGS.out_dir, run_id + ".csv")
    write_trap_records(csv_path, records)

    gate_by_level = {}
    for r in records:
        gate_by_level.setdefault(r["terrain_level"], r["gate_height"])
    EC.write_manifest(
        os.path.join(ARGS.out_dir, run_id + ".manifest.json"),
        run_id=run_id, method="ETRAP", controller=ARGS.controller,
        policy_tag=ARGS.policy_tag,
        alpha=(ARGS.alpha if cmd_filt is not None else None),
        omega_range=list(omega_range), trained_yaw_range=list(train_w),
        trained_vx_range=list(train_vx), lookahead=ARGS.lookahead,
        w_v=ARGS.w_v, w_omega=ARGS.w_omega, barrier=ARGS.barrier,
        exit_margin=ARGS.exit_margin, wall_contact_tol=wall_tol,
        gate_difficulty=ARGS.gate_difficulty,
        gate_height_by_level={str(k): v for k, v in
                              sorted(gate_by_level.items())},
        checkpoint=resume_path, checkpoint_sha=EC.sha256_file(resume_path),
        terrain="step_trap", dr_mode=ARGS.dr,
        n_episodes=len(records), eval_envs=ARGS.eval_envs,
        terrain_seed=EC.TERRAIN_SEED, cmd_lin_vel_x=EC.EVAL_CMD_LIN_VEL_X,
        control_dt=float(env.dt),
        collision_metric="volumetric_link_in_wall_box",
    )

    n = len(records)
    k_exit = sum(r["exited"] for r in records)
    p, lo, hi = ST.wilson_ci(k_exit, max(n, 1))
    prog = sum(r["progress_beyond_gate"] for r in records) / max(n, 1)
    parked = sum(r["parked"] for r in records) / max(n, 1)
    apex = [r["gate_foot_apex"] for r in records if r["exited"]]
    print("[etrap] {}  exit={} ({})  progress+={:.2f} m  parked={:.0f}%  "
          "wall-contact eps={:.1f}%  gate apex (exited)={:.3f} m".format(
              run_id, "{}/{}".format(k_exit, n),
              ST.fmt_rate(p, lo, hi),
              prog, 100.0 * parked,
              100.0 * sum(r["contacted_wall"] for r in records) / max(n, 1),
              (sum(apex) / len(apex)) if apex else float("nan")))
    print("  wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
