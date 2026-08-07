"""E4-Y pre-flight: what authority does the command interface actually have?

This runs *before* the filter is built, and it is a gate as well as a result.

A command-space filter can only be as good as the policy's ability to track
the commands it issues.  The completed braking-only study found the policy had
been trained with `lin_vel_y` pinned to [0, 0], so a lateral command was
fiction.  Yaw is the axis that matters for steering, and its trained range is
narrow too -- so before claiming that steering does or does not rescue command
filtering, measure what the interface can actually deliver.

Three outputs:

  * **trained ranges**, read straight from the config;
  * **tracking gain and saturation** per axis, from an open-loop sweep on flat
    ground with no obstacles -- command a setpoint, measure what the base
    actually does;
  * **minimum turn radius** at the nominal speed, compared against the
    corridor's width and lateral clearances.  This single number says in
    advance whether steering-based evasion is geometrically possible here.

The branch it decides: good yaw tracking makes E4-Y a fair test of the
steering mechanism; degenerate yaw tracking makes it a fair test of
*retrofitting onto this policy* and a weak test of the mechanism, which must
then be said plainly rather than discovered by a reader.

Usage:
    python experiments/e4y/calibrate_commands.py --load_run pi_nom \
        --task go1_amp --headless
"""

import argparse
import json
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--omega_setpoints", type=float, nargs="*",
                   default=[0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0])
    p.add_argument("--vy_setpoints", type=float, nargs="*",
                   default=[0.0, 0.2, -0.2, 0.4, -0.4])
    p.add_argument("--vx", type=float, default=0.6)
    p.add_argument("--hold_s", type=float, default=5.0)
    p.add_argument("--settle_s", type=float, default=1.0,
                   help="discarded before measuring, so the transient after a "
                        "setpoint change does not enter the gain")
    p.add_argument("--eval_envs", type=int, default=100)
    p.add_argument("--corridor_width", type=float, default=2.5)
    p.add_argument("--min_clearance", type=float, default=0.8)
    p.add_argument("--out", type=str, default="logs/e4y/calibration.json")
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
import torch  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402


def measure(env, policy, rt, axis, value, vx, hold_steps, settle_steps):
    """Hold one setpoint and report what the base actually did."""
    env.commands[:, 0] = vx
    env.commands[:, 1] = value if axis == "vy" else 0.0
    env.commands[:, 2] = value if axis == "omega" else 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0

    obs, _, _ = env.reset()
    rt.reset(obs)
    acc = {"yaw": [], "vy": [], "vx": []}
    survived = []
    for t in range(settle_steps + hold_steps):
        rt.maybe_update_world_model()
        # The command buffer is re-randomised on reset; re-pin every step.
        env.commands[:, 0] = vx
        env.commands[:, 1] = value if axis == "vy" else 0.0
        env.commands[:, 2] = value if axis == "omega" else 0.0
        # The policy reads the command out of the observation, so refresh it
        # after writing rather than letting the policy act on last step's.
        env.compute_observations()
        obs = env.obs_buf
        with torch.no_grad():
            a = policy(obs.detach(), rt.history().detach(),
                       rt.wm_feature.detach(), grid=rt.occ_grid)
        obs, _, _, _, _, _, _ = env.step(a.detach())
        if t >= settle_steps:
            # Did the command we wrote survive the step, or did something in
            # the env overwrite it? A gain of zero means nothing about the
            # policy if the command never reached it.
            col = 2 if axis == "omega" else 1
            survived.append(float((env.commands[:, col] - value).abs().mean()))
            acc["yaw"].append(float(env.base_ang_vel[:, 2].mean()))
            acc["vy"].append(float(env.base_lin_vel[:, 1].mean()))
            acc["vx"].append(float(env.base_lin_vel[:, 0].mean()))
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    out["cmd_drift"] = (sum(survived) / len(survived)) if survived else 0.0
    return out


def fit_gain(points, axis):
    """Least-squares slope of realised against commanded, through the origin."""
    num = sum(p["cmd"] * p[axis] for p in points)
    den = sum(p["cmd"] ** 2 for p in points)
    return (num / den) if den > 1e-12 else float("nan")


def main():
    args = get_args()
    args.rl_device = args.sim_device
    os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)

    tcfg, _ = task_registry.get_cfgs(name=args.task)
    rng = tcfg.commands.ranges
    trained = {a: list(getattr(rng, a, [0.0, 0.0]))
               for a in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}

    env, runner, policy, env_cfg, _, resume_path = EC.build_env_and_policy(
        args, "flat", ARGS.eval_envs, dr_mode="off")

    # The yaw-rate command has to actually reach the policy, and by default it
    # does not.  With `heading_command = True` the env recomputes
    # `commands[:, 2]` every step from the heading error, discarding whatever
    # was written there.  How many envs that covers depends on the terrain --
    # `[:roughflat_start_idx]`, which is all of them on flat ground and none
    # of them in the corridor.  So an open-loop sweep on flat ground measures
    # a yaw tracking gain of zero regardless of what the policy can do, while
    # the corridor sweep this calibration is meant to inform is unaffected.
    covered, total = EC.heading_control_coverage(env)
    heading_was_on = EC.disable_heading_command(env)
    if heading_was_on:
        print("[calib] heading_command was ON and covered {}/{} envs on this "
              "terrain; disabled for the sweep so the commanded yaw rate "
              "reaches the policy. The corridor sweep is unaffected either "
              "way -- there its coverage is 0.".format(covered, total))

    rt = EC.PolicyRuntime(env, runner)

    dt = float(env.dt)
    hold = int(ARGS.hold_s / dt)
    settle = int(ARGS.settle_s / dt)

    results = {"omega": [], "vy": []}
    # Absolute drift scales with the setpoint, so a fixed threshold flags a
    # large command and misses a small one.  What matters is the *fraction*
    # of the commanded magnitude that failed to survive: ~1% is the
    # occasional resample or episode reset and is harmless, because the
    # command is re-pinned and the observation recomputed before the policy
    # next acts; ~100% means the axis is being overwritten wholesale.
    drift, drift_frac = 0.0, 0.0
    for w in ARGS.omega_setpoints:
        m = measure(env, policy, rt, "omega", w, ARGS.vx, hold, settle)
        results["omega"].append({"cmd": w, "yaw": m["yaw"], "vx": m["vx"],
                                 "cmd_drift": m["cmd_drift"]})
        drift = max(drift, m["cmd_drift"])
        if abs(w) > 1e-9:
            drift_frac = max(drift_frac, m["cmd_drift"] / abs(w))
        print("  omega cmd %+.2f -> realised %+.3f rad/s  (vx %.3f, cmd drift "
              "%.4f)" % (w, m["yaw"], m["vx"], m["cmd_drift"]))
    for vy in ARGS.vy_setpoints:
        m = measure(env, policy, rt, "vy", vy, ARGS.vx, hold, settle)
        results["vy"].append({"cmd": vy, "vy": m["vy"], "vx": m["vx"],
                              "cmd_drift": m["cmd_drift"]})
        drift = max(drift, m["cmd_drift"])
        if abs(vy) > 1e-9:
            drift_frac = max(drift_frac, m["cmd_drift"] / abs(vy))
        print("  v_y   cmd %+.2f -> realised %+.3f m/s    (vx %.3f, cmd drift "
              "%.4f)" % (vy, m["vy"], m["vx"], m["cmd_drift"]))

    gain_w = fit_gain(results["omega"], "yaw")
    gain_vy = fit_gain(results["vy"], "vy")
    w_max = max((abs(p["yaw"]) for p in results["omega"]), default=0.0)
    turn_r = (ARGS.vx / w_max) if w_max > 1e-6 else float("inf")

    lateral_authority = abs(trained["lin_vel_y"][1] - trained["lin_vel_y"][0]) > 1e-9
    yaw_authority = abs(trained["ang_vel_yaw"][1] - trained["ang_vel_yaw"][0]) > 1e-9

    blob = {
        "trained_ranges": trained,
        "lateral_command_authority": lateral_authority,
        "yaw_command_authority": yaw_authority,
        "vx_nominal": ARGS.vx,
        "omega": results["omega"], "vy": results["vy"],
        "yaw_tracking_gain": gain_w,
        "vy_tracking_gain": gain_vy,
        "max_realised_yaw_rate": w_max,
        "min_turn_radius_m": turn_r,
        "corridor_width_m": ARGS.corridor_width,
        "min_clearance_m": ARGS.min_clearance,
        "steering_geometrically_possible": turn_r < ARGS.corridor_width,
        "checkpoint": resume_path,
        "git_commit": EC.git_commit(),
    }
    # Before any tracking gain means anything, the robot has to be walking.
    # Forward speed is the axis the policy was definitely trained on, at the
    # dead centre of its trained range, so if it cannot track that, nothing
    # measured here is about the command interface.  The first version of this
    # script ran on wave terrain by mistake and reported vx = 0.05 m/s against
    # a 0.6 m/s command; the yaw gain it produced was meaningless and there
    # was no check to say so.
    vx_realised = (sum(p["vx"] for p in results["omega"]) /
                   max(len(results["omega"]), 1))
    vx_gain = vx_realised / ARGS.vx if ARGS.vx > 1e-9 else float("nan")

    blob["heading_command_was_on"] = heading_was_on
    blob["heading_control_covered_envs"] = covered
    blob["max_cmd_drift"] = drift
    blob["max_cmd_drift_fraction"] = drift_frac
    blob["vx_realised_mean"] = vx_realised
    blob["vx_tracking_gain"] = vx_gain
    blob["locomotion_ok"] = bool(vx_gain > 0.5)
    # A gain near 1 up to a clear saturation point means the filter's commands
    # are real; a gain near 0 means it is issuing instructions into the void.
    #
    # But a gain near 0 also results from the command never arriving, and the
    # two must not be confused: one is a property of the policy, the other a
    # bug in the measurement. `cmd_drift` is how far the command had moved by
    # the end of each step, so a large value means something overwrote it and
    # the gain says nothing about the policy at all.
    #
    # Order matters: not-walking invalidates everything, a systematically
    # overwritten command invalidates the axis, and only if both are clean
    # does a low gain say something about the policy.
    if not blob["locomotion_ok"]:
        blob["verdict"] = (
            "MEASUREMENT INVALID -- the robot is not tracking forward speed "
            "({:.3f} m/s realised against {:.2f} commanded, gain {:.2f}). It "
            "is not walking, so no tracking gain here is about the command "
            "interface. Check the terrain and the checkpoint before reading "
            "anything else.".format(vx_realised, ARGS.vx, vx_gain))
    elif drift_frac > 0.10:
        blob["verdict"] = (
            "MEASUREMENT INVALID -- the commanded setpoint did not survive "
            "the step ({:.0f}% of the commanded magnitude). Something in the "
            "env is overwriting the command, so this sweep says nothing about "
            "what the policy can track. Do not read a low gain as a policy "
            "limitation.".format(100 * drift_frac))
    else:
        blob["verdict"] = (
            "yaw tracking usable" if gain_w > 0.5 else
            "yaw tracking degenerate -- E4-Y tests retrofitting onto this "
            "policy, not the steering mechanism in its native setting")

    with open(ARGS.out, "w") as fh:
        json.dump(blob, fh, indent=2)

    print("\n" + "=" * 70)
    print("trained ranges          : %s" % trained)
    print("yaw tracking gain       : %.3f  (realised / commanded)" % gain_w)
    print("lateral tracking gain   : %.3f" % gain_vy)
    print("max realised yaw rate   : %.3f rad/s" % w_max)
    print("min turn radius @%.1f m/s: %.2f m   vs corridor width %.1f m"
          % (ARGS.vx, turn_r, ARGS.corridor_width))
    print("steering possible here  : %s" % blob["steering_geometrically_possible"])
    print("forward tracking gain   : %.3f  (%.3f m/s realised / %.2f commanded)  %s"
          % (vx_gain, vx_realised, ARGS.vx,
             "OK -- the robot is walking" if blob["locomotion_ok"]
             else "*** NOT WALKING; nothing below is meaningful ***"))
    print("command drift (max)     : %.5f  (%.1f%% of commanded)  %s" % (
        drift, 100 * drift_frac,
        "OK -- the setpoint survived each step" if drift_frac <= 0.10
        else "*** the setpoint was OVERWRITTEN; gain is meaningless ***"))
    print("VERDICT: %s" % blob["verdict"])
    print("wrote %s" % ARGS.out)
    print("=" * 70)


if __name__ == "__main__":
    main()
