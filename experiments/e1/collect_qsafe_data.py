"""Step 1 of E1: collect the buffers Q_safe is regressed on.

experiment_protocol_phased.md §2:

    Action-input head on the existing reachability-critic pipeline:
    Q_phi(o_t, a_t) regressed onto the worst-case returns R_safe(t) (Eq. 5)
    using stored (o_t, a_t, R_safe) triples from pi_nom **and** pi_rs corridor
    buffers (mixed substrates broaden action coverage, since the filter queries
    off-manifold actions).  Train/val split by episode.

One addition beyond the protocol, flagged explicitly because it matters for how
E1 is read: **injected action noise** on a configurable fraction of envs
(`--noise_frac`, `--action_noise`).  Mixed substrates broaden coverage over
*states*, but the E1 filter evaluates Q_hat at actions drawn uniformly from an
epsilon-box around a_pol -- actions no rollout policy ever takes.  Without
off-policy action coverage, Q_hat's dependence on `a` is unconstrained by data
and the sweep would be measuring an artefact of extrapolation rather than
measuring projection.  Set `--noise_frac 0` to reproduce the literal protocol.
The choice is recorded in the buffer metadata and in the Q-hat report.

Storage is float16; a 256-env x 1000-step buffer is roughly 1.9 GB with the
occupancy grid included.

Usage
-----
    python experiments/e1/collect_qsafe_data.py --task go1_amp \
        --terrain corridor --headless --load_run <PI_NOM_RUN> \
        --collect_steps 1000 --collect_envs 256 --out logs/e1/qsafe_data/pi_nom
"""

import argparse
import inspect
import json
import os

CURDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT = os.path.dirname(os.path.dirname(CURDIR))
os.sys.path.insert(0, ROOT)

from safeloco_eval.cli import parse_and_strip  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--collect_steps", type=int, default=1000)
    p.add_argument("--collect_envs", type=int, default=256)
    p.add_argument("--collect_terrain", type=str, default="corridor")
    p.add_argument("--action_noise", type=float, default=0.20,
                   help="std of the injected action noise, raw action units "
                        "(0.20 raw == 0.05 rad at action_scale 0.25)")
    p.add_argument("--noise_frac", type=float, default=0.5,
                   help="fraction of envs running the noisy behaviour policy; "
                        "0 reproduces the literal protocol")
    p.add_argument("--store_grid", type=int, default=1)
    p.add_argument("--collect_dr", type=str, default="on",
                   choices=["off", "on"],
                   help="DR *on* by default here: the training distribution "
                        "for Q-hat should be broad. Eval DR is separate.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--collect_seed", type=int, default=12345)
    return p


ARGS = parse_and_strip(build_parser())

import isaacgym  # noqa: E402,F401  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403  (registers tasks)
from legged_gym.utils import get_args  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402

import torch  # noqa: E402

from safeloco_eval import eval_common as EC  # noqa: E402


def main():
    args = get_args()
    args.rl_device = args.sim_device

    env, runner, policy, env_cfg, train_cfg, resume_path = \
        EC.build_env_and_policy(args, ARGS.collect_terrain, ARGS.collect_envs,
                                dr_mode=ARGS.collect_dr)
    device = env.device
    rt = EC.PolicyRuntime(env, runner)

    set_seed(ARGS.collect_seed)
    obs, privileged_obs, _ = env.reset()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    rt.reset(obs)

    T, B = ARGS.collect_steps, env.num_envs
    store_grid = bool(ARGS.store_grid) and rt.grid_enabled
    clip_actions = float(env_cfg.normalization.clip_actions)

    buf = {
        "critic_obs": torch.zeros(T, B, critic_obs.shape[1],
                                  dtype=torch.float16),
        "wm_feature": torch.zeros(T, B, runner.wm_feature_dim,
                                  dtype=torch.float16),
        "action": torch.zeros(T, B, env.num_actions, dtype=torch.float16),
        "safety_value": torch.zeros(T, B, dtype=torch.float32),
        "min_cbf_h": torch.zeros(T, B, dtype=torch.float32),
        "done": torch.zeros(T, B, dtype=torch.bool),
        "noisy": torch.zeros(B, dtype=torch.bool),
    }
    if store_grid:
        buf["grid"] = torch.zeros(T, B, *rt.grid_shape, dtype=torch.float16)

    # Which envs run the noisy behaviour policy -- fixed by index, so the
    # train/val split by episode never mixes an env's clean and noisy halves.
    n_noisy = int(round(ARGS.noise_frac * B))
    noisy = torch.zeros(B, dtype=torch.bool, device=device)
    noisy[:n_noisy] = True
    buf["noisy"] = noisy.cpu()

    gen = torch.Generator(device=device)
    gen.manual_seed(ARGS.collect_seed + 7)

    print("[collect] T={} B={} grid={} noisy_envs={}/{} sigma={}".format(
        T, B, store_grid, n_noisy, B, ARGS.action_noise))

    for t in range(T):
        rt.maybe_update_world_model()
        with torch.no_grad():
            a = policy(obs.detach(), rt.history().detach(),
                       rt.wm_feature.detach(), grid=rt.occ_grid)
            a = a.clamp(-clip_actions, clip_actions)
            if ARGS.action_noise > 0 and n_noisy > 0:
                eps = torch.randn(a.shape, generator=gen, device=device)
                a = torch.where(noisy.unsqueeze(1),
                                (a + ARGS.action_noise * eps).clamp(
                                    -clip_actions, clip_actions), a)

        buf["critic_obs"][t] = critic_obs.detach().half().cpu()
        buf["wm_feature"][t] = rt.wm_feature.detach().half().cpu()
        buf["action"][t] = a.detach().half().cpu()
        if store_grid:
            buf["grid"][t] = rt.occ_grid.detach().half().cpu()

        obs, privileged_obs, rews, dones, infos, reset_ids, _ = env.step(
            a.detach())
        critic_obs = privileged_obs if privileged_obs is not None else obs

        # sv(t) and h(t) are the margins reached by executing a(t); the env
        # computes both before reset_idx, so terminal steps are clean.
        buf["safety_value"][t] = env.safety_values.detach().float().cpu()
        buf["min_cbf_h"][t] = infos["cbf_h_min"].detach().float().cpu()
        buf["done"][t] = dones.bool().cpu()

        rt.post_step(obs, infos, dones, reset_ids)
        if (t + 1) % 100 == 0:
            print("[collect] {}/{} steps, {} episode ends so far".format(
                t + 1, T, int(buf["done"][:t + 1].sum())))

    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)) or ".",
                exist_ok=True)
    torch.save(buf, ARGS.out + ".pt")
    meta = {
        "checkpoint": resume_path,
        "checkpoint_sha": EC.sha256_file(resume_path),
        "terrain": ARGS.collect_terrain,
        "steps": T, "envs": B,
        "action_noise": ARGS.action_noise,
        "noise_frac": ARGS.noise_frac,
        "dr_mode": ARGS.collect_dr,
        "collect_seed": ARGS.collect_seed,
        "store_grid": store_grid,
        "action_scale": float(env_cfg.control.action_scale),
        "clip_actions": clip_actions,
        "num_critic_obs": int(critic_obs.shape[1]),
        "wm_feature_dim": int(runner.wm_feature_dim),
        "grid_shape": list(rt.grid_shape) if store_grid else None,
        "git_commit": EC.git_commit(),
        "n_episode_ends": int(buf["done"].sum()),
    }
    with open(ARGS.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[collect] wrote {}.pt ({} episode ends)".format(
        ARGS.out, meta["n_episode_ends"]))


if __name__ == "__main__":
    main()
