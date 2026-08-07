"""Regression tests for the plain-PPO safety branch (E5's training path).

Two things are pinned here:

1. `RolloutStorage.compute_safety_returns` -- the worst-case (reachability)
   return recursion, checked against hand-computed values on a synthetic fall
   trajectory: negativity propagates backward from the fall, the terminal step
   is masked, the safe env stays at the clamp, and the `alpha` parameter
   actually widens the predictive horizon (it used to be hardcoded).

2. `PPO.update` with `safety_coef == 0` -- the pi_nom configuration.  On the
   pre-fix code `safety_loss` was referenced outside its defining branch and
   the very first update raised NameError, so the vanilla arm could not train
   at all.  Also covers `train_safety_critic_when_off` (critic-only fitting:
   safety loss finite, actor update untouched by construction) and the
   `safety_coef > 0` path accepting `d_safe`/`d_danger`/`safety_return_alpha`.

Needs torch (+ tensorboard, via the rsl_rl module imports); skips cleanly
without them.

Run:  python tests/test_ppo_safety_guard.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
except ImportError:                                            # pragma: no cover
    print("SKIP tests/test_ppo_safety_guard.py -- torch not available")
    sys.exit(0)

try:
    from rsl_rl.storage.rollout_storage import RolloutStorage
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.modules import ActorCriticWMP
except ImportError as e:                                       # pragma: no cover
    print("SKIP tests/test_ppo_safety_guard.py -- {}".format(e))
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        FAILED.append(name)


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


# ------------------------------------------------- safety return recursion --

def _storage(T, E):
    return RolloutStorage(E, T, [3], [3], [1], history_dim=2,
                          wm_feature_dim=2, device="cpu")


def test_safety_returns():
    T, E = 10, 2
    st = _storage(T, E)
    # env 0: safe cruise, margin dips, falls at t=8 (done), fresh episode t=9.
    sv0 = [0.4] * 7 + [-0.2, -1.0, 0.3]
    done0 = [0] * 8 + [1, 0]
    # env 1: safe throughout.
    for t in range(T):
        st.safety_values_buf[t, 0, 0] = sv0[t]
        st.safety_values_buf[t, 1, 0] = 0.5
        st.dones[t, 0, 0] = done0[t]
    last = torch.full((E, 1), 0.5)

    st.compute_safety_returns(last, alpha=0.7)
    R = st.safety_returns[:, 0, 0]

    # Hand-computed, alpha = 0.7:
    #   R[9] = .7*min(.3,.5)   + .3*.3   =  0.30
    #   R[8] = .7*min(-1,0)    + .3*(-1) = -1.00   (terminal masking)
    #   R[7] = .7*min(-.2,-1)  + .3*(-.2)= -0.76
    #   R[6] = .7*min(.4,-.76) + .3*.4   = -0.412
    #   R[5] = -0.1684   R[4] = +0.00212
    for t, want in ((9, 0.30), (8, -1.0), (7, -0.76), (6, -0.412),
                    (5, -0.1684), (4, 0.00212)):
        check("recursion: R[{}] == {:+.5f}".format(t, want),
              approx(R[t], want, tol=1e-4), float(R[t]))

    check("recursion: negativity propagates monotonically toward the fall",
          all(R[t] >= R[t + 1] - 1e-6 for t in range(3, 8)),
          [float(x) for x in R[3:9]])

    Rsafe_env1 = st.safety_returns[:, 1, 0]
    check("recursion: all-safe env pinned at the 0.5 clamp",
          all(approx(x, 0.5) for x in Rsafe_env1),
          [float(x) for x in Rsafe_env1])

    # alpha widens the horizon: with 0.9 the fall reaches t=3 negative,
    # with 0.7 t=3 is still positive.
    st07 = _storage(T, E)
    st09 = _storage(T, E)
    for s in (st07, st09):
        s.safety_values_buf.copy_(st.safety_values_buf)
        s.dones.copy_(st.dones)
    st07.compute_safety_returns(last.clone(), alpha=0.7)
    st09.compute_safety_returns(last.clone(), alpha=0.9)
    r07, r09 = st07.safety_returns[3, 0, 0], st09.safety_returns[3, 0, 0]
    check("recursion: alpha=0.9 sees the fall from farther back",
          r09 < 0 < r07, "alpha .7 -> {:+.4f}, .9 -> {:+.4f}".format(
              float(r07), float(r09)))


# ------------------------------------------------------------ PPO updates ---

PRIV, H, NOBS, NACT = 4, 5, 16, 2


def _actor_critic():
    torch.manual_seed(0)
    return ActorCriticWMP(
        num_actor_obs=NOBS, num_critic_obs=NOBS, num_actions=NACT,
        encoder_hidden_dims=[8], wm_encoder_hidden_dims=[4],
        actor_hidden_dims=[8], critic_hidden_dims=[8],
        latent_dim=4, height_dim=H, privileged_dim=PRIV,
        history_dim=6, wm_feature_dim=3, wm_latent_dim=3)


def _drive(ppo, T=8, E=4, seed=1):
    """Roll synthetic transitions through act/process_env_step/update."""
    torch.manual_seed(seed)
    ppo.init_storage(E, T, [NOBS], [NOBS], [NACT], history_dim=6,
                     wm_feature_dim=3)
    obs = torch.randn(E, NOBS)
    critic_obs = torch.randn(E, NOBS)
    history = torch.randn(E, 6)
    wm_feature = torch.randn(E, 3)
    for t in range(T):
        sv = torch.full((E,), 0.4)
        sv[0] = -0.5 if t >= T - 2 else 0.4       # one env goes unsafe
        ppo.act(obs, critic_obs, history, wm_feature, safety_value=sv)
        rewards = torch.randn(E)
        dones = torch.zeros(E)
        if t == T - 1:
            dones[0] = 1
        ppo.process_env_step(rewards, dones, {})
    ppo.compute_returns(critic_obs, wm_feature)
    return ppo.update()


def test_pi_nom_no_nameerror():
    """The regression pin: safety_coef = 0 must complete an update."""
    ppo = PPO(_actor_critic(), num_learning_epochs=1, num_mini_batches=2,
              device="cpu", safety_coef=0.0)
    try:
        v_loss, s_loss, safety_loss = _drive(ppo)
    except NameError as e:                                     # pre-fix code
        check("pi_nom: update completes with safety_coef=0", False, str(e))
        return
    check("pi_nom: update completes with safety_coef=0", True)
    check("pi_nom: reported safety loss is exactly 0",
          safety_loss == 0.0, safety_loss)


def test_critic_only_fitting():
    ac = _actor_critic()
    before = {k: v.clone() for k, v in ac.safety_critic.state_dict().items()}
    actor_before = {k: v.clone() for k, v in ac.actor.state_dict().items()}
    ppo = PPO(ac, num_learning_epochs=1, num_mini_batches=2, device="cpu",
              safety_coef=0.0, train_safety_critic_when_off=True)
    _, _, safety_loss = _drive(ppo)
    check("critic-only: safety loss finite and nonzero",
          safety_loss > 0.0, safety_loss)
    changed = any(not torch.equal(before[k], v)
                  for k, v in ac.safety_critic.state_dict().items())
    check("critic-only: safety head received gradients", changed)
    # The actor still updates (task loss), just not from the safety branch;
    # parameter-disjointness of the head is what guarantees that.
    actor_changed = any(not torch.equal(actor_before[k], v)
                        for k, v in ac.actor.state_dict().items())
    check("critic-only: actor still trains on the task loss", actor_changed)


def test_ours_path_kwargs():
    ppo = PPO(_actor_critic(), num_learning_epochs=1, num_mini_batches=2,
              device="cpu", safety_coef=0.1, d_safe=-0.05, d_danger=-0.35,
              safety_return_alpha=0.9)
    check("pi_ours: biped kwargs accepted",
          ppo.d_safe == -0.05 and ppo.d_danger == -0.35
          and ppo.safety_return_alpha == 0.9)
    _, _, safety_loss = _drive(ppo)
    check("pi_ours: safety loss finite and nonzero",
          safety_loss > 0.0 and safety_loss == safety_loss, safety_loss)


def main():
    print("tests/test_ppo_safety_guard.py")
    test_safety_returns()
    test_pi_nom_no_nameerror()
    test_critic_only_fitting()
    test_ours_path_kwargs()
    if FAILED:
        print("{} FAILED".format(len(FAILED)))
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
