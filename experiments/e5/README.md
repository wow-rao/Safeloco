# E5 — Bipedal dynamic safety: fall avoidance as reachability

The reviewer's point this experiment answers: quadruped "safety" (obstacle
clearance, joint limits) is quasi-static in configuration space, while bipedal
safety — *not falling* — is dynamic: an upright biped with the wrong momentum
is already doomed, and no configuration-space test can see it. The definition,
its grounding in the paper's formalism, and the margin design live in
`docs/bipedal_dynamic_safety.md`. This directory is the runnable experiment.

## Arms

| arm | safety reward shaping | reachability critic + projection |
|---|---|---|
| `pi_nom` | – | – (critic fitted diagnostically, zero policy influence) |
| `pi_rs` | `rewards.scales.safety_value = 1.0` | – |
| `pi_ours` | – | `safety_coef = 0.1`, damped null-space projection |
| `pi_ours_dcm` | – | as `pi_ours`, margin `fail_set_dcm` (ablation B) |

All arms share the task (Cassie velocity tracking, 70% flat / 30% stairs,
training pushes at 1.5 m/s), the standard termination penalty, and the seeds.
They differ only in the *mechanism* carrying the fall-safety signal.

## Registered predictions (fixed before running; reported verbatim either way)

  (i)   `pi_ours` falls/1000-steps < `pi_nom` at push magnitudes >= 1.0 m/s
        (vx = 0.8, flat), with non-overlapping episode-clustered 95% CIs;
  (ii)  `pi_rs` lands between `pi_nom` and `pi_ours`;
  (iii) the learned reachability value predicts falls better than the static
        fail-set margin: AUC(-V_safe) > AUC(-l_fall) for "falls within 1 s",
        and V_safe's zero crossing leads the margin's by a positive median
        lead time on fallen episodes;
  (iv)  the tracking cost of `pi_ours` is bounded: vel_err(pi_ours) -
        vel_err(pi_nom) < 0.1 m/s at zero push magnitude.

Falsification of (i) or (iii) falsifies the E5 claim; `analyze_e5.py` applies
the rules mechanically and prints whichever verdict the numbers support.

## Run order

```bash
# 0. offline gates (any torch machine, no GPU / isaacgym needed)
python tests/test_fall_margin.py
python tests/test_ppo_safety_guard.py
python tests/test_e5_offline.py

# 1. construction smoke + nominal-height probe
python experiments/e5/train_biped.py --policy pi_nom --task cassie --headless \
    --num_envs 64 --probe_steps 60

# 2. short training smoke (safety loss must be 0 for pi_nom; no NaNs)
python experiments/e5/train_biped.py --policy pi_nom --task cassie --headless \
    --num_envs 256 --max_iterations 30

# 3. full training (add --margin_mode fail_set_dcm for the ablation arm)
python experiments/e5/train_biped.py --policy pi_nom  --task cassie --headless
python experiments/e5/train_biped.py --policy pi_rs   --task cassie --headless
python experiments/e5/train_biped.py --policy pi_ours --task cassie --headless

# 4. eval sweep + analysis
PI_NOM_RUN=pi_nom PI_RS_RUN=pi_rs PI_OURS_RUN=pi_ours \
    bash experiments/e5/sweep_e5.sh
```

Checkpoints land in `logs/cassie_viploco_warp_v2/<run_name>/`; the sweep
writes per-cell CSV + manifest + per-step npz under `logs/e5/sweep` and the
analysis under `logs/e5/R5` (`table_r5a.md`, `verdict.txt`, `summary.json`,
figures).

## Files

- `train_biped.py` — the three arms (+ `--probe_steps` for the h_nom check)
- `run_e5.py` — one eval cell: pinned command, directed push schedule
- `analyze_e5.py` — Table R5a, sanity checks, paired McNemar, calibration
- `sweep_e5.sh` — the grid
- margin math: `safeloco_eval/fall_margin.py`; eval loop:
  `safeloco_eval/eval_biped.py`; env override:
  `legged_gym/envs/cassie/cassie.py::_compute_safety_value`

## Notes

- `pi_nom` keeps the standard `termination = -200` penalty: the arms differ
  in mechanism, not in whether falling is penalized. That is deliberate —
  the claim is that termination-based incentives alone leave a fall rate on
  the table that the reachability gradient removes.
- Time-to-fall is censored by timeouts; falls/1000-steps is the
  censoring-robust headline. The analyzer reports censored counts.
- V_safe is policy-conditioned. Calibration is reported per arm on that
  arm's own rollouts; `pi_nom`/`pi_rs` get a diagnostically-fitted critic
  (`train_safety_critic_when_off`) so the comparison exists on every arm.
- Eval disables the base env's velocity-error termination
  (`fall_safety.disable_vel_violate`) so a hard backward push that the robot
  survives is not censored as a reset.
