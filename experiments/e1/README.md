# E1 — Deployment-time joint-space projection, collision

Implements `experiment_protocol_phased.md` §3 (E1), the §2 Q̂_safe construction
it depends on, and the §1.1 shared-invariant machinery, against the actual
Safeloco code (`viploco_codebase_addendum.md` §3.1–3.2).

E1 itself trains no policy: π_nom is frozen and the filter sits between
`act_inference(...)` and `env.step(...)`. But with no surviving runs, π_nom has
to be trained first — `train_policy.py` does that, and it is the dominant cost.

---

## What was added

| Path | Role |
|---|---|
| `safeloco_eval/` | **Cross-phase invariant module** — copy verbatim to the joint-limit branch (§1.1) |
| `safeloco_eval/metrics.py` | every threshold and derived metric, defined once (§1) |
| `safeloco_eval/stats.py` | Wilson CIs, Spearman, AUROC, bootstrap — pure stdlib (§2) |
| `safeloco_eval/eval_common.py` | the eval loop refactored out of `play.py`, termination definition, per-episode records |
| `safeloco_eval/filters.py` | E1-A sampling projection, E1-B gradient projection (J1's clamp lands here in Phase 2) |
| `safeloco_eval/qsafe.py` | Q̂_safe / V̂_safe, safety-return targets, collision labels |
| `safeloco_eval/seeds.py` + `eval_seeds.json` | the shared 1,000-episode seed list |
| `experiments/e1/train_policy.py` | train π_nom / π_rs / π_ours from scratch with the right config deltas |
| `experiments/e1/collect_qsafe_data.py` | roll out π_nom / π_rs → Q̂ training buffers |
| `experiments/e1/train_qsafe.py` | offline Q̂ fit + the §2 validation battery |
| `experiments/e1/run_e1.py` | one sweep point → per-episode CSV + manifest |
| `experiments/e1/analyze_e1.py` | Table R1a, §5 sanity checks, §3.1 verdict, §7 sentence, headline figure |
| `experiments/e1/sweep_e1.sh` | the whole pipeline |
| `tests/test_e1_offline.py` | offline tests — no GPU, no IsaacGym, no torch needed |

One 4-line change to `legged_gym/envs/base/legged_robot.py::check_termination`:
it now snapshots `term_contact`, `term_proj_grav_z`, `term_base_z_rel`,
`term_base_vel_z`. `reset_idx()` runs immediately after and overwrites
`root_states`, so the fall definition has to read the terminal pose there or
not at all. Behaviour is otherwise unchanged.

---

## Run it

**0. Sanity, before touching the GPU** (takes ~5 s, needs nothing installed):

```bash
python tests/test_e1_offline.py
```

**1. Train π_nom.** With no surviving runs this is step one. It is extra cost
on top of the protocol's 2–3 day Phase 1 estimate, which assumed π_nom already
existed; you know your own wall-clock for 2,000 iterations at 2,048 envs better
than I can guess it.

```bash
python experiments/e1/train_policy.py --policy pi_nom \
    --task go1_amp --headless   # 2000 iterations, the train.py default
# -> logs/go1_viploco_warp_v5/pi_nom/model_*.pt  (+ policy_config.json)
```

`--policy pi_nom` sets `algorithm.safety_coef = 0`, which makes
`AMPPPO.update` skip the entire safety branch (`amp_ppo.py:350`) — no safety
surrogate, no safety-critic update, no damped null-space projection. The reward
is untouched (`safety_value` is not in `rewards.scales`, so
`_reward_safety_value` is never registered). The world model trains inside the
same run and lands in the same checkpoint, so there is no separate world-model
step.

**2,000 iterations, matching the original π_nom.** The comparison E1 makes is
filter-vs-no-filter on the *same* policy, so what matters is that π_nom is
trained the way the reported π_nom was. `train_policy.py` defaults to 2,000, as
`train.py` does.

(The `feet_edge` reward curriculum is irrelevant either way:
`_reward_feet_edge` zeroes everything outside `[gap_start_idx, pit_end_idx]`,
an empty slice under corridor + rough-flat proportions, so the term is
identically zero here — and it is go1's only `reward_curriculum_term`.)

If you ever want to extend a run rather than restart it, `--max_iterations` on
a resume is *additional* iterations, not a target:

```bash
python experiments/e1/train_policy.py --policy pi_nom --task go1_amp \
    --headless --resume --load_run pi_nom --max_iterations 1000
```

**2. Gate on π_nom's quality before spending anything else.** Do not skip this
— every later number is relative to this row.

```bash
python experiments/e1/run_e1.py --task go1_amp --headless \
    --load_run pi_nom --filter none \
    --n_episodes 250 --eval_envs 250 --out_dir logs/e1/precheck
```

This is a characterisation of π_nom, not a pass/fail on training length. Eval
deliberately disables the terrain curriculum and spreads envs over all ten
levels (`terrain_levels = arange(num_envs) % 10`), including levels the training
curriculum never promoted anyone to — a policy too weak for the hard corridors
should be *shown* failing on them. The filtered and unfiltered rows see exactly
the same level distribution, so the E1 comparison stays internally valid however
far training got.

The one thing worth checking before spending GPU-hours is that there is a signal
to measure at all: a **non-trivial collision rate** (near-0% means the cylinders
are never being hit and a collision filter has nothing to improve) and a return
that isn't pinned at zero.

The per-level breakdown is worth having as a reported property:

```python
import csv, collections
rows = list(csv.DictReader(open("logs/e1/precheck/pi_nom_unfiltered.csv")))
by = collections.defaultdict(list)
for r in rows:
    by[int(r["terrain_level"])].append(r)
for lvl in sorted(by):
    g = by[lvl]
    print(lvl, len(g),
          "fall {:.0%}".format(sum(int(r["fell"]) for r in g) / len(g)),
          "coll {:.1%}".format(
              sum(int(r["n_collision_steps"]) for r in g)
              / sum(int(r["n_steps"]) for r in g)))
```

Corridor difficulty is `sinusoid_amplitude = 0.5 + 0.5·difficulty`
(`terrain.py:362`), so the level sets how sharply the corridor snakes. Where the
curve breaks tells you which corridor geometries π_nom cannot handle unaided —
useful context for reading how much of E1's fall rate the filter caused versus
inherited.

**3. Pin it.** This is the §1.1 invariant — one checkpoint, hash recorded, and
the *same* one on the joint-limit branch later. The hash goes into every run
manifest automatically; record it in App. A too.

```bash
export PI_NOM_RUN=pi_nom
export PI_NOM_CKPT=-1          # or a specific iteration number
export PI_RS_RUN=               # leave empty -- see below
```

**On π_rs — your call, see the open questions below.** The protocol wants Q̂
trained on mixed π_nom + π_rs substrates for action coverage, and lists a π_rs
sweep point as optional. It costs a second training run of the same length. If
you skip it, note the deviation in App. J. π_rs is separately required for E2,
where it is the Reward-only row:

```bash
python experiments/e1/train_policy.py --policy pi_rs --rs_weight 1.0 \
    --task go1_amp --headless
```

**π_ours is not needed for E1.** The verdict rule quotes the paper's own
numbers for it (4.4% collision / 33.9 return / 0.085 m/s), so nothing in this
experiment requires retraining the full method. `--policy pi_ours` exists in the
same script for when the Pareto plots need it.

**4. Smoke-test the E1 plumbing** — 5 minutes, and it catches every path,
shape and import problem before you commit 4 hours. Run from the repo root
(the world-model config loader resolves `dreamer/configs.yaml` relative to
`sys.argv[0]`'s grandparent, so the scripts must stay three levels deep and be
invoked from the root):

```bash
python experiments/e1/collect_qsafe_data.py --task go1_amp --headless \
    --load_run "$PI_NOM_RUN" --collect_envs 16 --collect_steps 60 \
    --out logs/e1_smoke/buf
python experiments/e1/train_qsafe.py --buffers logs/e1_smoke/buf \
    --out logs/e1_smoke/qsafe --epochs 2 --ensemble 1
python experiments/e1/run_e1.py --task go1_amp --headless \
    --load_run "$PI_NOM_RUN" --qsafe_ckpt logs/e1_smoke/qsafe.pt \
    --filter A --epsilon 0.1 --n_episodes 16 --eval_envs 16 \
    --out_dir logs/e1_smoke/sweep
python experiments/e1/run_e1.py --task go1_amp --headless \
    --load_run "$PI_NOM_RUN" --qsafe_ckpt logs/e1_smoke/qsafe.pt \
    --filter none --n_episodes 16 --eval_envs 16 --out_dir logs/e1_smoke/sweep
python experiments/e1/analyze_e1.py --dir logs/e1_smoke/sweep \
    --out logs/e1_smoke/R1a
```

The numbers will be meaningless at 16 episodes; what you're checking is that
all five stages complete, the trust-region assertion never fires, and the
unfiltered row reports 0% activation. Then `rm -rf logs/e1_smoke`.

**5. Run the E1 pipeline** (≈ 4–6 h once π_nom exists):

```bash
./experiments/e1/sweep_e1.sh all
```

or stage by stage, which is what I'd actually do:

```bash
./experiments/e1/sweep_e1.sh collect    # ~20-40 min, writes ~2 GB per policy
./experiments/e1/sweep_e1.sh train      # ~10-30 min  <-- READ THE GATE OUTPUT
./experiments/e1/sweep_e1.sh sweep      # ~2-4 h, 10-11 configurations
./experiments/e1/sweep_e1.sh analyze    # seconds
```

Useful overrides: `N_EPISODES` (default 1000), `EVAL_ENVS` (default 250 —
`N_EPISODES` should be a multiple of it), `COLLECT_STEPS`, `EPOCHS`, `DR`
(`off`/`on`), `ALPHA`, `DEVICE`, `OUT`.

### The gate after stage 2

`train_qsafe.py` prints the §2 validation battery and whether the critic is
informative (ρ ≥ 0.6, AUROC ≥ 0.8). **This changes which §7 sentence E1
produces, not whether E1 runs.** If the battery fails, E1's finding is not
"projection fails" but "usable learned joint-space barriers are hard to
obtain" — still supportive, different sentence. `analyze_e1.py` picks the right
one automatically from the manifest, so run the sweep either way.

The battery also checks whether Q̂ beats a trivial distance heuristic
(`min_cbf_h` alone) at the same AUROC task (analysis protocol §5). If it does
not, the honest move is to re-run the sweep with the heuristic as the filter
target — simpler and more defensible. Tell me if that fires and I'll wire it.

### Sweep points that get run

| # | Configuration | Why |
|---|---|---|
| 0 | unfiltered π_nom | the reference every verdict rule is relative to |
| 1–5 | variant A, ε ∈ {0.02, 0.05, 0.1, 0.2, 0.5} rad, τ = −0.25 | the primary sweep |
| 6–7 | variant B (gradient), ε ∈ {0.1, 0.5} | preempts "sampling is a strawman" |
| 8 | variant A always-on, ε = 0.1 | separates trigger timing from correction effects |
| 9 | variant A, ε = 0.1, τ = 0 | the secondary trigger |
| 10 | variant A, ε = 0.1, 3-head min-aggregated Q̂ | the ensemble control |
| 11 | π_rs, variant A, ε = 0.1 | optional, only if `PI_RS_RUN` is set |

Each writes `logs/e1/sweep/<run_id>.csv` (per-episode records) and
`<run_id>.manifest.json` (checkpoint hash, seed-file hash, eval-module hash,
DR setting, δ, τ, thresholds).

---

## Sending me the results

Everything I need is small — the CSVs are ~1000 rows each, and the whole set
zips to a few MB. **Nothing binary, no checkpoints.**

```bash
cd /path/to/Safeloco
zip -r e1_results.zip logs/e1/sweep/*.csv logs/e1/sweep/*.manifest.json \
                      logs/e1/qsafe.report.json
```

Send me `e1_results.zip`. If you'd rather not zip, paste the output of:

```bash
python experiments/e1/analyze_e1.py --dir logs/e1/sweep --out logs/e1/R1a \
    --qsafe_report logs/e1/qsafe.report.json
cat logs/e1/R1a.md
```

— but the raw CSVs are better, because they let me recompute anything (e.g.
re-derive the fall definition, split by terrain level, or check whether a
difference in per-episode means clears its standard error) without another GPU
run. Also send the console output of the `train` stage if anything looked off.

I will then apply the §3.1 verdict rule, report every registered prediction as
held / falsified / not-run, and give you the §7 rebuttal sentence plus the
Table R1a markdown and LaTeX ready to paste.

---

## Settings in force, and why

Where the protocol left room or the code forced a choice. The first four were
your calls; the rest follow from the code. All are recorded in the run manifest.

**Confirmed choices:** π_nom only (no π_rs — note the §2 deviation in App. J);
action-noise injection kept at σ = 0.05 rad on half the collection envs; domain
randomisation **off** at eval (must match on the joint-limit branch); safety-return
**α = 0.95**.

**π_nom's safety critic is untrained, so E1 trains its own trigger.**
`AMPPPO.update` gates the entire safety branch behind `if self.safety_coef > 0`
(`amp_ppo.py:350`), and π_nom is trained with `safety_coef = 0`. The
checkpoint's `safety_critic` head has therefore never received a gradient —
triggering on it would be triggering on noise. `train_qsafe.py` fits a
state-only V̂ head alongside Q̂ on the same targets, and that is the default
trigger (`--trigger_source vhat`). `qpol` and `sv_oracle` exist as controls.

**ε is in radians of joint command.** The protocol's sweep is in rad; this
actor has no tanh (`actor_critic_wmp.py:163`), so the raw action box is
`clip_actions = 6.0` and joint targets are `action_scale = 0.25` × action.
Filters take ε in rad and convert internally. `‖Δa‖` is reported in rad, and
`‖Δa‖∞ ≤ ε` is asserted on every filtered step.

**Trust region is ∞-norm; min-distortion ranking is L2.** The constraint is the
ε-box (that is what ε bounds); the objective that picks among qualifying
samples is `‖a − a_pol‖₂`, which is what "minimum-distortion projection"
means. Both norms are logged.

**a_pol is not among the K candidates.** With δ > 0 it could never clear its own
margin, and including it in the max-Q̂ fallback would turn the fallback into a
no-op. Executing a correction that fails to clear the margin *is* the behaviour
under test — it is what produces the large-ε regime.

**Variant B normalises the gradient by default.** `--grad_normalize inf`
rescales ∇_a Q̂ to unit ∞-norm so each of the m steps moves exactly η = ε/m.
A raw gradient whose magnitude happens to be small would make B a no-op, and B
exists precisely to rule out "sampling is a strawman". `--grad_normalize none`
gives the literal form.

**Injected action noise during Q̂ data collection** (`--action_noise 0.20` raw
≈ 0.05 rad, on half the envs) — an addition beyond the protocol, kept by your
call. Mixed substrates broaden coverage over *states*; the E1 filter evaluates
Q̂ at actions drawn uniformly from an ε-box around a_pol, which no rollout
policy ever takes. Without off-policy action coverage, Q̂'s dependence on `a` is
unconstrained by data and the sweep measures extrapolation rather than
projection. `--noise_frac 0` reproduces the literal protocol. Since π_rs is not
being trained, this is the *only* source of action-space breadth in Q̂'s
training set — worth one sentence in App. J.

**Episode identity across methods.** IsaacGym has no per-episode seed, so
`eval_seeds.json` is consumed as *chunk* seeds: re-seed globally, full
`env.reset()`, then record **only the first episode each env completes**. Every
method sees the same 1,000 initial conditions, and no drift from earlier
filtered steps contaminates later episodes. Terrain is generated once at env
creation under a fixed `TERRAIN_SEED = 1`, so the corridors themselves are
identical across all rows.

**DR off at eval** (§1.1 invariant #4 — decided once, recorded, and it must
match on the joint-limit branch). Ranges are collapsed rather than flags flipped: several `randomize_*` booleans gate blocks
of `privileged_obs_buf` and turning them off would change the observation width
and silently break the checkpoint.

**Fall definition** (one definition, both branches): terminal base tilt past
60° (`projected_gravity_z > −0.5`) **or** base height below 0.15 m above its
terrain cell, **or** the simulator's own hard flag. The repo's own `self.fall`
alone (z-velocity < −3 m/s or fully inverted) is too strict to catch a stumble
that ends in a belly-flop. All four raw terminal quantities are in the CSV, so
the definition is recomputable post hoc without re-running.

**Timing convention.** Per-step *state* quantities (joint margins, base
velocities, yaw rate, contact phase, jerk) come from the state the policy acted
on; per-transition quantities (reward, `min_cbf_h`) come from immediately after
the step, where the env computes them before `reset_idx`. No post-reset value
enters any metric. Episodes end at their fall, so no post-fall timesteps enter
any denominator either.

---

## Things I could not do here, flagged rather than hidden

1. **Validation battery item (iv), the branched-rollout spot check**, is not
   implemented. It needs the simulator state-reset harness that E6 builds. The
   Q̂ report lists it under `not_run` rather than leaving it silently absent.
2. **I could not execute any of this.** This container has no GPU, no
   IsaacGym, and no torch. The statistics, metric definitions, verdict rule and
   full analysis pipeline are covered by `tests/test_e1_offline.py`, which does
   run and passes; the simulator-facing code is written against the actual call
   signatures in `play.py`, `wmp_runner.py` and `legged_robot.py` but has not
   been executed. Expect to shake out an import or a shape on the first
   `collect` run — send me the traceback and it'll be a quick fix.
3. **The `alpha` reconciliation is still open in the code.**
   `rollout_storage.py:176` runs 0.7, its own docstring says 0.95/0.05, and
   Table 4 says 0.80 (`viploco_codebase_addendum.md` §2). E1 uses **0.95** for
   Q̂'s targets, recorded in `qsafe.report.json` — but the training path still
   runs 0.7, so the two are not yet consistent. Changing your mind later is
   cheap: re-run stage 2 with `ALPHA=<x>`; stage 1's buffers are unaffected.
4. **`cone_constraint.py:63-64` still hardcodes `d_safe = -0.4`,
   `d_danger = -0.6`**, discarding the passed arguments. That is the addendum's
   highest-priority §2 item. It does not affect E1 (which never calls the
   projection), but it does affect whether Table 5's ablation rows differ by
   anything but seed noise. Out of scope here; flagging it because you'll want
   it fixed before E2.
5. **Only E1.** No E2/E3/E4/E5′/E6, no Phase 2. The `safeloco_eval` package is
   laid out so J1's clamp is a new `ActionFilter` and nothing else changes.
