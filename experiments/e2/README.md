# E2 — Training-time filtering (the CBF-RL mechanism), collision

`analysis_protocol.md` §3.2. Where E1 asked whether a projection filter works
*at deployment*, E2 asks whether the same filter, applied *during training*,
teaches the policy to be safe on its own — the claim CBF-RL rests on.

| file | what it does |
|---|---|
| `train_e2.py` | trains one variant × seed; optionally filters actions during rollout |
| `run_e2.py` | evaluates one checkpoint, with and without the test filter |
| `analyze_e2.py` | §3.2 verdict, Table R2a, §7 sentence |
| `sweep_e2.sh` | calibrate → train → eval → analyze |
| `tests/test_e2_offline.py` | the verdict rule on synthetic worlds, no GPU needed |

---

## The three variants

| variant | reward shaping | filter during training | protocol row |
|---|---|---|---|
| `reward_only` | yes (`rewards.scales.safety_value = w`) | no | "Reward-only" |
| `filter_only` | no | yes | the CBF-RL nav-ablation |
| `dual` | yes | yes | "Dual" |

Every checkpoint is then evaluated **twice** — with the test filter attached and
without — because §3.2's verdict is entirely about the difference between those
two rows.

## The verdict rule (§3.2, applied mechanically)

- **Internalization failure** — `filter_only`'s collision rate *without* the
  test filter is materially worse than *with* it, seed-level intervals not
  overlapping. The policy only ever looked safe because the filter was there.
- **Reduces to shaping** — `dual` *without* the filter lands in the
  reward-shaping band (~12–14%) rather than near 4.4%. The filter bought
  nothing that reward shaping did not.
- **Either supports C2a. Both is stronger. Neither triggers the §7 reframing.**

## Four decisions this experiment had to make, and why

### 1. Collision is the geometric test, not `min_cbf_h < −0.05`

§1 defines collision by the margin threshold. E1 deviated to `play_plan.py`'s
geometric link-vs-cylinder test, and E2 follows E1. The E1 runs let us check
which convention the paper's reference numbers use, because both columns were
logged on the same episodes:

| convention | unfiltered π_nom |
|---|---|
| geometric link-vs-cylinder | **11.83%** |
| `min_cbf_h < −0.05` | 49.25% |

§3.2's verdict compares against a reward-shaping band of ~12–14% and a 4.4%
reference. Only the geometric number is commensurate with those; under the
margin threshold every policy in the study would read ~50% and the band test
would be meaningless. The margin threshold is retained as the secondary
`n_proximity_steps` column. **This is a deviation from §1 as written and belongs
in App. J.**

### 2. The PPO update sees the *sampled* action, not the executed one

The rollout executes the filtered action; the storage keeps the action the
policy sampled. So the update is credited with returns produced by actions it
did not take. That mismatch is the mechanism under test — it is why a filtered
policy can fail to internalize anything — not an implementation slip.

`--store_executed` flips it, updating the policy toward the filtered action
instead. That is the contrast run: if internalization failure disappears under
`--store_executed`, the failure is attributable to the PPO channel specifically
rather than to filtering per se.

### 3. `KL(executed‖sampled)` is operationalised as a recentred Gaussian KL

§3.2 names the diagnostic but does not define it. An executed action is a
point, not a distribution, so we use the KL between the policy's own Gaussian
recentred at the executed action and at the sampled one. With a shared σ the
covariance terms cancel and this is

    KL = 0.5 * Σ_j ((a_exec,j − a_samp,j) / σ_j)²

i.e. the squared action displacement measured in units of the policy's own
exploration noise. It is zero when the filter does not engage and grows with
how far outside its own noise the policy is being pushed. Reported per
iteration in `train_diag.csv`.

### 4. ε is chosen by calibration, not by hand

The training-time trust region is not registered anywhere. §2 scopes sweeps to
"ε in E1/J1" and puts E2 under trained studies, so a full training-time sweep is
outside the registered design and would take the matrix from 9 runs to 21.

Instead `sweep_e2.sh calibrate` runs a short pass (default 150 iterations) at
each candidate ε and reports the three numbers that decide usability:

- **fraction of steps filtered** — near zero means `filter_only` is just π_nom
  and the experiment is vacuous;
- **mean KL(executed‖sampled)** — how far outside its own noise the policy is
  being pushed;
- **return trend across the window** — falling return means the filter is
  destabilising training rather than shaping it.

Pick the largest ε that binds without a downward return trend, record the
calibration table alongside the result, and the choice is a measurement rather
than a preference. E1's deployment sweep is prior evidence — inert at ε ≤ 0.05,
gait intact through ε = 0.2, destructive at 0.5 — but it was measured on π_nom
without training adaptation, which is exactly why it is not simply reused.

---

## Running it

```bash
# 0. gate: pick epsilon from measurements (4 short runs)
./experiments/e2/sweep_e2.sh calibrate
#    read logs/<experiment>/e2_filter_only_s1/calibration.json

# 1. the matrix: 3 variants x 3 seeds
EPS=0.2 ./experiments/e2/sweep_e2.sh train

# 2. each checkpoint twice, with and without the test filter (18 evals)
EPS=0.2 ./experiments/e2/sweep_e2.sh eval

# 3. Table R2a, verdict, §7 sentence
./experiments/e2/sweep_e2.sh analyze
```

Offline, before any of that:

```bash
python tests/test_e2_offline.py      # verdict rule on synthetic worlds
python tests/test_cone_constraint.py # the E2 prerequisite fix
```

## What to check before believing the result

- **`cone_constraint.py` must be fixed first.** Until the `d_safe`/`d_danger`
  override was removed, every run used the hardcoded −0.4/−0.6 whatever the
  config said. `pi_ours` rows and any threshold ablation predate the fix.
- **Per-seed values before averaging** (§5). A bimodal outcome — one seed
  collapses, two succeed — is a finding, not something to average away.
  `analyze_e2.py` prints the per-seed column for exactly this.
- **Seed intervals at n = 3 are wide.** A 95% t-interval is mean ± 4.30·SEM.
  `--interval std` reports mean ± 1 std instead, which is what §2 asks to
  *report*; the verdict defaults to the conservative `t95`. Whichever is used
  is recorded in the output JSON. If the verdict comes out N/A because
  intervals overlap, that is a real inconclusive result and the honest fix is
  more seeds, not a narrower interval.
- **The paired per-seed gap** is printed as a supporting diagnostic and marked
  unregistered. It is much tighter than the registered overlap test, because
  with/without share a checkpoint and a seed list — but it is not the rule
  that was fixed in advance, so it never decides the verdict.

## Known limitation, and why it belongs in the write-up

Q̂ was fit on π_nom rollouts and is frozen during E2. As training moves the
policy the critic drifts off-distribution, so the filter's corrections degrade
over the run. That is not a bug in the harness — it is what any learned-barrier
CBF-RL transplant faces, and it is worth stating plainly rather than
engineering around, since the alternative (refitting Q̂ online) is a different
method with its own deployment cost.
