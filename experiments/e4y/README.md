# E4-Y — command filtering *with steering authority*

## Why this exists

The completed command-space study (E4) filtered forward speed only. That was
not an arbitrary restriction: with the barrier written on the base centre,

```
h    = ||o − p|| − r − margin
ḣ    = −⟨v_world , d⟩ ,        d = (o − p)/||o − p||
```

yaw does not appear at all — turning in place does not move the base at first
order. So the filter's input space was one-dimensional and its only answer to
"too close" was "stop". Two consequences followed directly from that, and both
were reported as findings when they were partly artifacts:

* it had **no legal option on 79–100%** of the steps it engaged, and
* the robot **parked rather than steered**.

Neither OCR nor ABS works that way. Both filter a *twist* — OCR on a Dubins
model, ABS on planar centroidal velocity — and both steer. Until yaw is in the
filtered set, that row has not been fairly tested, and a rebuttal built on it
is answerable in one sentence.

E4-Y fixes the geometry rather than the conclusion.

## The fix: a look-ahead point

Write the barrier not on the base but on a point a fixed distance `L` ahead of
it:

```
P     = p + L·e(θ),          e(θ) = (cos θ, sin θ)
Ṗ     = v·e + L·ω·e⊥
ḣ     = −⟨Ṗ, d⟩ = −v⟨e,d⟩ − L·ω⟨e⊥,d⟩
```

Now **both** `v` and `ω` enter at first order, and the CBF condition
`ḣ ≥ −α·h` is a single linear inequality in the twist `u = (v, ω)`:

```
a·u ≤ b,      a = (⟨e,d⟩, L⟨e⊥,d⟩),      b = α·h
```

The feasible set is a half-plane intersected with the command box, so the
projection is closed-form — no QP solver, no learned critic.

This is the standard near-identity-diffeomorphism trick, not a novelty of
ours. It is what makes the comparison fair.

### Yaw is deliberately the cheaper correction

The projection is weighted, `min ‖u − u_cmd‖²_W` with

```
W = diag(w_v, w_ω),     w_v = 4.0,   w_ω = 1.0
```

so that when both steering and braking would satisfy the constraint, the
filter **prefers to steer**. That is a deliberate choice and the entire point
of the variant — a steel-man, not a neutral setting. It is stated here so it
does not have to be discovered in the source.

### The control arm is the same code

`--no_yaw` sets `omega_range = (0, 0)`. Everything else — barrier, weights,
look-ahead, projection loop, clamping, logging — is byte-identical. The
yaw-disabled arm therefore *is* the braking-only filter, re-measured inside
this harness, so the yaw/no-yaw comparison is within-study rather than across
studies with different seeds and different code.

## Run it

```bash
# 0. Strongly recommended first: same-harness reference rows.
#    Without logs/reference/reference.json every "dominated by the proposed
#    method" call is made against Table 2's 4.4%, which was measured with a
#    different collision detector.  The analyser still runs, and shouts.
./experiments/reference/sweep_reference.sh all

# 1. Calibration — a GATE, and a result in its own right.  ~15 min.
./experiments/e4y/sweep_e4y.sh calibrate

# 2. Sweep: unfiltered + α ∈ {1, 2, 5} × {yaw, no-yaw}, N = 1000.  ~4-6 h.
./experiments/e4y/sweep_e4y.sh sweep

# 3. Analysis + packaging.
./experiments/e4y/sweep_e4y.sh analyze
./experiments/e4y/sweep_e4y.sh package
```

Environment overrides: `TASK`, `DEVICE`, `PI_NOM_RUN`, `PI_NOM_CKPT`,
`ALPHAS`, `N_EPISODES`, `EVAL_ENVS`, `LOOKAHEAD`, `W_V`, `W_OMEGA`,
`BARRIER`, `OMEGA_LIMIT`.

## Step 1 is a gate — read its verdict before believing anything downstream

A command filter can only be as good as the policy's ability to track the
commands it issues. `calibrate_commands.py` measures that, open-loop, on flat
ground with no obstacles, and writes `logs/e4y/calibration.json`:

| Output | What it decides |
|---|---|
| trained `commands.ranges` | whether an axis was ever trained at all. E4 found `lin_vel_y` pinned to `[0, 0]` — a lateral command there was fiction. |
| yaw tracking gain (realised / commanded) | whether the filter's `ω` is a real instruction or one issued into the void |
| max realised yaw rate | the saturation point |
| min turn radius at 0.6 m/s vs corridor width | whether steering-based evasion is **geometrically possible here**, known in advance |

**The branch, decided by the gain:**

* **Gain near 1 up to a clear saturation point** → E4-Y is a fair test of the
  OCR/ABS *mechanism*, and the result is strong whichever way it lands.
* **Degenerate gain** → E4-Y is a fair test of *retrofitting onto this
  policy*, and a weak test of the mechanism in its native setting. That has to
  be said in the writeup, in those words. The fair version of the mechanism
  test needs `π_nom` retrained with wider command ranges, which is expensive
  and is future work.

The calibration is publishable on its own either way: it turns "the retrofit
has nowhere to steer" from an assertion into a measured property of the
command interface.

## Registered predictions

Registered before the sweep runs, per §1.3 of the completion plan.

| | Prediction |
|---|---|
| (i) | No-legal-option rate drops sharply from 79–100% once yaw enters the input space — the earlier rate was an artifact of the 1-D input |
| (ii) | Contact falls below the braking-only filter's 12.1% **at matched speed** |
| (iii) | It still does not reach the joint frontier: either contact stays above the same-harness reference, or realised speed and distance fall materially short |
| (iv) | The steer/brake decomposition shows a substantial share of correction going to yaw — i.e. the filter actually uses the new axis |

**Falsification:** some α reaches contact ≤ the reference at comparable speed,
distance and return. If that happens it gets reported plainly — see the gate.

Prediction (i) already has a unit-test demonstration independent of the sweep:
in `tests/test_e4y_offline.py`, a geometry exists where the yaw-disabled arm is
infeasible and the steering arm is not, using the same filter object.

### Prediction (i) needs a caveat, because the 79–100% baseline is wrong

The braking-only study measured its no-legal-option rate with a solver that
projected onto the half-plane and clamped to the command box **afterwards**.
With lateral velocity pinned to zero, the clamp deleted the `v_y` part of
every correction and left `v_x` above the constraint — so the residual came
back positive and the step was logged as having no legal option, when simply
slowing to `b/a_x` was legal. On a 64-point sweep of triggered geometries it
did that on 92% of cases where a legal command existed.

So some of the 79–100% belongs to the solver rather than to the
one-dimensional input space, and prediction (i) cannot be scored against that
figure. What it must be scored against is the **yaw-disabled arm of this
sweep**, which runs the corrected solver — a within-study, within-harness
contrast where the only difference is whether `ω` is in the input.

That is the honest version of the test, and it is also the harder one.

### The optional third arm

`REPLICA_ALPHAS="2.0" ./experiments/e4y/sweep_e4y.sh sweep` adds
`--no_yaw --lookahead 0` rows. A look-ahead of zero puts the barrier back on
the base centre, where the yaw term vanishes identically — so those rows *are*
the braking-only filter of the completed study, re-run under the corrected
projection. Use them if you want the corrected number to be like-for-like with
what was already reported. They are off by default because they are beyond the
registered α × {yaw, no-yaw} design, and the analyser keeps them out of the
yaw/no-yaw contrast (different barrier, so the difference would not be "yaw").

The same defect also made the old filter execute a **less** conservative
command than the constraint allowed, so the braking-only steel-man was weaker
than intended. That cuts against the earlier "command filtering is dominated"
conclusion rather than for it, which is the direction that matters.

## The four-way gate

`analyze_e4y.py` classifies every row against the reference on the **joint**
frontier — contact, return, realised speed, distance — never on contact alone.

| Outcome | What it means for the rebuttal |
|---|---|
| `dominated` | The OCR/ABS row closes empirically. Command filtering *with genuine steering authority* was tested and does not reach our operating point. |
| `competitive` | Better contact, worse speed/return, or very conservative. Include with Pareto framing plus the structural points: permanent runtime loop, cannot express joint-level constraints. |
| `matches_or_dominates` | Command filtering suffices for pure collision avoidance in this corridor — which is the setting these methods were designed for. The contribution then rests on joint-limit constraints (which command filters cannot express **at all** — categorical, not empirical), zero deployment-time components, and first-order task neutrality. J1/J2 become mandatory. |
| `not_decidable` | Every row is a `trivial_stop`. Nothing about safety was learned; a robot that is barely moving does not sit on a frontier. |

`trivial_stop` exists because the first real E4 sweep produced contact 0.0%,
return *up* and velocity error *down* — from a filter that had ratcheted the
command to zero. Any row whose realised forward speed has collapsed below half
the unfiltered policy's is disqualified before it can be called dominating.

## What gets logged that E4 did not have

* **`sum_dnorm_v` / `sum_dnorm_omega`** per episode → the steer/brake
  decomposition, which is the mechanism evidence for prediction (iv) and the
  command-space analogue of E1's gait-phase figure.
* **Realised forward speed, distance travelled, contact per metre** on every
  row. E4 established that any command-modifying intervention can buy a lower
  contact rate with slowness; without these columns the table is unreadable.
  Contact-per-metre is what showed a permissive filter to be genuinely *worse*
  than no filter rather than merely slower.
* **Episode-clustered bootstrap intervals** on the pooled-timestep contact
  rate. Timesteps within an episode are not independent trials — on this data
  the clustered interval is about **10× wider** than the Wilson interval, so
  the Wilson version was overstating precision by an order of magnitude.
* **Feasibility measured after clamping**, so `infeasible_steps` can never
  exceed `trigger_steps`. (It did once, which is what produced a bare
  `math domain error` inside `sqrt`.)

## Things that will bite, and are guarded

| Trap | Guard |
|---|---|
| Projecting onto the half-plane and clamping to the box afterwards — the clamp pushes the result back out of the half-plane, so a legal command is reported as "no legal option" *and* the executed command under-brakes | `project_halfplane_box` solves each constraint-plus-box subproblem exactly. See below — this one had already corrupted a published number |
| The filter reads back the command it wrote last step and filters its own output — a ratchet that parks the robot | `EvalCollector` snapshots the nominal command per chunk; the filter only ever sees nominal input |
| Velocity error measured against the rewritten command, flattering the filter | `vel_err` is measured against the **nominal** command |
| `sv` barrier (`2r + 0.35`) demands 0.95 m clearance in a corridor whose minimum is 0.8 m — empty safe set, nothing is satisfiable | default barrier is `geometric` (0.53 m), matching what the collision detector actually measures; `sv` is retained but its required clearance is written into every manifest |
| Command ranges read *after* the eval overrides collapse them to `[0.6, 0.6]`, leaving the filter unable even to brake | trained ranges are read from `task_registry` **before** the env is built |
| Yaw commanded outside the trained range is not tracked | defaults to the trained range; `--omega_limit` widens it only on purpose, and the run prints a warning when the trained span is degenerate |

## Files

| File | |
|---|---|
| `calibrate_commands.py` | open-loop command-response sweep; the gate |
| `run_e4y.py` | one row: unfiltered, or α × {yaw, no-yaw} |
| `analyze_e4y.py` | Table R1d, registered predictions, four-way gate |
| `sweep_e4y.sh` | `calibrate` / `sweep` / `analyze` / `package` |
| `../../safeloco_eval/command_filters.py` | `UnicycleCBFCommandFilter` |
| `../../tests/test_e4y_offline.py` | offline tests, no GPU needed |
