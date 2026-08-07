# E4 — Command-space filtering, the steel-man

`analysis_protocol.md` §3.4. Evaluation-only (§2): one policy, 1,000 episodes
on the fixed seed list, Wilson CIs, sweeping the CBF gain α.

**This is the experiment that should work.** It exists to give the strongest
form of the baseline its best shot, not to knock it down — and §3.4 pre-commits
to acting on the answer if it does work.

---

## Why command space is a different problem from E1

What differs is *what the filter is allowed to move*.

| | E1 | E4 |
|---|---|---|
| filter acts on | joint position targets | commanded base velocity |
| relative degree of `h` | high, contact-gated | **one** |
| barrier | learned Q̂ | analytic, closed form |
| constraint | 64-sample search in an ε-box | one linear inequality per obstacle |

Because `ḣᵢ = −⟨v_world, dᵢ⟩` and the command *is* `v_world` to first order,
the CBF condition `ḣ ≥ −α·h` becomes linear in the command:

```
⟨R(ψ) u, dᵢ⟩ ≤ α hᵢ        ⟹        aᵢ · u ≤ bᵢ,   u = (v_x, v_y)
```

with a closed-form projection onto each half-plane and no learned critic
anywhere. That is why E1's failure says nothing about E4, and why E4 is the
honest steel-man rather than a straw one.

### Which barrier — this decides the experiment

`legged_robot.py:1397` computes `h = ‖o − p‖₃ − 2·r − 0.35`. That expression
is a **reward-shaping** term, not a deployment constraint, and using it as a
hard CBF asks the base to stay **0.95 m** from every obstacle centre. App. G
puts the corridor's minimum lateral clearance at **0.8 m**, so its safe set is
*empty* at the tightest points: no command satisfies the constraint, and a
forward-only robot has nothing to do but stop. The first E4 sweep did exactly
that — infeasibility 91–100%, forward speed 0.017–0.067 m/s against 0.591
unfiltered.

So E4 defaults to a barrier that matches what the collision metric actually
measures — contact when a link comes within `r_obs + 0.03` of the cylinder
axis in XY:

| preset | `h` | clearance demanded (r = 0.3) | satisfiable in an 0.8 m corridor |
|---|---|---|---|
| `geometric` *(default)* | `dist_xy − (r + 0.03 + body_extent)` | 0.53 m | yes |
| `sv` | `dist₃ − 2r − 0.35` | 0.95 m | **no** |

`body_extent` (default 0.20 m) accounts for links reaching beyond the base the
barrier is written on; set it with `--body_extent` if you have a better figure
for the Go1's outer link envelope. Distance is XY because the obstacles are
vertical cylinders.

Filtering against `sv` is still available with `--barrier sv`, and is worth one
run as the contrast: it is what a practitioner would get by reusing the
codebase's existing safety expression, and the answer is that the robot stops.

### What is deliberately not filtered

**Yaw rate.** Turning does not move the base at first order, so `ω_z` has
relative degree two with respect to `h` and cannot appear in a first-order CBF
constraint. Filtering it would be unsound, not conservative. Recorded in the
manifest as `yaw_filtered: false`.

> **Correction.** That paragraph is true of a barrier written on the base
> centre, and false as a general statement — which is how it was read, and it
> made this study unfair to the methods it was meant to steel-man. Put the
> barrier on a **look-ahead point** `P = p + L·e(θ)` and `Ṗ = v·e + L·ω·e⊥`,
> so `ω` enters at first order and the filter can genuinely steer. That is the
> standard near-identity-diffeomorphism construction, and it is what OCR and
> ABS actually do. See `experiments/e4y/`, which supersedes this study; the
> filter here is its `--no_yaw --lookahead 0` arm.

### Two corrections to the projection, and what they change

Both were found while building E4-Y, and both affect numbers already reported
here, so they are recorded rather than quietly fixed.

**1. The command box belongs inside the projection loop, not after it.**
The filter is solving for a point in (half-planes) ∩ (box). This code
projected onto the half-plane and clamped to the box afterwards. Whenever the
obstacle was off the heading — and lateral velocity is pinned to zero here, so
the correction always had a `v_y` component to lose — the clamp deleted that
component and left `v_x` **above** what the constraint allowed. Two
consequences, in opposite directions:

* the residual was then positive, so the step was reported **"no legal
  option"** even though slowing to `b/a_x` was legal all along. The 91–100%
  infeasibility rate quoted below is inflated by this, and is a property of
  the solver as much as of the one-dimensional input space.
* the executed command was **less conservative** than the constraint required
  — the filter under-braked — so the steel-man was weaker than intended, which
  cuts *against* this study's "dominated" conclusion rather than for it.

On a 64-point sweep of triggered geometries (bearing 0–85°, range 0.6–1.0 m),
the old ordering reported no-legal-option on 92% of cases where a legal
command existed, and commanded a higher-than-legal speed on the same 92%;
worst case, an obstacle 65° off the heading at 0.60 m got `v_x = 0.522` where
`0.166` was the limit. Head-on approaches are unaffected, so the true effect
on the run depends on the corridor's bearing distribution and needs a re-run
to pin down.

**2. Alternating projections converge too slowly to fix it by clamping in the
loop.** The convergence rate is set by the angle between the half-plane and
the box, and the near-tangent cases are exactly the ones that dominate the
no-legal-option statistic; a dozen passes still reports feasible problems as
infeasible. `project_halfplane_box` therefore solves each constraint-plus-box
subproblem **exactly** — in two variables the emptiness test is a corner
evaluation and the projection is a three-candidate active-set enumeration.
Verified against brute force on 4,000 random instances: 0 feasibility errors,
0 suboptimal projections.

**What this means for the numbers below.** The gate outcome here rested on
contact, return, speed and distance, not on the infeasibility rate, so the
qualitative reading is not overturned by the fix. But the infeasibility rate
*was* the mechanism story, and it needs re-measuring. Rather than re-running
this study, run E4-Y: its `--no_yaw` arm is a properly-solved braking-only
filter in the same harness, and `REPLICA_ALPHAS=... --lookahead 0` reproduces
this exact filter if you want the like-for-like row.

### α

α is the class-K gain. **Small α is more conservative** — it permits only slow
approach — and α → ∞ recovers the unfiltered policy. Sweeping it traces the
frontier. The default grid is `0.25 0.5 1.0 2.0 5.0`.

### Multiple obstacles

Each obstacle contributes one half-plane. The filter cyclically projects onto
the most-violated constraint **and the box together** until all are satisfied
or `--max_passes` is exhausted. A step is `infeasible` when any single
constraint is unsatisfiable inside the box (a closed-form corner test, so this
is definitive) or when a residual violation survives the passes (the case
where each constraint is individually satisfiable but not jointly). §1's
Infeasibility %, with triggered steps as the denominator.

A general QP solver would give the same answer for two variables and is not
worth the dependency — but see the correction above: the per-constraint
subproblem has to be solved *exactly*, not approximated by projecting and
clamping.

### Command clamping, and what authority the filter actually has

The filtered command is clamped to the range the policy **was trained on**,
read before `apply_eval_overrides` collapses those ranges to the fixed eval
command. Clamping to the collapsed range would pin `v_x` back to 0.6 every
step and the filter could not even brake.

This matters more than it sounds, because π_nom's trained ranges are:

```
lin_vel_x   = [0.0, 0.8]
lin_vel_y   = [-0.0, 0.0]      <-- never trained to sidestep
ang_vel_yaw = [-0.01, 0.01]    <-- effectively straight-ahead only
```

**So the filter's only real authority is braking.** It can slow the robot from
0.8 m/s to a stop; it cannot steer around anything. Widening the lateral clamp
would let it *command* a sidestep, but the policy was never trained to track
one, so the commanded velocity would not be the realised velocity — and that
premise is exactly what gives the barrier relative degree one. A filter whose
commands the robot ignores is unsound, not conservative, and would look safe on
paper while nothing changed on the ground.

This is reported rather than engineered around. It does narrow what E4 can
show: it is a fair test of *command-space CBF on this policy*, and a weaker
test of command-space CBF in general than it would be against a policy trained
with lateral and yaw authority. `--no_clamp_commands` removes the clamp for a
contrast run, and `lateral_authority` is recorded in every manifest.

---

## The gate (§3.4)

Each α row is compared as a tuple against our Pareto point
**(4.4% collision, 33.9 return, 0.085 vel err, 0.085 mean |lateral vel|)**:

- **Outcome 1** — every row dominated by ours → include; completes the taxonomy.
- **Outcome 2** — competitive collision but worse on return/vel-err, or high
  activation indicating conservatism → include with Pareto framing.
- **Outcome 3** — some row matches or dominates ours → **pull J1 forward and
  treat J2 as rebuttal-critical.** §3.4 is explicit that this is decided from
  the numbers the day E4 finishes, not later.

### On "all four"

Dominance is scored on the three axes that have a direction: collision ↓,
return ↑, velocity error ↓. **Mean |lateral velocity| is reported but not
scored.** It has no monotone sense here — our own method sits *above* the
unfiltered baseline on it (0.085 against ≈0.049), because sidestepping is how
it buys clearance. Scoring it as "lower is better" would count our own
signature against us; scoring it as "higher is better" would reward any filter
that merely wobbles. This is a deviation from a literal reading of "all four"
and belongs in App. J.

---

## Running it

```bash
./experiments/e4/sweep_e4.sh sweep      # unfiltered row + 5 alphas
./experiments/e4/sweep_e4.sh analyze    # Table R1c, gate, §7 sentence
./experiments/e4/sweep_e4.sh package    # tarball for analysis
```

Offline first, no GPU needed:

```bash
python tests/test_e4_offline.py
```

## What to check before believing the result

- **The unfiltered row must reproduce E1's** (11.83% collision, 29.94 return).
  Same policy, same seeds, same eval module — if it drifts, something in the
  command hook is altering the unfiltered path, and `activation %` on that row
  must be exactly 0.
- **Activation must rise as α tightens.** The analyser checks this; a
  non-monotone activation curve means the constraint is not binding the way the
  gain says it should.
- **Infeasibility should be rare.** A high rate means the robot is routinely
  boxed in between obstacles with no admissible command, which is a statement
  about the corridor geometry rather than about the filter.
- **Expect braking, not sidestepping.** With `lin_vel_y` trained to `[0, 0]`,
  safety here can only come from slowing down, so `|lat vel|` should stay near
  π_nom's ≈0.049 and velocity error should rise as α tightens. Lateral velocity
  climbing toward our method's 0.085 would mean the robot is being pushed
  sideways by something other than the command, and is worth investigating
  before believing any safety improvement that accompanies it.
