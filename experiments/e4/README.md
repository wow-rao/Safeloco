# E4 — Command-space filtering, the steel-man

`analysis_protocol.md` §3.4. Evaluation-only (§2): one policy, 1,000 episodes
on the fixed seed list, Wilson CIs, sweeping the CBF gain α.

**This is the experiment that should work.** It exists to give the strongest
form of the baseline its best shot, not to knock it down — and §3.4 pre-commits
to acting on the answer if it does work.

---

## Why command space is a different problem from E1

Both experiments filter against the same barrier:

```
h_i = ‖o_i − p‖ − 2 r_i − 0.35          (legged_robot.py:1397)
```

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

### What is deliberately not filtered

**Yaw rate.** Turning does not move the base at first order, so `ω_z` has
relative degree two with respect to `h` and cannot appear in a first-order CBF
constraint. Filtering it would be unsound, not conservative. Recorded in the
manifest as `yaw_filtered: false`.

### α

α is the class-K gain. **Small α is more conservative** — it permits only slow
approach — and α → ∞ recovers the unfiltered policy. Sweeping it traces the
frontier. The default grid is `0.25 0.5 1.0 2.0 5.0`.

### Multiple obstacles

Each obstacle contributes one half-plane. The filter cyclically projects onto
the most-violated constraint until all are satisfied or `--max_passes` is
exhausted; a residual violation is recorded as `infeasible_steps` (§1's
Infeasibility %, with activation as the denominator). A QP solver would give
the same answer for two variables and is not worth the dependency.

### Command clamping

The filtered command is clamped to the range the policy was trained on. An
off-distribution command would be tracked badly and would flatter the filter —
it would look safe while the robot simply failed to follow it. The eval config
pins lateral velocity to `[0, 0]`, which would forbid the sidestepping that is
the whole reason command space can work, so the lateral clamp is widened to the
training range. `--no_clamp_commands` disables clamping entirely.

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
- **Watch lateral velocity.** If command-space filtering works, it should work
  by sidestepping, and `|lat vel|` should climb toward our method's 0.085.
  Safety improving *without* lateral velocity moving would be worth
  investigating before believing it.
