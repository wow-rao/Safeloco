# ETRAP — the step trap: body-level avoidance vs. gradient-projection leg-lift

## The claim this experiment turns into numbers

E4/E4-Y steel-manned **body-level command filtering** (ABS/OCR-style: obstacles
become constraints on the commanded base velocity) against the paper's
**training-time gradient projection**. Those studies compare operating points
on a corridor. This one makes a *structural* difference measurable:

> A body-level controller reasons about obstacles as regions of the **plane** —
> every obstacle is a 2-D no-go footprint, whatever its height. A low block in
> front of the robot is therefore something to drive *around*. The
> gradient-projection method's safety margin ℓ can be **three-dimensional**:
> the space above a low sill is inside the safe set, so the trained policy can
> lift its legs higher than usual and step **over**.

The terrain is built so the difference is not a trade-off but a binary:

```
              tall 0.4 m
           +--------------+
           |              |
 tall 0.4m |   x  spawn   | tall 0.4m        --> +x  (the commanded direction)
           |              |
           +-----====-----+
                 GATE: a 0.08–0.16 m sill (difficulty), full width
```

The robot spawns **inside** a closed pen. Three walls are 0.4 m — above any
step-over and level with the base, so nothing crosses them under any
controller. The fourth is a sill low enough to step across and too long to
walk around: its footprint meets the tall side walls at both ends. Projected
to the plane, the pen is a **closed ring**: the 2-D free space inside is
disconnected from the outside. So a body-level controller — braking-only or
with full steering authority, at any gain — has no safe path out. That is a
property of the world, not of any particular filter, and
`tests/test_etrap_offline.py` proves it as code: the ring is closed even for a
zero-radius body, and unicycle rollouts under the E4-Y filter escape from no
heading.

The volumetric ℓ (below) keeps the space above the sill safe, so the
gradient-projection policy may cross — and the sill heights are chosen so it
must *earn* the crossing: 0.08 m at difficulty 0 is near the Go1's natural
swing apex (~0.05–0.09 m); 0.16 m at difficulty 1 is roughly twice it.

## The custom l-value (the paper's Eq. 16/17 gets a third sibling)

The codebase's existing collision margin (`_compute_safety_value`) is the
paper's Eq. 17: ℓ_col = h + α_h·ḣ with h a **planar/radial** distance to an
obstacle point. That failure set has no notion of "over" — reused on the pen
walls it would forbid the crossing exactly as the command filter does.

Step-trap cells therefore publish their walls as axis-aligned **volumes**, and
`legged_gym/utils/box_sdf.py` scores them volumetrically:

```
ℓ_step = min over keypoints k, wall volumes B of [ SDF_B(p_k) − margin_k ]
sv     = ℓ_step + α_h·ℓ̇_step          (α_h = 0.3, the same spatial-temporal
                                        form as Eq. 17)
```

* `SDF_B` is the exact signed distance to the box; for a point above a box it
  equals the height above the top face — so the safe set **contains the
  step-over trajectory by construction**.
* Keypoints: the four feet (margin 0.03 m — they legitimately skim the sill),
  the four calf links (0.05 m — the knee leads a swing leg into a wall first),
  and the base (0.10 m — trunk half-height plus clearance). The base keypoint
  is also what keeps the 0.4 m walls un-crossable: the same formula, no
  special-casing.
* ℓ̇ comes from the analytic SDF gradient dotted with the keypoint velocity
  (verified against finite differences in the offline tests).

Envs on box-publishing cells use **only** this ℓ; every cylinder terrain keeps
the existing math bit-for-bit. Both feed the same `safety_values` /
`min_cbf_h` tensors, so the reachability critic, safety advantages, damped
null-space projection, adaptive μ, and all logging are unchanged — swapping
the sensor-driven margin ℓ(s_t) is exactly the extension point the paper's
formulation promises.

Both wall representations are published together per cell
(`corridor_obstacle_info`): `positions`/`radius` — the centrelines sampled as
a closed point ring, the format every planar consumer (E4/E4-Y command
filters, `link_obstacle_collision`) already reads — and `boxes` for the ℓ.
Same walls, two projections; each controller family sees the world the way it
natively does.

## The six registered rows

| # | policy | controller | what it answers |
|---|---|---|---|
| 1 | `pi_nom` | body (steering CBF, ω authority ±1.0 — beyond the trained range, **on purpose**) | the high-level-controller steel-man |
| 2 | `pi_nom` | body, braking-only | same, without steering |
| 3 | `pi_nom` | none | does the naive policy blunder across (or trip)? |
| 4 | `pi_trap_ours` | none | **the headline**: crosses by lifting its legs |
| 5 | `pi_trap_ours` | body | **killer control**: a policy that *can* cross is still parked by the 2-D constraint set — the representation binds, not capability |
| 6 | `pi_trap_nom` | none | terrain exposure without safety machinery — separates "saw the terrain" from "the ℓ shaped how it crosses" |

Per episode the runner records: **exited** (base past the gate's outer face
+0.25 m), time-to-exit, progress beyond the gate, **parked** (timed out inside
with <0.25 m net progress over the episode's second half), **wall contact**
via the volumetric test (`any_point_in_boxes` — passing over the sill with
clearance is *not* contact; that distinction is the point), and the
**swing-foot apex over the gate zone** vs. the same policy's apex on the open
pen floor — the "lifts its legs higher than usual" mechanism number.

## Registered predictions

| | prediction |
|---|---|
| (i) | Rows 1, 2, 5: exit rate **exactly 0/N** and parked ≈ 100%. Not "low" — zero, because the 2-D free space has no exit. Any nonzero count is a terrain bug (`analyze_etrap.py` calls this `geometry_violated`). |
| (ii) | Row 4 exits with a Wilson CI clear of zero, with exit rate decaying along the sill-height curve (per-terrain-level breakdown). |
| (iii) | Mechanism: row 4's gate-zone swing apex exceeds both its own open-floor apex and row 3's — the ℓ gradient, not gait noise, produces the lift. |
| (iv) | Row 4's volumetric wall-contact rate stays well below row 3's and row 6's: the crossing is *clean* because ℓ penalizes proximity to the volume, not just failure. |
| (v) | Row 5 matches rows 1–2 (parked), isolating the 2-D representation as the binding constraint. |

**Falsification:** row 4 failing to exit (`not_separated`) means the crossing
was not learned and the argument is not backed — report that plainly and look
at the training gate (below) before touching the terrain.

## Running it

```bash
# 0. Offline checks first — no GPU, no isaacgym needed:
python tests/test_etrap_offline.py

# 1. Train the two trap policies (~2000 iters each; pi_nom is untouched):
python experiments/etrap/train_trap_policy.py --policy ours --task go1_amp \
    --headless --max_iterations 2000 --seed 1        # -> pi_trap_ours
python experiments/etrap/train_trap_policy.py --policy nom --task go1_amp \
    --headless --max_iterations 2000 --seed 1        # -> pi_trap_nom

# 2. The six rows (needs pi_nom plus the two runs above):
./experiments/etrap/sweep_etrap.sh sweep

# 3. Table + verdict, then the tarball:
./experiments/etrap/sweep_etrap.sh analyze
./experiments/etrap/sweep_etrap.sh package
```

Environment overrides: `TASK, DEVICE, PI_NOM_RUN, PI_OURS_RUN, PI_ABL_RUN,
ALPHA, N_EPISODES, EVAL_ENVS, LOOKAHEAD, W_V, W_OMEGA, OMEGA_LIMIT, BARRIER`.
Single rows: `python experiments/etrap/run_etrap.py --task go1_amp --headless
--load_run <run> --policy_tag <tag> --controller {none,body,body_noyaw}`;
`--gate_difficulty d` pins every env to one sill height instead of the 0→1
spread.

## What to check before believing the result

- **The training gate.** `pi_trap_ours` must actually leave the pen during
  training — watch `Episode/terrain_level` climb (the curriculum promotes
  robots that travel > 4 m, which requires exiting) and `Safety/damping_t`
  stay mostly near 1 (the projection is task-neutral except near the sill).
  A run whose terrain level never rises never learned the crossing, and its
  eval row will honestly say `not_separated`.
- **Rows 1/2/5 must be exactly zero.** One exit means the point ring has a
  gap or the spawn leaked outside the pen — `geometry_violated`; nothing else
  in the table is worth reading until that is fixed.
- **Row 3 is the interesting control.** An unfiltered `pi_nom` walks into the
  sill blind. Occasional stumble-overs at the 8 cm end are expected and fine —
  they come with wall contact and falls, which is exactly the contrast to
  row 4's clean, deliberate crossings.
- **ω authority is granted, not trained.** The body arms get ±1.0 rad/s of
  steering authority although `pi_nom` was trained at ±0.01 (E4-Y measured a
  0.015 tracking gain). For this study that *strengthens* the conclusion: even
  granting the filter a steering channel the platform cannot track — and even
  if it could — the pen has no 2-D exit to steer toward. Recorded per-row in
  the manifests.

## Backward compatibility (why nothing else moves)

The step trap is terrain slot 12, reached only by a 13-entry
`terrain_proportions`; the shipped `go1_amp` proportions are unchanged, so
`pi_nom` and every completed study reproduce exactly. Configs with 11- or
12-entry lists hit the identical branch chain as before. Envs whose cells
publish no boxes compute the identical safety value as before (the box branch
is dead code for them); `a1`/`cassie`/`trona1_w` read defaults and are
untouched. The E5 damping-fraction diagnostic (`Safety/damping_t`) reads the
same `safety_values` tensor the box ℓ feeds, so it works on trap training
without modification.
