# Dynamic safety for bipedal locomotion: fall avoidance as reachability

*Companion note to "Reachability-Guided Safe Policy Learning for Legged
Locomotion". Defines the fall-safety constraint the E5 experiment
(`experiments/e5/`) instantiates, and the claim it tests.*

## 1. The reviewer's distinction, taken seriously

In the paper's quadruped experiments, the margin function ℓ is obstacle
clearance or joint-limit distance. Both are (nearly) functions of the
*configuration* alone: the safe set they induce is a region of configuration
space, and at moderate speeds the robot can stop or steer within it, so the
instantaneous margin ℓ(s) and the reachability value V_safe(s) largely agree.
Safety there is *quasi-static*.

Bipedal locomotion breaks this. The failure — falling — is certainly
*detectable* from the configuration (a torso past 60°, a pelvis on the
ground), but the *hazard* is not a region of configuration space: an upright
biped whose center of mass carries the wrong momentum is already doomed, and
a tilted biped mid-recovery-step may be perfectly safe. Whether a state is
safe depends irreducibly on velocities and contact, i.e. on what the dynamics
still permit. This is the classical observation behind viability theory for
walking [Wieber 2002; Wieber 2008] and capturability [Pratt et al. 2006;
Koolen et al. 2012]: *balance is not a function of the state's pose but of
the reachable future*.

Our claim: the paper's framework already contains exactly the object this
requires — the learned reachability value — and extending it to bipedal fall
safety needs no structural change, only a new margin function. Moreover, the
biped is the regime where the reachability machinery stops being a
convenience and becomes the content: for the quadruped tasks V_safe ≈ ℓ; for
falling, the entire signal is the *gap* between them.

## 2. Definition

### 2.1 Fallen set: detection is static

Let ĝ_z(s) ∈ [−1, 1] be the z-component of the projected gravity vector in
the base frame (−1 upright, 0 at 90° of tilt), h(s) the pelvis height above
the local terrain, and C_illegal(s) the indicator that a non-foot body
(pelvis, shin, tarsus) carries ground contact force above 1 N. Define the
**fallen-state margin**

    ℓ_fall(s) = min( ℓ_tilt(s), ℓ_height(s), ℓ_contact(s) )

    ℓ_tilt   = (ĝ_z* − ĝ_z) / (1 + ĝ_z*),         ĝ_z* = −0.5  (60° tilt)
    ℓ_height = (h − h_fall) / (h_nom − h_fall),     h_fall = 0.15 m
    ℓ_contact = −1 if C_illegal else inactive

All components are normalized and dimensionless: ℓ_tilt is +1 upright, 0 at
the tilt threshold; ℓ_height is 1 at nominal pelvis height, 0 at the fall
height. The **fallen set** is F = {s : ℓ_fall(s) < 0}. The thresholds are
the evaluation fall definition already used for the quadruped results
(`safeloco_eval/metrics.py`), so the margin's zero crossings and the metric
"the robot fell" cannot disagree.

ℓ_fall is *deliberately* a configuration-space quantity. It detects a fall;
it cannot predict one. That is not a deficiency of the definition — it is
the definition doing its job of separating the two roles.

### 2.2 Dynamic safety: viability with respect to F

**Definition (dynamic fall safety).** A state s is dynamically safe iff it
lies in the viability kernel of the fallen set:

    S_safe = { s : V_safe(s) ≥ 0 },
    V_safe(s) = sup_π inf_{t≥0} ℓ_fall( s_t | s_0 = s, π )

i.e. *some* control policy keeps the robot un-fallen for all future time.
This is the paper's Eq. (1) with ℓ = ℓ_fall, and the training constraint is
the paper's Eq. (4), V_safe(s_t) ≥ δ_safe, unchanged. The learned safety
critic V̂_safe is trained with the same worst-case temporal recursion (Eq. 5)
on the same rollouts; the damped null-space projection consumes its policy
gradient exactly as before. Operationally, the entire extension is swapping
the sensor-driven margin function — the recipe Section 5 of the paper already
uses to move between joint-limit and collision constraints.

Three properties make this the right definition of "dynamic safety":

1. **It subsumes the classical notions.** The viability kernel of F is
   precisely Wieber's viable set for walking; N-step capturability [Koolen
   et al. 2012] is its inner approximation under the linear-inverted-pendulum
   (LIP) restriction; the reach-avoid value of ABS [He et al. 2024] is the
   analogous object computed over a 2-D centroidal velocity space. Our
   V_safe is the model-free, full-state generalization: it is defined over
   the same state the policy acts on (joint states, base pose and twist,
   contact), so "recoverable" includes swing-leg state and actuation limits
   that pendulum models abstract away.

2. **The unsafe set is genuinely dynamic.** On a substantial set of states,
   sign(V_safe) ≠ sign(ℓ_fall): upright-but-doomed states (ℓ_fall > 0,
   V_safe < 0 — the capture point beyond the reachable footholds) and
   recoverable-tilt states (ℓ_fall barely positive after a shove, V_safe
   comfortably positive because a recovery step exists). No margin that is a
   function of configuration alone can have the right sign on both.

3. **It is falsifiable, per state, in simulation.** V_safe < 0 asserts "no
   policy avoids F from here" — testable by whether *this* policy falls, and
   quantified below as fall-prediction AUC and warning lead time.

### 2.3 Capturability-informed margin (the dense variant)

The paper's corridor experiments found that a velocity-augmented margin
ℓ_col = h + α_h·ḣ outperforms the purely spatial ℓ = h (Table 10), with
α_h = 0.3 hand-tuned. For the biped, the analogous densification is the
divergent component of motion (DCM / instantaneous capture point)
[Pratt et al. 2006; Englsberger et al. 2015]:

    ξ = p_com,xy + v_com,xy / ω₀,   ω₀ = sqrt(g / z_com)
    ℓ_dcm(s) = ( r_cap − ‖ξ − p_support‖ ) / r_cap

positive while the capture point stays inside the disc of radius r_cap
(the reachable foothold region) around the support midpoint — one-step
capturability of the LIP. Note the structural identity: ℓ_dcm is exactly
"h + α·ḣ" applied to the fall constraint, with the time constant α = 1/ω₀
*fixed by the pendulum dynamics* rather than tuned. The corridor margin was,
in hindsight, a one-dimensional DCM margin.

This yields the two experimental variants:

- **A — `fail_set` (primary definition).** ℓ = ℓ_fall. The margin is static;
  every bit of dynamic content must be learned by the reachability critic.
  The mirror of the paper's sparse joint-limit regime.
- **B — `fail_set_dcm` (ablation).** ℓ = min(ℓ_fall, ℓ_dcm). A model-based
  prior densifies the signal; velocity enters the margin directly. The
  mirror of the paper's spatial-temporal corridor margin.

A is the definition; B measures how much of the dynamics the critic recovers
on its own versus how much a physics prior helps — the biped counterpart of
Table 10's ablation.

## 3. Why the paper's machinery is the right tool here

- **The worst-case recursion is the fall semantics.** Fall risk does not
  time-average: a rollout that is safe for 900 steps and fallen for 3 is a
  fallen rollout. The paper's Rsafe recursion (Eq. 5) propagates the *min*
  backward, which is exactly the inf over the trajectory in the viability
  definition — with α setting the effective horizon. Falls need a longer
  horizon than obstacle proximity (the point of no return precedes the tilt
  threshold by several hundred milliseconds), so E5 raises α from 0.7 to 0.9
  (~0.75 s at 40 Hz); this knob is now exposed in the config
  (`algorithm.safety_return_alpha`).
- **The damping schedule is push recovery.** Near the viability boundary,
  safety must override the velocity command (brake, sidestep, take the
  recovery step). The reachability-aware damping λ(V̂_safe) is precisely the
  mechanism that grants safety that authority only when the *predicted*
  worst-case margin — not the instantaneous pose — says danger. The biped
  thresholds live on the normalized margin scale
  (`algorithm.d_safe/d_danger = −0.05/−0.35`).
- **Asymmetry is right for falling.** A symmetric trade (reward shaping)
  buys safety with tracking everywhere; the asymmetric projection pays only
  near the boundary. On a biped the boundary is crossed rarely but
  catastrophically — the regime where the asymmetric design should show the
  largest advantage over scalarization.

## 4. The experiment (E5) and its registered claims

Platform: Cassie (12 actuated DoF) in IsaacGym; velocity tracking on 70%
flat / 30% stairs terrain with training pushes; standard termination penalty
in *all* arms. Arms differ only in mechanism: π_nom (no safety signal beyond
termination), π_rs (ℓ as a scalar reward penalty), π_ours (reachability
critic + damped null-space projection). Evaluation: directed base-velocity
impulses of magnitude 0–2 m/s on a fixed schedule, fall rate / falls per
1000 steps / time-to-fall, velocity-tracking error and return, and
fall-prediction calibration.

The registered predictions (`experiments/e5/README.md`):

  (i)   π_ours reduces falls/1000-steps vs π_nom at push magnitudes ≥ 1 m/s
        with non-overlapping episode-clustered CIs — the mechanism claim;
  (ii)  π_rs lands between — scalarization helps but pays elsewhere;
  (iii) **the dynamic-safety claim**: AUC(−V̂_safe) > AUC(−ℓ_fall) for
        predicting "fall within 1 s", and V̂_safe's zero crossing leads
        ℓ_fall's by a positive median lead time on fallen episodes. This is
        the quantitative form of "the static margin detects, the
        reachability value predicts";
  (iv)  the tracking cost of π_ours is bounded (< 0.1 m/s added velocity
        error at zero push) — the Pareto claim.

Prediction (iii) is the direct answer to the reviewer: it measures, on the
same states, how much safety-relevant information exists in the full-state
reachability value that is invisible to any configuration-space margin.

## 5. Relation to prior work (rebuttal anchors)

- P.-B. Wieber, "On the stability of walking systems," IARP 2002; and
  "Viability and predictive control for safe locomotion," IROS 2008 —
  viability kernel as the correct safety object for walking.
- J. Pratt, J. Carff, S. Drakunov, A. Goswami, "Capture point: A step toward
  humanoid push recovery," Humanoids 2006 — the capture point.
- T. Koolen, T. de Boer, J. Rebula, A. Goswami, J. Pratt, "Capturability-
  based analysis and control of legged locomotion, Part 1," IJRR 31(9),
  2012 — N-step capturability as a computable inner approximation of
  viability under the LIP.
- J. Englsberger, C. Ott, A. Albu-Schäffer, "Three-dimensional bipedal
  walking control based on divergent component of motion," T-RO 31(2), 2015
  — the DCM used in ℓ_dcm.
- T. He et al., "Agile but safe" (ABS), RSS 2024 — already cited as [18]:
  learns a reach-avoid value over 2-D centroidal velocities to switch to a
  recovery policy. Our fall-safety value generalizes this to the full state
  and consumes it at training time through the projection instead of a
  runtime switch.

## 6. Implementation map

| concern | where |
|---|---|
| margin math (variants A/B) | `safeloco_eval/fall_margin.py` |
| env override (margin each step) | `legged_gym/envs/cassie/cassie.py::_compute_safety_value` |
| margin config | `CassieCfg.fall_safety` (`cassie_config.py`) |
| reachability recursion + α | `rsl_rl/storage/rollout_storage.py::compute_safety_returns` |
| projection + damping | `rsl_rl/algorithms/{cone_constraint,ppo}.py` (unchanged math) |
| biped safety hyperparameters | `CassieCfgPPO.algorithm` |
| training arms | `experiments/e5/train_biped.py` |
| push eval + V_safe logging | `safeloco_eval/eval_biped.py`, `experiments/e5/run_e5.py` |
| tables, calibration, verdict | `experiments/e5/analyze_e5.py` |
| offline tests | `tests/test_fall_margin.py`, `tests/test_ppo_safety_guard.py`, `tests/test_e5_offline.py` |
