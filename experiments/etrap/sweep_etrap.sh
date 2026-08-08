#!/usr/bin/env bash
# ETRAP -- the step trap: body-level avoidance vs gradient-projection leg-lift.
#
# The robot spawns inside a closed pen; three walls are 0.4 m, the fourth is
# a 0.08-0.16 m sill in the commanded direction.  Projected to the plane the
# pen is a closed ring, so a body-level (command-space) controller has no
# exit whatever its gain or steering authority -- its best safe answer is to
# park.  The gradient-projection policy is trained with the volumetric
# l-value (legged_gym/utils/box_sdf.py), whose safe set contains the space
# above the sill, so it can lift its swing legs higher than usual and leave.
#
# Stages (run individually, or `all`):
#   train     pi_trap_ours (safety on) + pi_trap_nom (task-only ablation).
#             ~2000 iterations each; hours of GPU time. Skipped by `all`
#             unless TRAIN=1, so a re-analysis cannot silently retrain.
#   sweep     the six registered rows (see below)
#   analyze   the exit-rate table, the sill-height curve, the verdict
#   package   tarball for analysis
#
# Registered rows:
#   1. pi_nom       + body        the high-level controller steel-man
#   2. pi_nom       + body_noyaw  same filter, braking only
#   3. pi_nom       unfiltered    does the naive policy blunder across?
#   4. pi_trap_ours unfiltered    THE HEADLINE: crosses by lifting its legs
#   5. pi_trap_ours + body        killer control: a policy that CAN cross is
#                                 still parked by the 2-D constraint set --
#                                 the representation, not capability, binds
#   6. pi_trap_nom  unfiltered    terrain exposure without the safety
#                                 machinery -- separates "saw the terrain"
#                                 from "the l-value shaped how it crosses"
#
# Configure by environment variable, e.g.
#   ALPHA=2.0 N_EPISODES=1000 ./experiments/etrap/sweep_etrap.sh sweep
#
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/etrap}"
PI_NOM_RUN="${PI_NOM_RUN:-pi_nom}"
PI_OURS_RUN="${PI_OURS_RUN:-pi_trap_ours}"
PI_ABL_RUN="${PI_ABL_RUN:-pi_trap_nom}"
ALPHA="${ALPHA:-2.0}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"
DR="${DR:-off}"
LOOKAHEAD="${LOOKAHEAD:-0.35}"
W_V="${W_V:-4.0}"
W_OMEGA="${W_OMEGA:-1.0}"
OMEGA_LIMIT="${OMEGA_LIMIT:-1.0}"
BARRIER="${BARRIER:-geometric}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
TRAIN="${TRAIN:-0}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
SWEEP_DIR="${OUT}/sweep"
EVAL="--n_episodes ${N_EPISODES} --eval_envs ${EVAL_ENVS} --dr ${DR} --out_dir ${SWEEP_DIR}"
GEOM="--alpha ${ALPHA} --lookahead ${LOOKAHEAD} --w_v ${W_V} --w_omega ${W_OMEGA} --omega_limit ${OMEGA_LIMIT} --barrier ${BARRIER}"

stage="${1:-all}"

run_row () {  # run_row <load_run> <tag> <controller>
  # shellcheck disable=SC2086
  python experiments/etrap/run_etrap.py ${COMMON} ${EVAL} ${GEOM} \
    --load_run "$1" --policy_tag "$2" --controller "$3"
}

if [ "$stage" = "train" ] || { [ "$stage" = "all" ] && [ "${TRAIN}" = "1" ]; }; then
  echo "=== ETRAP train: pi_trap_ours (gradient projection) + pi_trap_nom ==="
  # shellcheck disable=SC2086
  python experiments/etrap/train_trap_policy.py ${COMMON} \
    --policy ours --max_iterations "${TRAIN_ITERS}" --seed 1
  # shellcheck disable=SC2086
  python experiments/etrap/train_trap_policy.py ${COMMON} \
    --policy nom --max_iterations "${TRAIN_ITERS}" --seed 1
fi

if [ "$stage" = "sweep" ] || [ "$stage" = "all" ]; then
  echo "=== ETRAP sweep: the six registered rows ==="
  mkdir -p "${SWEEP_DIR}"
  run_row "${PI_NOM_RUN}"  "${PI_NOM_RUN}"  body
  run_row "${PI_NOM_RUN}"  "${PI_NOM_RUN}"  body_noyaw
  run_row "${PI_NOM_RUN}"  "${PI_NOM_RUN}"  none
  run_row "${PI_OURS_RUN}" "${PI_OURS_RUN}" none
  run_row "${PI_OURS_RUN}" "${PI_OURS_RUN}" body
  run_row "${PI_ABL_RUN}"  "${PI_ABL_RUN}"  none
fi

if [ "$stage" = "analyze" ] || [ "$stage" = "all" ]; then
  echo "=== ETRAP analyze ==="
  python experiments/etrap/analyze_etrap.py --dir "${SWEEP_DIR}" --out "${OUT}/R_trap"
fi

if [ "$stage" = "package" ] || [ "$stage" = "all" ]; then
  echo "=== ETRAP package ==="
  STAMP="$(date +%Y%m%d_%H%M%S)"
  PKG="${OUT}/pkg_${STAMP}"
  MISSING="${PKG}/MISSING.txt"
  mkdir -p "${PKG}/sweep"
  : > "${MISSING}"

  take () { if [ -f "$1" ]; then cp "$1" "$2/"; else echo "$1" >> "${MISSING}"; fi }

  take "${OUT}/R_trap.json" "${PKG}"
  take "${OUT}/R_trap.md" "${PKG}"

  ag="$(printf '%g' "${ALPHA}")"
  Lg="$(printf '%g' "${LOOKAHEAD}")"
  wg="$(printf '%g' "${OMEGA_LIMIT}")"
  n=0; exp=0
  for base in \
    "${PI_NOM_RUN}_body_a${ag}_L${Lg}_w${wg}" \
    "${PI_NOM_RUN}_body_noyaw_a${ag}_L${Lg}_w${wg}" \
    "${PI_NOM_RUN}_unfiltered" \
    "${PI_OURS_RUN}_unfiltered" \
    "${PI_OURS_RUN}_body_a${ag}_L${Lg}_w${wg}" \
    "${PI_ABL_RUN}_unfiltered"; do
    exp=$((exp + 1))
    take "${SWEEP_DIR}/${base}.csv" "${PKG}/sweep"
    take "${SWEEP_DIR}/${base}.manifest.json" "${PKG}/sweep"
    if [ -f "${SWEEP_DIR}/${base}.csv" ]; then n=$((n + 1)); fi
  done

  {
    echo "packaged   $(date -Is)"
    echo "git_commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty  $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "policies   nom=${PI_NOM_RUN} ours=${PI_OURS_RUN} abl=${PI_ABL_RUN}"
    echo "alpha      ${ALPHA}"
    echo "lookahead  ${LOOKAHEAD}"
    echo "omega_lim  ${OMEGA_LIMIT}"
    echo "weights    w_v=${W_V} w_omega=${W_OMEGA}"
    echo "barrier    ${BARRIER}"
    echo "episodes   ${N_EPISODES}"
  } > "${PKG}/RUN_INFO.txt"

  TAR="${OUT}/etrap_results_${STAMP}.tar.gz"
  tar -czf "${TAR}" -C "$(dirname "${PKG}")" "$(basename "${PKG}")"
  echo
  echo "  sweep CSVs     ${n}/${exp}"
  if [ -s "${MISSING}" ]; then
    echo "  MISSING $(wc -l < "${MISSING}") file(s):"
    sed 's/^/    /' "${MISSING}"
  else
    echo "  nothing missing"
  fi
  echo
  echo "  -> ${TAR}  ($(du -h "${TAR}" | cut -f1))"
fi
