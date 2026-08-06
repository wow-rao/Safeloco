#!/usr/bin/env bash
# E4 -- command-space filtering, the steel-man baseline.
#
# Stages (run individually, or `all`):
#   sweep    unfiltered reference row + one row per alpha
#   analyze  Table R1c, the §3.4 gate, the §7 sentence
#   package  tar what an analysis needs, reporting anything missing
#
# Configure by environment variable, e.g.
#   ALPHAS="0.25 0.5 1 2 5" ./experiments/e4/sweep_e4.sh all
#
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/e4}"
PI_NOM_RUN="${PI_NOM_RUN:-pi_nom}"
PI_NOM_CKPT="${PI_NOM_CKPT:--1}"
# Small alpha permits only slow approach and is MORE conservative;
# large alpha is permissive and tends to the unfiltered policy.
ALPHAS="${ALPHAS:-0.25 0.5 1.0 2.0 5.0}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"
DR="${DR:-off}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
SWEEP_DIR="${OUT}/sweep"
LOAD="--load_run ${PI_NOM_RUN} --checkpoint ${PI_NOM_CKPT}"
EVAL="--n_episodes ${N_EPISODES} --eval_envs ${EVAL_ENVS} --dr ${DR} --out_dir ${SWEEP_DIR}"

stage="${1:-all}"

if [ "$stage" = "sweep" ] || [ "$stage" = "all" ]; then
  echo "=== E4 sweep: unfiltered + alpha in ${ALPHAS} ==="
  mkdir -p "${SWEEP_DIR}"
  # shellcheck disable=SC2086
  python experiments/e4/run_e4.py ${COMMON} ${LOAD} ${EVAL} --filter none
  for a in ${ALPHAS}; do
    # shellcheck disable=SC2086
    python experiments/e4/run_e4.py ${COMMON} ${LOAD} ${EVAL} \
      --filter cbf --alpha "${a}"
  done
fi

if [ "$stage" = "analyze" ] || [ "$stage" = "all" ]; then
  echo "=== E4 analyze ==="
  python experiments/e4/analyze_e4.py --dir "${SWEEP_DIR}" --out "${OUT}/R1c"
fi

if [ "$stage" = "package" ] || [ "$stage" = "all" ]; then
  echo "=== E4 package ==="
  STAMP="$(date +%Y%m%d_%H%M%S)"
  PKG="${OUT}/pkg_${STAMP}"
  MISSING="${PKG}/MISSING.txt"
  mkdir -p "${PKG}/sweep"
  : > "${MISSING}"

  take () { if [ -f "$1" ]; then cp "$1" "$2/"; else echo "$1" >> "${MISSING}"; fi }

  for f in "${OUT}/R1c.json" "${OUT}/R1c.md"; do take "${f}" "${PKG}"; done

  n=0; exp=1
  take "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.csv" "${PKG}/sweep"
  take "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.manifest.json" "${PKG}/sweep"
  if [ -f "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.csv" ]; then n=$((n + 1)); fi
  for a in ${ALPHAS}; do
    exp=$((exp + 1))
    base="${PI_NOM_RUN}_cbf_alpha${a}"
    take "${SWEEP_DIR}/${base}.csv" "${PKG}/sweep"
    take "${SWEEP_DIR}/${base}.manifest.json" "${PKG}/sweep"
    if [ -f "${SWEEP_DIR}/${base}.csv" ]; then n=$((n + 1)); fi
  done

  {
    echo "packaged   $(date -Is)"
    echo "git_commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty  $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "policy     ${PI_NOM_RUN} @ ${PI_NOM_CKPT}"
    echo "alphas     ${ALPHAS}"
    echo "episodes   ${N_EPISODES}"
  } > "${PKG}/RUN_INFO.txt"

  TAR="${OUT}/e4_results_${STAMP}.tar.gz"
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
