#!/usr/bin/env bash
# E4-Y -- command filtering with steering authority.
#
# The completed command-space study filtered forward speed only.  With the
# barrier written on the base centre yaw does not appear in h_dot, so that
# filter could only brake: it had no legal option on 79-100% of the steps it
# engaged, and the robot parked instead of steering.  E4-Y writes the barrier
# on a look-ahead point so both v and omega enter at first order, and runs the
# braking-only behaviour as the yaw-disabled control arm through the same code.
#
# Stages (run individually, or `all`):
#   calibrate  open-loop command-response sweep -- a GATE and a result.
#              Run this first and read its verdict before believing anything
#              downstream: a filter can only be as good as the policy's
#              ability to track the commands it issues.
#   sweep      unfiltered row + alpha x {yaw, noyaw}
#   analyze    Table R1d, the registered predictions, the four-way gate
#   package    tar what an analysis needs, reporting anything missing
#
# Configure by environment variable, e.g.
#   ALPHAS="1 2 5" OMEGA_LIMIT=1.0 ./experiments/e4y/sweep_e4y.sh all
#
# Run experiments/reference/sweep_reference.sh first if you can.  Without
# logs/reference/reference.json the analyser falls back to Table 2's numbers,
# which were measured under a different detector, and it says so loudly.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/e4y}"
PI_NOM_RUN="${PI_NOM_RUN:-pi_nom}"
PI_NOM_CKPT="${PI_NOM_CKPT:--1}"
# Small alpha permits only slow approach and is MORE conservative; large alpha
# is permissive and tends to the unfiltered policy.  §1.3 registers {1, 2, 5}.
ALPHAS="${ALPHAS:-1.0 2.0 5.0}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"
DR="${DR:-off}"
LOOKAHEAD="${LOOKAHEAD:-0.35}"
W_V="${W_V:-4.0}"
W_OMEGA="${W_OMEGA:-1.0}"
BARRIER="${BARRIER:-geometric}"
# Empty means "use the trained yaw range", which is what the policy can
# actually track.  Set it only if calibration shows headroom past that range.
OMEGA_LIMIT="${OMEGA_LIMIT:-}"
CALIB="${CALIB:-${OUT}/calibration.json}"
# Look-ahead 0 reproduces the completed braking-only filter exactly, under the
# corrected projection.  Empty by default; set e.g. REPLICA_ALPHAS="2.0".
REPLICA_ALPHAS="${REPLICA_ALPHAS:-}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
SWEEP_DIR="${OUT}/sweep"
LOAD="--load_run ${PI_NOM_RUN} --checkpoint ${PI_NOM_CKPT}"
EVAL="--n_episodes ${N_EPISODES} --eval_envs ${EVAL_ENVS} --dr ${DR} --out_dir ${SWEEP_DIR}"
GEOM="--lookahead ${LOOKAHEAD} --w_v ${W_V} --w_omega ${W_OMEGA} --barrier ${BARRIER}"
if [ -n "${OMEGA_LIMIT}" ]; then GEOM="${GEOM} --omega_limit ${OMEGA_LIMIT}"; fi

stage="${1:-all}"

if [ "$stage" = "calibrate" ] || [ "$stage" = "all" ]; then
  echo "=== E4-Y calibration: what can the command interface deliver? ==="
  mkdir -p "${OUT}"
  # shellcheck disable=SC2086
  python experiments/e4y/calibrate_commands.py ${COMMON} ${LOAD} --out "${CALIB}"
  echo
  echo "Read the VERDICT above before continuing:"
  echo "  yaw tracking usable      -> E4-Y is a fair test of the steering mechanism"
  echo "  yaw tracking degenerate  -> E4-Y tests retrofitting onto THIS policy;"
  echo "                              that must be stated in the writeup, not"
  echo "                              left for a reader to discover."
fi

if [ "$stage" = "sweep" ] || [ "$stage" = "all" ]; then
  echo "=== E4-Y sweep: unfiltered + alpha in ${ALPHAS} x {yaw, noyaw} ==="
  mkdir -p "${SWEEP_DIR}"
  # shellcheck disable=SC2086
  python experiments/e4y/run_e4y.py ${COMMON} ${LOAD} ${EVAL} --filter none
  for a in ${ALPHAS}; do
    # shellcheck disable=SC2086
    python experiments/e4y/run_e4y.py ${COMMON} ${LOAD} ${EVAL} ${GEOM} \
      --filter unicycle --alpha "${a}" --yaw
    # Control arm: identical code path, yaw axis removed.  This is the
    # completed braking-only filter, re-run here so the comparison is
    # within-harness rather than across studies.
    # shellcheck disable=SC2086
    python experiments/e4y/run_e4y.py ${COMMON} ${LOAD} ${EVAL} ${GEOM} \
      --filter unicycle --alpha "${a}" --no_yaw
  done
  # Optional third arm: look-ahead 0 puts the barrier back on the base centre,
  # which is the completed braking-only study exactly -- re-run here under the
  # corrected half-plane-plus-box projection.  Off by default because it is
  # beyond the registered alpha x {yaw, no-yaw} design and costs one extra run
  # per alpha; set REPLICA_ALPHAS to enable it.
  for a in ${REPLICA_ALPHAS}; do
    # shellcheck disable=SC2086
    python experiments/e4y/run_e4y.py ${COMMON} ${LOAD} ${EVAL} \
      --w_v ${W_V} --w_omega ${W_OMEGA} --barrier ${BARRIER} \
      --filter unicycle --alpha "${a}" --no_yaw --lookahead 0
  done
fi

if [ "$stage" = "analyze" ] || [ "$stage" = "all" ]; then
  echo "=== E4-Y analyze ==="
  python experiments/e4y/analyze_e4y.py --dir "${SWEEP_DIR}" --out "${OUT}/R1d"
fi

if [ "$stage" = "package" ] || [ "$stage" = "all" ]; then
  echo "=== E4-Y package ==="
  STAMP="$(date +%Y%m%d_%H%M%S)"
  PKG="${OUT}/pkg_${STAMP}"
  MISSING="${PKG}/MISSING.txt"
  mkdir -p "${PKG}/sweep"
  : > "${MISSING}"

  take () { if [ -f "$1" ]; then cp "$1" "$2/"; else echo "$1" >> "${MISSING}"; fi }

  for f in "${OUT}/R1d.json" "${OUT}/R1d.md" "${CALIB}"; do take "${f}" "${PKG}"; done
  # The reference rows decide every "dominated" call, so ship them with the
  # results rather than leaving the reader to guess which reference was used.
  take "logs/reference/reference.json" "${PKG}"

  n=0; exp=1
  take "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.csv" "${PKG}/sweep"
  take "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.manifest.json" "${PKG}/sweep"
  if [ -f "${SWEEP_DIR}/${PI_NOM_RUN}_unfiltered.csv" ]; then n=$((n + 1)); fi
  # run_e4y names files with %g, so 1.0 becomes "a1" and 0.35 becomes "L0.35";
  # normalise here or the package silently omits every whole-numbered alpha.
  Lg="$(printf '%g' "${LOOKAHEAD}")"
  for a in ${ALPHAS}; do
    ag="$(printf '%g' "${a}")"
    for arm in yaw noyaw; do
      exp=$((exp + 1))
      base="${PI_NOM_RUN}_uni_a${ag}_L${Lg}_${arm}"
      take "${SWEEP_DIR}/${base}.csv" "${PKG}/sweep"
      take "${SWEEP_DIR}/${base}.manifest.json" "${PKG}/sweep"
      if [ -f "${SWEEP_DIR}/${base}.csv" ]; then n=$((n + 1)); fi
    done
  done
  for a in ${REPLICA_ALPHAS}; do
    exp=$((exp + 1))
    base="${PI_NOM_RUN}_uni_a$(printf '%g' "${a}")_L0_noyaw"
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
    echo "lookahead  ${LOOKAHEAD}"
    echo "weights    w_v=${W_V} w_omega=${W_OMEGA}"
    echo "barrier    ${BARRIER}"
    # Read from the manifests, not from the environment: package is often run
    # in a later shell than sweep, and reporting this shell's OMEGA_LIMIT said
    # "trained range" for a sweep that had actually used +-1.0 rad/s.
    echo "omega_lim  $(python - "${SWEEP_DIR}" <<'PYEOF'
import glob, json, os, sys
vals = set()
for f in glob.glob(os.path.join(sys.argv[1], "*.manifest.json")):
    try:
        m = json.load(open(f))
    except Exception:
        continue
    if m.get("yaw_enabled"):
        vals.add(tuple(m.get("omega_range") or []))
print("; ".join(str(list(v)) for v in sorted(vals)) if vals else "n/a")
PYEOF
)"
    echo "replica_a  ${REPLICA_ALPHAS:-none}"
    echo "episodes   ${N_EPISODES}"
  } > "${PKG}/RUN_INFO.txt"

  TAR="${OUT}/e4y_results_${STAMP}.tar.gz"
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
