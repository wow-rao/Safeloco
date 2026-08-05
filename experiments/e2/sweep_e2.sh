#!/usr/bin/env bash
# E2 -- training-time filtering (the CBF-RL mechanism), collision.
#
# Stages (run individually, or `all` for calibrate -> train -> eval -> analyze):
#   calibrate  short runs across EPS_GRID; reports how hard the filter binds
#              <-- GATE: read calibration.json before committing to `train`
#   train      3 variants x SEEDS full training runs
#   eval       every checkpoint evaluated twice, with and without the filter
#   analyze    Table R2a, the §3.2 verdict, the §7 sentence
#
# Configure by environment variable, e.g.
#   EPS=0.2 SEEDS="1 2 3" ./experiments/e2/sweep_e2.sh all
#
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- configuration -------------------------------------------------------
TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/e2}"
EXPERIMENT="${EXPERIMENT:-go1_viploco_warp_v5}"   # logs/<EXPERIMENT>/<run_name>
QSAFE="${QSAFE:-logs/e1/qsafe.pt}"
QSAFE_REPORT="${QSAFE_REPORT:-logs/e1/qsafe.report.json}"

SEEDS="${SEEDS:-1 2 3}"                  # §2 wants >= 3; never pooled
VARIANTS="${VARIANTS:-reward_only filter_only dual}"
EPS="${EPS:-0.2}"                        # training-time trust region
EPS_GRID="${EPS_GRID:-0.05 0.1 0.2 0.5}" # calibration only
TAU="${TAU:--0.25}"
FILTER="${FILTER:-A}"
RS_WEIGHT="${RS_WEIGHT:-1.0}"
TRAIN_ITERS="${TRAIN_ITERS:-1500}"
CAL_ITERS="${CAL_ITERS:-150}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"
DR="${DR:-off}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
EVAL_DIR="${OUT}/eval"

stage="${1:-all}"

uses_filter () { [ "$1" = "filter_only" ] || [ "$1" = "dual" ]; }

# ---- calibrate -----------------------------------------------------------
# epsilon is not registered anywhere in the protocol. Rather than pick one by
# hand, run a short pass at each candidate and read off whether the filter
# actually binds without dragging return down. A filter that never engages
# makes filter_only a copy of pi_nom; one that engages too hard collapses
# training and says nothing about internalization.
if [ "$stage" = "calibrate" ] || [ "$stage" = "all" ]; then
  echo "=== E2 calibrate: eps in ${EPS_GRID}, ${CAL_ITERS} iters each ==="
  for e in ${EPS_GRID}; do
    python experiments/e2/train_e2.py ${COMMON} \
      --variant filter_only --seed 1 --epsilon "${e}" --tau "${TAU}" \
      --filter "${FILTER}" --qsafe_ckpt "${QSAFE}" \
      --qsafe_report "${QSAFE_REPORT}" --calibrate "${CAL_ITERS}"
  done
  echo
  echo "Read logs/${EXPERIMENT}/e2_filter_only_s1/calibration.json for each eps."
  echo "Pick the largest eps that binds (frac_filtered well above 0) without a"
  echo "downward return trend, then re-run with EPS=<choice> ./sweep_e2.sh train"
fi

# ---- train ---------------------------------------------------------------
if [ "$stage" = "train" ] || [ "$stage" = "all" ]; then
  echo "=== E2 train: ${VARIANTS} x seeds ${SEEDS}, eps=${EPS} ==="
  for v in ${VARIANTS}; do
    for s in ${SEEDS}; do
      args="--variant ${v} --seed ${s} --rs_weight ${RS_WEIGHT}
            --max_iterations ${TRAIN_ITERS}"
      if uses_filter "${v}"; then
        args="${args} --epsilon ${EPS} --tau ${TAU} --filter ${FILTER}
              --qsafe_ckpt ${QSAFE} --qsafe_report ${QSAFE_REPORT}"
      fi
      # shellcheck disable=SC2086
      python experiments/e2/train_e2.py ${COMMON} ${args}
    done
  done
fi

# ---- eval ----------------------------------------------------------------
# Each checkpoint twice: the internalization gap is the difference between
# the two rows, so they must share the seed list and differ only in whether
# the filter is attached.
if [ "$stage" = "eval" ] || [ "$stage" = "all" ]; then
  echo "=== E2 eval: every checkpoint with and without the test filter ==="
  mkdir -p "${EVAL_DIR}"
  for v in ${VARIANTS}; do
    for s in ${SEEDS}; do
      run="e2_${v}_s${s}"
      for tf in on off; do
        args="--load_run ${run} --variant_tag ${v} --seed_tag ${s}
              --test_filter ${tf} --n_episodes ${N_EPISODES}
              --eval_envs ${EVAL_ENVS} --dr ${DR} --out_dir ${EVAL_DIR}"
        if [ "${tf}" = "on" ]; then
          args="${args} --qsafe_ckpt ${QSAFE} --epsilon ${EPS}
                --tau ${TAU} --filter ${FILTER}"
        fi
        # shellcheck disable=SC2086
        python experiments/e2/run_e2.py ${COMMON} ${args}
      done
    done
  done
fi

# ---- analyze -------------------------------------------------------------
if [ "$stage" = "analyze" ] || [ "$stage" = "all" ]; then
  echo "=== E2 analyze ==="
  python experiments/e2/analyze_e2.py \
    --dir "${EVAL_DIR}" \
    --train_dir "logs/${EXPERIMENT}" \
    --out "${OUT}/R2a"
fi
