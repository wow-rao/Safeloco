#!/usr/bin/env bash
# E1 -- deployment-time joint-space projection, collision.  Full pipeline.
#
# Stages (run them individually, or `all` for everything after pretrain):
#   pretrain         train pi_nom from scratch (TRAIN_ITERS) -- NOT part of `all`
#   pretrain_resume  continue that run for TRAIN_ITERS more iterations
#   precheck         evaluate unfiltered pi_nom  <-- GATE on policy quality
#   collect   roll out pi_nom (and optionally pi_rs) -> Q_safe training buffers
#   train     fit Q_safe offline + validation battery   <-- GATE, read it
#   sweep     the epsilon sweep + variants + ablations
#   analyze   Table R1a, verdict, §7 sentence, headline figure
#
# Configure by environment variable, e.g.
#   PI_NOM_RUN=WMP_task_only ./experiments/e1/sweep_e1.sh all
#
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- configuration -------------------------------------------------------
PI_NOM_RUN="${PI_NOM_RUN:-pi_nom}"        # run dir under logs/<experiment_name>/
PI_NOM_CKPT="${PI_NOM_CKPT:--1}"          # -1 = last checkpoint in that run
PI_RS_RUN="${PI_RS_RUN:-}"                # optional reward-shaping run
TRAIN_ITERS="${TRAIN_ITERS:-2000}"        # matches the original pi_nom run
TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/e1}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"             # N_EPISODES should be a multiple
COLLECT_ENVS="${COLLECT_ENVS:-256}"
COLLECT_STEPS="${COLLECT_STEPS:-1000}"
EPOCHS="${EPOCHS:-30}"
ALPHA="${ALPHA:-0.95}"                    # safety-return discount; see README
ENSEMBLE="${ENSEMBLE:-3}"                 # 1 primary head + 2 control members
DR="${DR:-off}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
SWEEP_DIR="${OUT}/sweep"
QSAFE="${OUT}/qsafe"

stage_pretrain() {
  echo "=== [E1 stage 0/4] training pi_nom from scratch (${TRAIN_ITERS} iters) ==="
  python experiments/e1/train_policy.py --policy pi_nom \
    --task "${TASK}" --headless --sim_device "${DEVICE}" \
    --rl_device "${DEVICE}" --run_name "${PI_NOM_RUN}" \
    --max_iterations "${TRAIN_ITERS}"
}

stage_pretrain_resume() {
  echo "=== [E1] resuming pi_nom for ${TRAIN_ITERS} more iterations ==="
  python experiments/e1/train_policy.py --policy pi_nom \
    --task "${TASK}" --headless --sim_device "${DEVICE}" \
    --rl_device "${DEVICE}" --run_name "${PI_NOM_RUN}" \
    --resume --load_run "${PI_NOM_RUN}" --max_iterations "${TRAIN_ITERS}"
}

stage_precheck() {
  echo "=== [E1 gate] unfiltered pi_nom quality check ==="
  python experiments/e1/run_e1.py ${COMMON} \
    --load_run "${PI_NOM_RUN}" --checkpoint "${PI_NOM_CKPT}" \
    --filter none --n_episodes 250 --eval_envs 250 \
    --eval_terrain corridor --dr "${DR}" --out_dir "${OUT}/precheck"
  echo
  echo ">>> Characterises pi_nom; not a pass/fail on training length. Eval"
  echo ">>> spreads envs over all ten terrain levels on purpose, including"
  echo ">>> ones the curriculum never reached. Check only that there is a"
  echo ">>> signal to measure: non-trivial collision rate, return not zero."
}

stage_collect() {
  echo "=== [E1 stage 1/4] collecting Q_safe buffers ==="
  python experiments/e1/collect_qsafe_data.py ${COMMON} \
    --load_run "${PI_NOM_RUN}" --checkpoint "${PI_NOM_CKPT}" \
    --collect_terrain corridor --collect_envs "${COLLECT_ENVS}" \
    --collect_steps "${COLLECT_STEPS}" \
    --out "${OUT}/qsafe_data/pi_nom"

  if [ -n "${PI_RS_RUN}" ]; then
    python experiments/e1/collect_qsafe_data.py ${COMMON} \
      --load_run "${PI_RS_RUN}" --checkpoint -1 \
      --collect_terrain corridor --collect_envs "${COLLECT_ENVS}" \
      --collect_steps "${COLLECT_STEPS}" \
      --out "${OUT}/qsafe_data/pi_rs"
  else
    echo "  (PI_RS_RUN unset -- training Q_safe on the pi_nom buffer alone."
    echo "   The protocol asks for mixed substrates; note the deviation in App. J.)"
  fi
}

stage_train() {
  echo "=== [E1 stage 2/4] training Q_safe + validation battery ==="
  BUFS="${OUT}/qsafe_data/pi_nom"
  [ -f "${OUT}/qsafe_data/pi_rs.pt" ] && BUFS="${BUFS} ${OUT}/qsafe_data/pi_rs"
  python experiments/e1/train_qsafe.py \
    --buffers ${BUFS} --out "${QSAFE}" --epochs "${EPOCHS}" \
    --alpha "${ALPHA}" --ensemble "${ENSEMBLE}" --device "${DEVICE}"
  echo
  echo ">>> GATE: read ${QSAFE}.report.json before running the sweep."
  echo ">>> If critic_informative is false, E1 still runs, but its §7 sentence"
  echo ">>> becomes the 'Q-hat failed validation' row -- analyze_e1.py applies"
  echo ">>> that automatically from the manifest."
}

run_point() {  # run_point <extra args...>
  python experiments/e1/run_e1.py ${COMMON} \
    --load_run "${PI_NOM_RUN}" --checkpoint "${PI_NOM_CKPT}" \
    --out_dir "${SWEEP_DIR}" --n_episodes "${N_EPISODES}" \
    --eval_envs "${EVAL_ENVS}" --eval_terrain corridor --dr "${DR}" \
    "$@"
}

stage_sweep() {
  echo "=== [E1 stage 3/4] epsilon sweep ==="

  # (0) unfiltered pi_nom -- the reference every verdict rule is relative to.
  #     Q_safe is loaded so the row also carries the counterfactual trigger
  #     rate, and so "activation % is 0" is an observation, not a tautology.
  run_point --filter none --qsafe_ckpt "${QSAFE}.pt"

  # (1) variant A, primary sweep, trigger tau = d_safe = -0.25.
  for EPS in 0.02 0.05 0.1 0.2 0.5; do
    run_point --filter A --epsilon "${EPS}" --qsafe_ckpt "${QSAFE}.pt"
  done

  # (2) variant B, gradient projection, at eps in {0.1, 0.5}: preempts
  #     "sampling is a strawman".
  for EPS in 0.1 0.5; do
    run_point --filter B --epsilon "${EPS}" --qsafe_ckpt "${QSAFE}.pt"
  done

  # (3) always-on ablation at eps = 0.1: separates trigger timing from
  #     correction effects.
  run_point --filter A --epsilon 0.1 --always_on --qsafe_ckpt "${QSAFE}.pt"

  # (4) secondary trigger tau = 0.
  run_point --filter A --epsilon 0.1 --tau 0.0 --qsafe_ckpt "${QSAFE}.pt"

  # (5) 3-head min-aggregated ensemble control at one epsilon.
  if [ -f "${QSAFE}.member1.pt" ]; then
    run_point --filter A --epsilon 0.1 --qsafe_ckpt "${QSAFE}.pt" \
      --qsafe_ensemble "${QSAFE}.member1.pt" "${QSAFE}.member2.pt"
  fi

  # (6) optional: the same filter over pi_rs at eps = 0.1.
  if [ -n "${PI_RS_RUN}" ]; then
    python experiments/e1/run_e1.py ${COMMON} \
      --load_run "${PI_RS_RUN}" --checkpoint -1 \
      --out_dir "${SWEEP_DIR}" --n_episodes "${N_EPISODES}" \
      --eval_envs "${EVAL_ENVS}" --eval_terrain corridor --dr "${DR}" \
      --filter A --epsilon 0.1 --qsafe_ckpt "${QSAFE}.pt" --policy_tag pi_rs
  fi
}

stage_analyze() {
  echo "=== [E1 stage 4/4] Table R1a + verdict ==="
  python experiments/e1/analyze_e1.py --dir "${SWEEP_DIR}" \
    --out "${OUT}/R1a" --qsafe_report "${QSAFE}.report.json"
}

case "${1:-all}" in
  pretrain)        stage_pretrain ;;
  pretrain_resume) stage_pretrain_resume ;;
  precheck)        stage_precheck ;;
  collect)  stage_collect ;;
  train)    stage_train ;;
  sweep)    stage_sweep ;;
  analyze)  stage_analyze ;;
  # `all` deliberately excludes pretrain -- it is a separate, long job.
  all)      stage_collect; stage_train; stage_sweep; stage_analyze ;;
  *) echo "usage: $0 {pretrain|pretrain_resume|precheck|collect|train|sweep|analyze|all}"
     exit 1 ;;
esac
