#!/usr/bin/env bash
# E5 sweep: arms x forward command x push magnitude on flat, plus a stairs row.
#
# Grid cells run at N_EPISODES_GRID; the headline cells (vx = 0.8, all
# magnitudes) at N_EPISODES_HEAD.  Point the *_RUN variables at the trained
# run directories (logs/cassie_viploco_warp_v2/<run_name>).
#
#   PI_NOM_RUN=pi_nom PI_RS_RUN=pi_rs PI_OURS_RUN=pi_ours \
#       bash experiments/e5/sweep_e5.sh
set -euo pipefail

TASK=${TASK:-cassie}
OUT=${OUT:-logs/e5/sweep}
N_EPISODES_GRID=${N_EPISODES_GRID:-300}
N_EPISODES_HEAD=${N_EPISODES_HEAD:-1000}
EVAL_ENVS=${EVAL_ENVS:-250}
HEAD_VX=0.8

PI_NOM_RUN=${PI_NOM_RUN:-pi_nom}
PI_RS_RUN=${PI_RS_RUN:-pi_rs}
PI_OURS_RUN=${PI_OURS_RUN:-pi_ours}

run_cell () {
    local arm=$1 load_run=$2 vx=$3 mag=$4 terrain=$5 n=$6
    python experiments/e5/run_e5.py --task "$TASK" --headless \
        --load_run "$load_run" --arm "$arm" \
        --cmd_vx "$vx" --push_mag "$mag" --push_dirs all \
        --eval_terrain "$terrain" \
        --n_episodes "$n" --eval_envs "$EVAL_ENVS" --out_dir "$OUT"
}

for arm_run in "pi_nom:$PI_NOM_RUN" "pi_rs:$PI_RS_RUN" "pi_ours:$PI_OURS_RUN"; do
    arm=${arm_run%%:*}; load_run=${arm_run##*:}

    # headline row: vx = 0.8, full magnitude ladder, 1000 episodes
    for mag in 0.0 0.5 1.0 1.5 2.0; do
        run_cell "$arm" "$load_run" "$HEAD_VX" "$mag" flat "$N_EPISODES_HEAD"
    done

    # command sweep at a fixed representative magnitude
    for vx in 0.4 1.2; do
        for mag in 0.0 1.0; do
            run_cell "$arm" "$load_run" "$vx" "$mag" flat "$N_EPISODES_GRID"
        done
    done

    # stairs row, no pushes: does the safety arm help on terrain-driven falls?
    run_cell "$arm" "$load_run" "$HEAD_VX" 0.0 stair "$N_EPISODES_GRID"
done

python experiments/e5/analyze_e5.py --dir "$OUT" --out "${OUT%/sweep}/R5"
