#!/usr/bin/env bash
# Same-harness reference rows (standing requirement 1).
#
# Every "is the baseline dominated" question is decided against the proposed
# method's operating point.  Table 2's 4.4% was measured with a base-level
# margin detector under a different harness, so comparing E1/E2/E4 rows to it
# is not like-for-like.  This evaluates the reference policies under *this*
# harness and detector, once, and writes logs/reference/reference.json for the
# analysers to read.
#
# Evaluation only -- no training.  Two runs, ~1-2 h.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-go1_amp}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-logs/reference}"
PI_OURS_RUN="${PI_OURS_RUN:-pi_ours}"
PI_RS_RUN="${PI_RS_RUN:-pi_rs}"
CKPT="${CKPT:--1}"
N_EPISODES="${N_EPISODES:-1000}"
EVAL_ENVS="${EVAL_ENVS:-250}"
DR="${DR:-off}"

COMMON="--task ${TASK} --headless --sim_device ${DEVICE} --rl_device ${DEVICE}"
EVAL="--filter none --n_episodes ${N_EPISODES} --eval_envs ${EVAL_ENVS} --dr ${DR} --out_dir ${OUT}"

stage="${1:-all}"

# Preflight.  These two runs are the paper's policies, not anything this
# package trains, so a fresh clone will not have them.  Check before spending
# an hour discovering it, and be explicit that everything else still runs.
LOG_ROOT="${LOG_ROOT:-logs/go1_viploco_warp_v5}"
missing=""
for r in "${PI_OURS_RUN}" "${PI_RS_RUN}"; do
  [ -d "${LOG_ROOT}/${r}" ] || missing="${missing} ${r}"
done
if [ -n "${missing}" ]; then
  echo "MISSING checkpoint run(s):${missing}"
  echo "  looked in : ${LOG_ROOT}"
  echo "  available : $(ls "${LOG_ROOT}" 2>/dev/null | tr '\n' ' ')"
  echo
  echo "These are the *paper's* policies -- the proposed method and the"
  echo "reward-shaping baseline. Nothing in this package trains them, so a"
  echo "checkout without them cannot produce same-harness reference rows."
  echo
  echo "This does NOT block E1, E2, E4 or E4-Y. Their analysers fall back to"
  echo "Table 2's numbers and print a warning saying the comparison is"
  echo "cross-detector rather than like-for-like. Run them now and come back"
  echo "to this when the checkpoints exist."
  echo
  echo "If they exist under other names, point at them:"
  echo "  PI_OURS_RUN=<run> PI_RS_RUN=<run> $0 ${stage}"
  exit 1
fi

if [ "$stage" = "eval" ] || [ "$stage" = "all" ]; then
  mkdir -p "${OUT}"
  echo "=== reference: proposed method under this harness ==="
  # shellcheck disable=SC2086
  python experiments/e1/run_e1.py ${COMMON} ${EVAL} \
    --load_run "${PI_OURS_RUN}" --checkpoint "${CKPT}" --policy_tag pi_ours
  echo "=== reference: reward shaping under this harness ==="
  # shellcheck disable=SC2086
  python experiments/e1/run_e1.py ${COMMON} ${EVAL} \
    --load_run "${PI_RS_RUN}" --checkpoint "${CKPT}" --policy_tag pi_rs
fi

if [ "$stage" = "collect" ] || [ "$stage" = "all" ]; then
  echo "=== writing ${OUT}/reference.json ==="
  python - "${OUT}" <<'PYEOF'
import csv, json, os, sys
sys.path.insert(0, os.getcwd())
from safeloco_eval.reference import summarise_records

out = sys.argv[1]
mapping = {"ours": "pi_ours_unfiltered.csv",
           "reward_shaping": "pi_rs_unfiltered.csv"}
blob, missing = {}, []
for key, name in mapping.items():
    path = os.path.join(out, name)
    if not os.path.exists(path):
        missing.append(path)
        continue
    with open(path) as fh:
        blob[key] = summarise_records(list(csv.DictReader(fh)))

if missing:
    print("MISSING, not written:")
    for m in missing:
        print("   ", m)
    raise SystemExit(1)

blob["detector"] = "geometric_link_cylinder"
blob["harness"] = "safeloco_eval.eval_common (DR off, 0.6 m/s, fixed seed list)"
dest = os.path.join(out, "reference.json")
with open(dest, "w") as fh:
    json.dump(blob, fh, indent=2)
for k in ("ours", "reward_shaping"):
    v = blob[k]
    print("  %-15s collision %.2f%%  return %.2f  fwd %.3f m/s  dist %.2f m"
          % (k, 100 * v["collision"], v["ret"], v["fwd_vel"], v["distance"]))
print("wrote", dest)
PYEOF
fi
