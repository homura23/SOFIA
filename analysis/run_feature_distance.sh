#!/bin/bash
# Output images:
# - ${OUT_DIR}/<csv>_ai<idx>_real<idx>.per_branch.png
# - ${OUT_DIR}/<csv>_ai<idx>_real<idx>.global.png
set -eo pipefail
# set -x

# ===== Silence noisy warnings =====
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY="error"
export HF_HUB_DISABLE_TELEMETRY=1

# ==== 1. Env ====
# conda activate your_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==== 2. Defaults ====
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANALYSIS_PY="${BASE_DIR}/analysis/analyze_feature_distance.py"
CONFIG="${BASE_DIR}/config/moe.yaml"
CKPT="${BASE_DIR}/checkpoint/ckpt.pt"
DEVICE="cuda"
AMP_FLAG="--no-amp"
TIMEOUT_SECS="1800"
LOG_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/analysis/outputs/feature_distance"
METRIC="cosine"
MAX_PAIRS="10"
PAIRS_PER_AI="5"
SEED="42"

REAL_CSV="${BASE_DIR}/test_human2.csv"

AI_CSV_FILES=(
  # "${BASE_DIR}/minimax_2.6.csv"
  # "${BASE_DIR}/mureka_v9.csv"
  "${BASE_DIR}/suno_v5.5.csv"
  # "${BASE_DIR}/heartmula_music.csv"
  # "${BASE_DIR}/acestep1.5_music.csv"
  # "${BASE_DIR}/sunov5_new.csv"
  # "${BASE_DIR}/test_ace.csv"
  # "${BASE_DIR}/test_mureka.csv"
  # "${BASE_DIR}/test_sunov5.csv"
  # "${BASE_DIR}/test_diffrythm.csv"
  # "${BASE_DIR}/test_riffusion.csv"
  # "${BASE_DIR}/test_yue.csv"
  # "${BASE_DIR}/sonics_csv/test_udio-120s.csv"
  # "${BASE_DIR}/sonics_csv/test_udio-30s.csv"
  # "${BASE_DIR}/sonics_csv/test_chirp-v3.csv"
  # "${BASE_DIR}/sonics_csv/test_chirp-v3.5.csv"
  # "${BASE_DIR}/sonics_csv/test_chirp-v2-xxl-alpha.csv"
)

PAIR_LIST=(
  "0,1"
  # "2,3"
)

AI_LABEL="1"
REAL_LABEL="0"

# ==== 3. Args ====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --checkpoint) CKPT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --amp) AMP_FLAG="--amp"; shift 1;;
    --no-amp) AMP_FLAG="--no-amp"; shift 1;;
    --timeout) TIMEOUT_SECS="$2"; shift 2;;
    --output_dir) OUT_DIR="$2"; shift 2;;
    --real_csv) REAL_CSV="$2"; shift 2;;
    --metric) METRIC="$2"; shift 2;;
    --max_pairs) MAX_PAIRS="$2"; shift 2;;
    --pairs_per_ai) PAIRS_PER_AI="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --ai_label) AI_LABEL="$2"; shift 2;;
    --real_label) REAL_LABEL="$2"; shift 2;;
    --help)
      echo "Usage: $0 [options] [csv1 csv2 ...]"
      echo "Options:"
      echo "  --config PATH       Config yaml"
      echo "  --checkpoint PATH   Checkpoint"
      echo "  --device STR        Device"
      echo "  --amp               Enable AMP"
      echo "  --no-amp            Disable AMP"
      echo "  --timeout SECS      Max seconds per pair"
      echo "  --output_dir PATH   Output directory"
      echo "  --real_csv PATH     Real CSV file"
      echo "  --metric STR        cosine or l2"
      echo "  --max_pairs INT     Max pairs per CSV"
      echo "  --pairs_per_ai INT  Random pairs per AI CSV"
      echo "  --seed INT          Random seed"
      echo "  --ai_label INT      Optional AI label check"
      echo "  --real_label INT    Optional real label check"
      exit 0
      ;;
    -* ) echo "Unknown argument: $1"; exit 1;;
    * ) AI_CSV_FILES+=("$1"); shift 1;;
  esac
done

# ==== 4. Checks ====
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
LOG_PATH="${LOG_DIR}/feature_distance_${TS}.log"

[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }
[[ -f "${ANALYSIS_PY}" ]] || { echo "Script not found: ${ANALYSIS_PY}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}"; exit 1; }
[[ -f "${REAL_CSV}" ]] || { echo "Real CSV not found: ${REAL_CSV}"; exit 1; }
[[ ${#AI_CSV_FILES[@]} -gt 0 ]] || { echo "No AI CSV files provided."; exit 1; }
[[ ${#PAIR_LIST[@]} -gt 0 ]] || { echo "No pairs provided."; exit 1; }

cd "$BASE_DIR"

echo "Starting feature distance analysis" | tee -a "${LOG_PATH}"

for CSV in "${AI_CSV_FILES[@]}"; do
  [[ -f "${CSV}" ]] || { echo "AI CSV not found: ${CSV}" | tee -a "${LOG_PATH}"; continue; }

  if [[ -n "${PAIRS_PER_AI}" ]]; then
    OUT_PREFIX="${OUT_DIR}/$(basename "${CSV}" .csv)"
    cmd=(timeout "${TIMEOUT_SECS}" python "${ANALYSIS_PY}" \
      --config "${CONFIG}" \
      --checkpoint "${CKPT}" \
      --ai_csv "${CSV}" \
      --real_csv "${REAL_CSV}" \
      --pairs_per_ai "${PAIRS_PER_AI}" \
      --seed "${SEED}" \
      --metric "${METRIC}" \
      --device "${DEVICE}" \
      --plot_path "${OUT_PREFIX}" \
      ${AMP_FLAG})

    if [[ -n "${AI_LABEL}" ]]; then
      cmd+=(--ai_label "${AI_LABEL}")
    fi
    if [[ -n "${REAL_LABEL}" ]]; then
      cmd+=(--real_label "${REAL_LABEL}")
    fi

    echo "${cmd[*]}" >> "${LOG_PATH}"
    "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
  else
    pair_count=0
    for PAIR in "${PAIR_LIST[@]}"; do
      if [[ -n "${MAX_PAIRS}" && "${pair_count}" -ge "${MAX_PAIRS}" ]]; then
        break
      fi
      AI_INDEX="${PAIR%%,*}"
      REAL_INDEX="${PAIR##*,}"

      OUT_PREFIX="${OUT_DIR}/$(basename "${CSV}" .csv)_ai${AI_INDEX}_real${REAL_INDEX}"

      cmd=(timeout "${TIMEOUT_SECS}" python "${ANALYSIS_PY}" \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --ai_csv "${CSV}" \
        --real_csv "${REAL_CSV}" \
        --ai_index "${AI_INDEX}" \
        --real_index "${REAL_INDEX}" \
        --metric "${METRIC}" \
        --device "${DEVICE}" \
        --plot_path "${OUT_PREFIX}" \
        ${AMP_FLAG})

      if [[ -n "${AI_LABEL}" ]]; then
        cmd+=(--ai_label "${AI_LABEL}")
      fi
      if [[ -n "${REAL_LABEL}" ]]; then
        cmd+=(--real_label "${REAL_LABEL}")
      fi

      echo "${cmd[*]}" >> "${LOG_PATH}"
      "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
      pair_count=$((pair_count + 1))
    done
  fi

done

echo "All feature distance analyses completed." | tee -a "${LOG_PATH}"
