#!/bin/bash
# Output images:
# - ${OUTPUT_DIR}/<csv>_moe_weights.png
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
ANALYSIS_PY="${BASE_DIR}/analysis/analyze_moe.py"
CONFIG="${BASE_DIR}/config/moe.yaml"
CKPT="${BASE_DIR}/checkpoint/ckpt.pt"
BATCH_SIZE="32"
NUM_WORKERS="32"
DEVICE="cuda"
AMP_FLAG="--no-amp"
TIMEOUT_SECS="7200"
OUTPUT_DIR="${BASE_DIR}/analysis/outputs/moe"
LOG_DIR="${BASE_DIR}/logs"
MAX_SAMPLES="500"

CSV_FILES=(
    # "${BASE_DIR}/minimax_2.6.csv"
    # "${BASE_DIR}/mureka_v9.csv"
    # "${BASE_DIR}/suno_v5.5.csv"
    # "${BASE_DIR}/heartmula_music.csv"
    # "${BASE_DIR}/acestep1.5_music.csv"
    # "${BASE_DIR}/sunov5_new.csv"
    # "${BASE_DIR}/test_ace.csv"
    # "${BASE_DIR}/test_mureka.csv"
    # "${BASE_DIR}/suno_v4.csv"
    # "${BASE_DIR}/test_diffrythm.csv"
    # "${BASE_DIR}/test_riffusion.csv"
    # "${BASE_DIR}/test_yue.csv"
    # "${BASE_DIR}/test_human2.csv"
    # "${BASE_DIR}/sonics_csv/udio-120s.csv"
    # "${BASE_DIR}/sonics_csv/udio-30s.csv"
    # "${BASE_DIR}/sonics_csv/chirp-v3.csv"
    # "${BASE_DIR}/sonics_csv/chirp-v3.5.csv"
    # "${BASE_DIR}/sonics_csv/chirp-v2-xxl-alpha.csv"
    # Add more files here if needed:
    # "${BASE_DIR}/sonics_human2_all_f1.csv"
    # "${BASE_DIR}/test_mom_human2.csv"
    # "${BASE_DIR}/mood_human2_all_f1.csv"
    "${BASE_DIR}/mood_human2_all_f1.csv"
    "${BASE_DIR}/mood_human2_all_f1.csv"
)

# ==== 3. Args ====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --checkpoint) CKPT="$2"; shift 2;;
    --batch_size|--batch-size) BATCH_SIZE="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --amp) AMP_FLAG="--amp"; shift 1;;
    --no-amp) AMP_FLAG="--no-amp"; shift 1;;
    --timeout) TIMEOUT_SECS="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --max_samples) MAX_SAMPLES="$2"; shift 2;;
    --help)
      echo "Usage: $0 [options] [csv1 csv2 ...]"
      echo "Options:"
      echo "  --config PATH       Config yaml"
      echo "  --checkpoint PATH   Checkpoint"
      echo "  --batch_size INT    Batch size"
      echo "  --num_workers INT   Num workers"
      echo "  --device STR        Device"
      echo "  --amp               Enable AMP"
      echo "  --no-amp            Disable AMP"
      echo "  --timeout SECS      Max seconds per CSV"
      echo "  --output_dir PATH   Output directory"
      echo "  --max_samples INT   Max samples per CSV"
      exit 0
      ;;
    -* ) echo "Unknown argument: $1"; exit 1;;
    * ) CSV_FILES+=("$1"); shift 1;;
  esac
done

# ==== 4. Checks ====
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_PATH="${LOG_DIR}/analyze_moe_${TS}.log"

[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }
[[ -f "${ANALYSIS_PY}" ]] || { echo "Script not found: ${ANALYSIS_PY}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}"; exit 1; }
[[ ${#CSV_FILES[@]} -gt 0 ]] || { echo "No CSV files provided."; exit 1; }

cd "$BASE_DIR"

echo "Starting MOE dataset analysis for ${#CSV_FILES[@]} files" | tee -a "${LOG_PATH}"

total=${#CSV_FILES[@]}
count=0
for CSV in "${CSV_FILES[@]}"; do
  count=$((count + 1))
  echo "--------------------------------------------------------" | tee -a "${LOG_PATH}"
  echo "Processing [${count}/${total}]: ${CSV}" | tee -a "${LOG_PATH}"
  [[ -f "${CSV}" ]] || { echo "CSV not found: ${CSV}" | tee -a "${LOG_PATH}"; continue; }

  cmd=(timeout "${TIMEOUT_SECS}" python "${ANALYSIS_PY}" \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    dataset \
    --csv "${CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    ${AMP_FLAG})

  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi

  echo "${cmd[*]}" >> "${LOG_PATH}"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
  echo "Finished: ${CSV}" | tee -a "${LOG_PATH}"
done

echo "All MOE analyses completed." | tee -a "${LOG_PATH}"
