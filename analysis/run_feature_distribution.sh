#!/bin/bash
# Output images:
# - ${OUTPUT_DIR}/feature_dist_<stage>_<method>.png
# - If stage != fused: feature_dist_<stage>_<branch>_<method>.png

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
ANALYSIS_PY="${BASE_DIR}/analysis/analyze_feature_distribution.py"
CONFIG="${BASE_DIR}/config/moe.yaml"
CKPT="${BASE_DIR}/outputs/checkpoints/ckpt.pt"
BATCH_SIZE="32"
NUM_WORKERS="32"
DEVICE="cuda"
AMP_FLAG="--no-amp"
TIMEOUT_SECS="7200"
OUTPUT_DIR="${BASE_DIR}/analysis/outputs/feature_dist"
LOG_DIR="${BASE_DIR}/logs"
METHOD="umap"   # pca, tsne, or umap
MAX_SAMPLES="500"
STAGE="projected"  # fused, encoder, projected, or expert
BRANCH=""
UMAP_MIN_DIST="0.2"
UMAP_N_NEIGHBORS="80"
UMAP_METRIC="cosine"
UMAP_SPREAD="1.0"
  
# BRANCH is the default value for --branch.
# Only needed when STAGE is not fused to select a branch (muq, mert, wave, rawnet, fxpp).

CSV_FILES=(
    "${BASE_DIR}/minimax_2.6.csv"
    "${BASE_DIR}/mureka_v9.csv"
    "${BASE_DIR}/suno_v5.5.csv"
    "${BASE_DIR}/heartmula_music.csv"
    "${BASE_DIR}/acestep1.5_music.csv"
    "${BASE_DIR}/sunov5_new.csv"
    "${BASE_DIR}/test_ace.csv"
    "${BASE_DIR}/test_mureka.csv"
    "${BASE_DIR}/suno_v4.csv"
    "${BASE_DIR}/test_human2.csv"
    "${BASE_DIR}/sonics_csv/udio-120s.csv"
    "${BASE_DIR}/sonics_csv/udio-30s.csv"
    "${BASE_DIR}/sonics_csv/chirp-v3.csv"
    "${BASE_DIR}/sonics_csv/chirp-v3.5.csv"
    "${BASE_DIR}/sonics_csv/chirp-v2-xxl-alpha.csv"
    # Add more files here if needed:
    # "${BASE_DIR}/sonics_human2_all_f1.csv"
    # "${BASE_DIR}/test_mom_human2.csv"
    # "${BASE_DIR}/mood_human2_all_f1.csv"
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
    --method) METHOD="$2"; shift 2;;
    --max_samples) MAX_SAMPLES="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --umap_min_dist) UMAP_MIN_DIST="$2"; shift 2;;
    --umap_n_neighbors) UMAP_N_NEIGHBORS="$2"; shift 2;;
    --umap_metric) UMAP_METRIC="$2"; shift 2;;
    --umap_spread) UMAP_SPREAD="$2"; shift 2;;
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
      echo "  --timeout SECS      Max seconds per run"
      echo "  --output_dir PATH   Output directory"
      echo "  --method STR        pca, tsne, or umap"
      echo "  --max_samples INT   Max samples per dataset"
      echo "  --stage STR         fused|encoder|projected|expert"
      echo "  --branch STR        Branch name for non-fused stages"
      echo "  --umap_min_dist F   UMAP min_dist (default: 0.1)"
      echo "  --umap_n_neighbors N  UMAP n_neighbors (default: 30)"
      echo "  --umap_metric STR     UMAP metric (default: cosine)"
      echo "  --umap_spread F       UMAP spread (default: 1.0)"
      exit 0
      ;;
    -* ) echo "Unknown argument: $1"; exit 1;;
    * ) CSV_FILES+=("$1"); shift 1;;
  esac
done

# ==== 4. Checks ====
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_PATH="${LOG_DIR}/feature_dist_${TS}.log"

[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }
[[ -f "${ANALYSIS_PY}" ]] || { echo "Script not found: ${ANALYSIS_PY}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}"; exit 1; }
[[ ${#CSV_FILES[@]} -gt 0 ]] || { echo "No CSV files provided."; exit 1; }

cd "$BASE_DIR"

echo "Starting feature distribution analysis" | tee -a "${LOG_PATH}"

cmd=(timeout "${TIMEOUT_SECS}" python "${ANALYSIS_PY}" \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --method "${METHOD}" \
  --max_samples "${MAX_SAMPLES}" \
  --stage "${STAGE}" \
  --umap_min_dist "${UMAP_MIN_DIST}" \
  --umap_n_neighbors "${UMAP_N_NEIGHBORS}" \
  --umap_metric "${UMAP_METRIC}" \
  --umap_spread "${UMAP_SPREAD}" \
  ${AMP_FLAG})

if [[ -n "${BRANCH}" ]]; then
  cmd+=(--branch "${BRANCH}")
fi

for CSV in "${CSV_FILES[@]}"; do
  cmd+=(--csv "${CSV}")
done

echo "${cmd[*]}" >> "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "Feature distribution analysis completed." | tee -a "${LOG_PATH}"
