#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/sharerobot_sg/unified_scene_graph_pipeline"
RUNTIME_ENV="${SEGMENTATION_ENV:-/datasets/goal_guided_segmentation.runtime.env}"

if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing SEGMENTATION_ENV file: $RUNTIME_ENV" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
set +a
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set in $RUNTIME_ENV" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"
variants=(qwen_text_sam3 qwen_box_sam3 qwen_box_sam2)
for variant in "${variants[@]}"; do
  config="/pipeline/configs/ablations/${variant}.json"
  log="$RUN_ROOT/logs/${variant}.log"
  echo "Running $variant; log: $log"
  /usr/bin/time -v "$PIPELINE_DIR/run.sh" batch \
    --stage sam \
    --manifest /runs/human_100_manifest.json \
    --source-root /datasets \
    --output-root "/runs/$variant" \
    --config "$config" 2>&1 | tee "$log"
done
