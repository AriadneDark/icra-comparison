#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/benchmark/compose.yaml"
ENV_FILE="$ROOT_DIR/benchmark/docker.env"
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
MANIFEST="${BENCHMARK_MANIFEST:-our_results/random_review_100_seed_20260907.json}"
SOURCE_ROOT="${BENCHMARK_SOURCE_ROOT:-our_results/scenes}"
WORK_ROOT="${BENCHMARK_WORK_ROOT:-baseline_runs}"
EVAL_ROOT="${EVAL_ROOT:-$WORK_ROOT/evaluation}"
OURS_ROOT="${BENCHMARK_OURS_ROOT:-our_results}"

if docker compose version >/dev/null 2>&1; then
  compose=(docker compose -f "$COMPOSE_FILE")
elif command -v docker-compose >/dev/null 2>&1; then
  compose=(docker-compose -f "$COMPOSE_FILE")
else
  printf '%s\n' "Docker Compose is required (docker compose v2 or docker-compose)." >&2
  exit 2
fi
if [[ -f "$ENV_FILE" ]]; then
  compose+=(--env-file "$ENV_FILE")
fi

# Bind-mounted outputs and model caches must remain writable by the host user.
# Without this, Docker's default root user leaves root-owned artifacts behind.
container_run=(run --rm --user "$(id -u):$(id -g)")

usage() {
  printf '%s\n' \
    "Usage: benchmark/docker-run.sh COMMAND [extra runner arguments]" \
    "" \
    "Commands:" \
    "  build       Build both GPU images" \
    "  smoke       Check imports and GPU visibility in both images" \
    "  prepare     Recreate/check the 100 clean MP4 inputs" \
    "  sg-ego      Batch-run adapted SG-Ego with persistent models" \
    "  svg2        Stage-wise batch-run SVG2 with persistent models" \
    "  all         Batch-run SG-Ego and then SVG2" \
    "  sg-ego-single  Legacy per-video SG-Ego runner" \
    "  svg2-single    Legacy per-video SVG2 runner" \
    "  visualize   Render OUR | SG-EGO | SVG2 comparison videos" \
    "  eval-prepare   Build candidates and choose the 80+20 human videos" \
    "  eval-vlm       Run VLM proposer/verifier over the evaluation set" \
    "  judge-start    Start local Gemma 4 31B VLM server" \
    "  judge-status   Show local judge container status" \
    "  judge-smoke    Check the local judge OpenAI-compatible endpoint" \
    "  judge-logs     Follow local judge startup/request logs" \
    "  judge-stop     Stop local judge and release the GPU" \
    "  eval-vlm-local Evaluate through the running local Gemma judge" \
    "  eval-annotate  Start blind human UI (optional annotator id)" \
    "  eval-report    Build human/calibrated/full reports" \
    "  estimate-time  Estimate a previous run from artifact timestamps" \
    "" \
    "Examples:" \
    "  benchmark/docker-run.sh sg-ego --limit 1" \
    "  benchmark/docker-run.sh svg2 --episode 28_viola/episode_15" \
    "  benchmark/docker-run.sh visualize --episode 28_viola/episode_15"
}

run_method() {
  local service="$1"
  local method="$2"
  shift 2
  "${compose[@]}" "${container_run[@]}" "$service" python \
    benchmark/run_batch_100.py \
    --manifest "$MANIFEST" \
    --source-root "$SOURCE_ROOT" \
    --work-root "$WORK_ROOT" \
    --method "$method" "$@"
}

run_single_method() {
  local service="$1"
  local method="$2"
  shift 2
  "${compose[@]}" "${container_run[@]}" "$service" python \
    benchmark/run_selected_100.py \
    --manifest "$MANIFEST" \
    --source-root "$SOURCE_ROOT" \
    --work-root "$WORK_ROOT" \
    --method "$method" "$@"
}

command="${1:-}"
if [[ -z "$command" ]]; then
  usage
  exit 2
fi
shift

case "$command" in
  build)
    "${compose[@]}" build sg-ego svg2
    ;;
  smoke)
    "${compose[@]}" "${container_run[@]}" sg-ego bash -lc \
      "nvidia-smi && python -c 'import torch, transformers, decord; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__)'"
    "${compose[@]}" "${container_run[@]}" svg2 bash -lc \
      "nvidia-smi && python -c 'import torch, transformers, cv2, decord, openai, sam2; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__)'"
    ;;
  prepare)
    run_method sg-ego prepare "$@"
    ;;
  sg-ego|svg-ego)
    run_method sg-ego sg_ego "$@"
    ;;
  svg2)
    run_method svg2 svg2 "$@"
    ;;
  sg-ego-single)
    run_single_method sg-ego sg_ego "$@"
    ;;
  svg2-single)
    run_single_method svg2 svg2 "$@"
    ;;
  all)
    run_method sg-ego sg_ego "$@"
    run_method svg2 svg2 "$@"
    ;;
  visualize)
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/visualize_comparison.py \
      --manifest "$MANIFEST" \
      --source-root "$SOURCE_ROOT" \
      --ours-root "$OURS_ROOT" \
      --sg-ego-root "$WORK_ROOT/sg_ego" \
      --svg2-root "$WORK_ROOT/svg2" \
      --output-root "$WORK_ROOT/comparisons" "$@"
    ;;
  eval-prepare)
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/prepare_hybrid_evaluation.py \
      --manifest "$MANIFEST" \
      --ours-root "$OURS_ROOT" \
      --sg-ego-root "$WORK_ROOT/sg_ego" \
      --svg2-root "$WORK_ROOT/svg2" \
      --output-root "$EVAL_ROOT" "$@"
    ;;
  eval-vlm)
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/run_vlm_evaluation.py \
      --study-root "$EVAL_ROOT" \
      --source-root "$SOURCE_ROOT" "$@"
    ;;
  judge-start)
    "${compose[@]}" up -d gemma-judge
    printf '%s\n' \
      "Gemma judge is starting. Initial model download/load can take a while." \
      "Follow it with: ./benchmark/docker-run.sh judge-logs"
    ;;
  judge-status)
    "${compose[@]}" ps gemma-judge
    ;;
  judge-smoke)
    "${compose[@]}" exec -T gemma-judge python3 -c \
      'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=30)))'
    ;;
  judge-logs)
    "${compose[@]}" logs -f gemma-judge
    ;;
  judge-stop)
    "${compose[@]}" stop gemma-judge
    ;;
  eval-vlm-local)
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/run_vlm_evaluation.py \
      --study-root "$EVAL_ROOT" \
      --source-root "$SOURCE_ROOT" \
      --base-url http://gemma-judge:8000/v1 \
      --model auto "$@"
    ;;
  eval-annotate)
    annotator="${1:-annotator1}"
    if [[ $# -gt 0 ]]; then shift; fi
    annotation_port="${ANNOTATION_PORT:-8765}"
    "${compose[@]}" run --rm --user "$(id -u):$(id -g)" \
      -p "$annotation_port:8765" svg2 python \
      benchmark/human_annotation_server.py \
      --study-root "$EVAL_ROOT" \
      --video-root "$WORK_ROOT/videos" \
      --source-root "$SOURCE_ROOT" \
      --annotator "$annotator" --port 8765 "$@"
    ;;
  eval-report)
    annotator="${1:-annotator1}"
    if [[ $# -gt 0 ]]; then shift; fi
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/report_hybrid_evaluation.py \
      --study-root "$EVAL_ROOT" \
      --annotator "$annotator" \
      --output "$EVAL_ROOT/reports/hybrid_metrics.json" "$@"
    ;;
  estimate-time)
    "${compose[@]}" "${container_run[@]}" svg2 python \
      benchmark/estimate_previous_runtime.py \
      --manifest "$MANIFEST" \
      --work-root "$WORK_ROOT" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
