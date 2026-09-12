# Docker: SG-Ego and SVG2 benchmark

The benchmark uses two images because the upstream projects require different
PyTorch, CUDA, and Transformers versions:

- `goal-role-sg-ego:cu126`: SG-Ego, CUDA 12.6 / PyTorch 2.6;
- `goal-role-svg2:cu128`: SVG2 inference pipeline, CUDA 12.8 / PyTorch 2.8.

Both containers receive the video and planning goal. Their final graphs are
restricted to `robot`, `manipulated_object`, `initial_support`, and `target`.

## 1. Host requirements

- Linux x86-64 with an NVIDIA GPU and a driver supporting CUDA 12.8;
- Docker Engine;
- Docker Compose v2 (`docker compose`) or legacy Compose v1
  (`docker-compose`);
- NVIDIA Container Toolkit configured for Docker;
- enough free disk space for two images and model caches (plan for tens of GB).

Configure the NVIDIA runtime once on the host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Run all commands below from the repository root.

Large input datasets do not need to be copied into the repository. Set
`BENCHMARK_HOST_SOURCE_ROOT` to the host directory in `benchmark/docker.env`;
Compose mounts it read-only at `/input/scenes`. Then export
`BENCHMARK_SOURCE_ROOT=/input/scenes` before using `docker-run.sh`.

## 2. Tokens and cache paths

```bash
cp benchmark/docker.env.example benchmark/docker.env
chmod 600 benchmark/docker.env
```

Edit `benchmark/docker.env`:

```dotenv
HF_TOKEN=hf_your_token
SVG2_API_KEY=your_provider_key
SVG2_API_KEY_ENV=SVG2_API_KEY
SVG2_API_BASE_URL=REPLACE_WITH_OPENAI_COMPATIBLE_BASE_URL
SVG2_API_MODEL=Qwen/Qwen3.8-27B
SVG2_API_CONCURRENCY=4
NVIDIA_VISIBLE_DEVICES=all
HF_CACHE_DIR=./.cache/huggingface
TORCH_CACHE_DIR=./.cache/torch
```

`HF_TOKEN` is used to download gated Hugging Face models. SVG2 sends semantic
structuring, role selection, and relation generation to the configured
OpenAI-compatible API. `Qwen3.8-27B` is the default recommendation because all
three stages can share one multimodal model. Use the exact base URL and model id
shown by your provider. The local `docker.env` and model cache directories are
ignored by Git and the Docker build context.

The API settings imported from
`sharerobot_sg/unified_scene_graph_pipeline/config.json` are:

```dotenv
SVG2_API_BASE_URL=REPLACE_WITH_OPENAI_COMPATIBLE_BASE_URL
SVG2_API_MODEL=Qwen/Qwen3.8-27B
```

Put the bearer token issued for that endpoint in `SVG2_API_KEY`. Qwen is an
external dependency of this benchmark and is not started or hosted by this
Compose project.

To select a single GPU, set `NVIDIA_VISIBLE_DEVICES=0` (or another GPU index).

## 3. Build and verify

Build both images:

```bash
./benchmark/docker-run.sh build
```

The first build downloads large CUDA/PyTorch dependencies. Verify imports and
GPU visibility afterward:

```bash
./benchmark/docker-run.sh smoke
```

Equivalent direct Compose commands are:

```bash
docker compose --env-file benchmark/docker.env \
  -f benchmark/compose.yaml build
docker compose --env-file benchmark/docker.env \
  -f benchmark/compose.yaml run --rm sg-ego nvidia-smi
```

Replace `docker compose` with `docker-compose` on hosts using Compose v1.

## 4. Prepare inputs

The repository currently contains the prepared 100-video set under
`baseline_runs/videos`. To recreate or validate it from the source frames:

```bash
./benchmark/docker-run.sh prepare
```

The fixed selection comes from
`our_results/random_review_100_seed_20260907.json`.

## 5. Run baselines

Start with one episode to validate credentials, model access, GPU memory, and
output formats:

```bash
./benchmark/docker-run.sh sg-ego \
  --episode 28_viola/episode_15

./benchmark/docker-run.sh svg2 \
  --episode 28_viola/episode_15
```

Test the first five manifest entries:

```bash
./benchmark/docker-run.sh sg-ego --limit 5
./benchmark/docker-run.sh svg2 --limit 5
```

Run all selected videos sequentially:

```bash
./benchmark/docker-run.sh all
```

These commands use the stage-wise batch runner by default. SG-Ego loads its
caption model once, GroundingDINO once, and the consolidation models once for
the selected set. SVG2 processes all selected videos through stage 1, releases
that stage's resources, then advances to stage 2, and so on. This prevents both
per-video model reloads and keeping every SVG2 model in GPU memory at once.

Independent SVG2 API requests use bounded concurrency controlled by
`SVG2_API_CONCURRENCY` (default `4`). Reduce it to `1` if the endpoint has a
strict rate limit. Model ids, prompts, image sampling, thresholds, and output
schemas are unchanged.

Or launch each method separately:

```bash
./benchmark/docker-run.sh sg-ego
./benchmark/docker-run.sh svg2
```

The runner is resumable: already complete per-episode artifacts are skipped.
Outputs are written to:

- `baseline_runs/sg_ego/`;
- `baseline_runs/svg2/`.

Model weights persist under `benchmark/.cache/`, so they are not downloaded on
every container run. The repository is bind-mounted at `/workspace`, therefore
outputs also persist after a container exits.

For debugging, the previous per-video behavior remains available as
`sg-ego-single` and `svg2-single`.

`docker-run.sh` starts containers with the current host UID/GID. Generated
artifacts and cache files therefore remain owned by the invoking user and do
not require root privileges. If an older run already created a root-owned
cache, point `HF_CACHE_DIR` and `TORCH_CACHE_DIR` in `docker.env` at new
user-owned directories instead of trying to modify the old cache.

## 6. Local Gemma 4 evaluation judge (A100 80 GB)

The evaluation judge is a separate vLLM service and is not used to generate the
three methods' graphs. Stop/finish SG-Ego and SVG2 first so the judge has the
GPU to itself. Defaults in `benchmark/docker.env.example` select
`google/gemma-4-31B-it`, BF16, an 8192-token context, one sequence, and at most
10 input images per request.

```bash
./benchmark/docker-run.sh judge-start
./benchmark/docker-run.sh judge-logs
```

On first start, vLLM and about 63 GB of weights are downloaded. Press `Ctrl-C`
after the server becomes ready; this exits the log viewer without stopping the
service. Then run:

```bash
./benchmark/docker-run.sh judge-smoke
./benchmark/docker-run.sh eval-vlm-local --limit 1 --workers 1
./benchmark/docker-run.sh eval-vlm-local --workers 1
./benchmark/docker-run.sh judge-stop
```

No real API key is used. The model cache is written as the invoking host user.
The evaluation runner discovers the model id from `/v1/models` and records it
in every result. Do not use more than one evaluation worker with this BF16 model
on one A100 80 GB.

Useful management commands:

```bash
./benchmark/docker-run.sh judge-status
./benchmark/docker-run.sh judge-logs
./benchmark/docker-run.sh judge-stop
```

If model loading is out of memory, set
`GEMMA_JUDGE_MODEL=google/gemma-4-26B-A4B-it` in `benchmark/docker.env` and run
`judge-start` again. Changing the judge invalidates existing VLM judgments; use
`--overwrite` only when intentionally regenerating the full set.

## 7. Three-panel visualization

After both baselines finish, render synchronized `OUR | SG-EGO | SVG2` videos:

```bash
./benchmark/docker-run.sh visualize
```

For one episode:

```bash
./benchmark/docker-run.sh visualize \
  --episode 28_viola/episode_15
```

Videos are written to `baseline_runs/comparisons/`. Add `--allow-missing` to
inspect the layout before every baseline result is available.

## 8. Precision and recall

Metrics require independent human reference graphs. Once they are available,
run the evaluator in the SVG2 image (replace the reference path):

```bash
docker compose --env-file benchmark/docker.env \
  -f benchmark/compose.yaml run --rm svg2 \
  python benchmark/evaluate.py \
  --manifest our_results/random_review_100_seed_20260907.json \
  --reference-root reference_graphs \
  --ours-root our_results \
  --sg-ego-root baseline_runs/sg_ego \
  --svg2-root baseline_runs/svg2 \
  --output baseline_runs/metrics.json
```

The evaluator scores only the four task roles and edges between them; distractor
objects are discarded. It fails on missing reference or prediction files rather
than silently evaluating fewer than 100 videos.

## Common failures

- `unknown or invalid runtime name: nvidia`: configure NVIDIA Container Toolkit
  and restart Docker as shown above.
- `CUDA driver version is insufficient`: update the host NVIDIA driver; the
  driver is not installed inside these images.
- Hugging Face `401/403`: check `HF_TOKEN` and accept any gated-model licence on
  the model page.
- `SVG2_API_KEY is not set`: add the provider key to `benchmark/docker.env` and
  keep `SVG2_API_KEY_ENV=SVG2_API_KEY`.
- CUDA out of memory: stop other GPU jobs and test one episode at a time. The
  SG-Ego 9B model generally needs a high-memory GPU.
