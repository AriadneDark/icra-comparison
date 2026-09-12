# Human-calibrated evaluation for 1,000 videos

This evaluation compares `OUR`, adapted `SG-Ego`, and adapted `SVG2` using only
`robot`, `manipulated_object`, `initial_support`, and `target`. It produces three
separate result blocks:

1. human-primary metrics on 80 proportionally stratified videos;
2. human-challenge metrics on 20 high-disagreement videos;
3. human-calibrated VLM estimates on the remaining 900 videos and a combined
   full-set report.

The VLM is also run on the human subset. Those overlapping decisions estimate
`P(human true | VLM verdict, fact type)`, which is then used instead of treating
raw VLM confidence as calibrated probability.

## 1. Configure the 1,000-video run

The three methods must have finished on the same manifest. Point the Docker
wrapper at that manifest and its image/video roots:

```bash
export BENCHMARK_MANIFEST=our_results/random_review_1000.json
export BENCHMARK_SOURCE_ROOT=our_results/scenes
export BENCHMARK_WORK_ROOT=baseline_runs
export EVAL_ROOT=baseline_runs/evaluation_1000
```

If the baseline outputs have not been generated yet, the same batch runner now
accepts manifests of arbitrary size:

```bash
./benchmark/docker-run.sh all --expected-count 1000
```

The recommended independent judge on an A100 80 GB is the local
`google/gemma-4-31B-it` service described below. A hosted independent judge can
instead be set in `benchmark/docker.env`:

```dotenv
EVAL_VLM_API_KEY=...
EVAL_VLM_BASE_URL=https://provider.example/v1
EVAL_VLM_MODEL=independent-multimodal-model
```

If these values are blank, the runner falls back to `SVG2_API_*`. This is useful
for debugging but not ideal for the final paper because SVG2 and SG-Ego already
use Qwen-family models. Every VLM artifact records whether the configured judge
is independent of those Qwen generators.

## 2. Freeze candidates and select human videos

```bash
./benchmark/docker-run.sh eval-prepare
```

This command:

- normalizes all three output formats;
- maps predicate aliases through `benchmark/predicate_ontology.json`;
- collapses frame facts into inclusive temporal intervals;
- creates the blind union of method claims;
- computes three-way disagreement;
- selects 80 stratified primary, 20 challenge, and 25 double-annotation videos.

Outputs are under `$EVAL_ROOT`:

```text
study_manifest.json
candidates/<video_id>.json
human_annotations/
vlm/
vlm_references/
reports/
```

The selection is deterministic (`--seed 20260911`). To change sizes:

```bash
./benchmark/docker-run.sh eval-prepare \
  --human-size 100 --challenge-size 20 --double-annotation-size 25 \
  --seed 20260911
```

Do not regenerate the study manifest after annotation begins.

## 3. Run the VLM proposer and verifier

Make sure SG-Ego and SVG2 have exited before starting the judge: Gemma 4 31B in
BF16 uses most of the A100 80 GB. Start the pinned vLLM server and follow its
startup logs:

```bash
./benchmark/docker-run.sh judge-start
./benchmark/docker-run.sh judge-logs
```

The first run downloads roughly 63 GB of model weights into `HF_CACHE_DIR`.
After vLLM reports that the server is ready, leave the log view with `Ctrl-C`;
the server keeps running. Check its model endpoint and run one video:

```bash
./benchmark/docker-run.sh judge-smoke
./benchmark/docker-run.sh eval-vlm-local --limit 1 --workers 1
```

To time exactly one representative proposer request without creating a study or
annotations, point the standalone benchmark at an MP4 or a frame directory:

```bash
./benchmark/docker-run.sh judge-benchmark \
  --video /workspace/baseline_runs/videos/28_viola__episode_15.mp4 \
  --goal "Pick up the object and place it on the target"
```

It samples 10 frames and reports model discovery, preprocessing, API inference,
total latency, token counts, and output tokens/second. The first request may
include CUDA graph/kernel warm-up; repeat the command to measure warm latency.
Each invocation still sends exactly one chat-completions request.

Then process all videos:

```bash
./benchmark/docker-run.sh eval-vlm-local --workers 1
./benchmark/docker-run.sh judge-stop
```

`eval-vlm-local` discovers the exact served model id and needs no real API key.
Keep one worker on a single 80 GB GPU. The local defaults can be changed through
the `GEMMA_JUDGE_*` values in `benchmark/docker.env`. If BF16 model loading runs
out of memory, set `GEMMA_JUDGE_MODEL=google/gemma-4-26B-A4B-it`, restart the
judge, and rerun the one-video check. Do not switch models in the middle of a
study unless all old VLM outputs are removed or deliberately regenerated with
`--overwrite`.

Each video uses two calls:

1. a method-independent proposer sees sampled raw frames and proposes role and
   relation facts separately for the frame indices it actually sees;
2. a blind verifier sees the union of proposer and method facts, without method
   names; candidate track boxes are drawn on selected evidence frames using only
   anonymous claim ids. For every claim it returns `yes`, `no`, or `uncertain`
   independently at every displayed frame.

The judge does **not** reconstruct intervals between sampled frames. With 30
prepared frames and `--max-frames 10`, its temporal reference contains only the
10 selected checkpoints. The other 20 frames are unknown and excluded from VLM
metrics. This prevents an event visible at frame 10 and absent at frame 20 from
being arbitrarily assigned to every frame in between.

Existing `vlm/<video_id>.json` and `vlm_references/<video_id>.json` files are
skipped only when they use the current `hybrid_vlm_assessment_v2` schema. Older
interval-inference outputs are stale and are regenerated automatically. Use
`--overwrite` to intentionally rerun current-schema results. `--max-frames 10`
controls the number of evaluated temporal checkpoints.

## 4. Human annotation

Start the first annotator UI:

```bash
./benchmark/docker-run.sh eval-annotate annotator1
```

Open `http://SERVER:8765`. Over SSH, tunnel the port:

```bash
ssh -L 8765:localhost:8765 USER@SERVER
```

Then open `http://localhost:8765` locally.

The annotator must:

- identify the canonical visible object for all four roles;
- mark every blind candidate claim `yes`, `no`, or `uncertain`;
- correct relation intervals;
- add relations missed by the entire candidate pool;
- mark the task complete.

For the second annotator, use another port and only the frozen 25-video overlap:

```bash
ANNOTATION_PORT=8766 ./benchmark/docker-run.sh \
  eval-annotate annotator2 --only-double
```

Annotations are written atomically to:

```text
$EVAL_ROOT/human_annotations/<annotator>/<video_id>.json
```

## 5. Build reports

```bash
./benchmark/docker-run.sh eval-report annotator1 \
  --second-annotator annotator2
```

Outputs:

```text
$EVAL_ROOT/reports/hybrid_metrics.json
$EVAL_ROOT/reports/hybrid_metrics.csv
```

The JSON contains:

- `human_metrics.human_primary`: unbiased headline scores;
- `human_metrics.human_challenge`: difficult-case diagnostics;
- `human_sampled_metrics`: exact human truth restricted to judge checkpoints;
- `vlm_calibrated_metrics`: estimated checkpoint metrics for VLM-only videos;
- `full_hybrid_metrics`: exact human checkpoint counts plus calibrated VLM
  checkpoint counts on the same temporal basis;
- `candidate_pool_coverage_on_human`: separate role and relation coverage;
- per-verdict calibration counts and VLM/human confusion;
- bootstrap 95% intervals for human subsets;
- inter-annotator agreement and Cohen's kappa.

Do not present `vlm_calibrated_metrics` as direct ground-truth precision/recall.
For the primary scientific claim, report the human-primary result and its paired
video-level confidence intervals. The full-set calibrated result is supporting
evidence and describes sampled checkpoints, not full video intervals. Exact
interval quality is reported only from human annotations. If candidate-pool
coverage is low, improve the independent proposer or increase the human subset
before interpreting VLM recall.
