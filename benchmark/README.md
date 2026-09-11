# Goal-relevant 100-video benchmark

For isolated SG-Ego/SVG2 GPU environments and exact build/run commands, see
[`DOCKER.md`](DOCKER.md).

For the 1,000-video human-calibrated protocol (100 human videos plus VLM
assessment of the remaining 900), see [`HYBRID_EVALUATION.md`](HYBRID_EVALUATION.md).

The benchmark evaluates only the induced scene graph on four roles:
`robot`, `manipulated_object`, `initial_support`, and `target`. Any node or edge
whose endpoint is outside this set is discarded before scoring.

The fixed input is `../our_results/random_review_100_seed_20260907.json`. Run both
adapted baselines with:

```bash
python3 benchmark/run_selected_100.py \
  --manifest our_results/random_review_100_seed_20260907.json \
  --source-root /path/to/all/selected/scenes \
  --work-root /path/to/baseline_runs
```

For the production 100-video run, use `benchmark/docker-run.sh`: its default
`sg-ego`, `svg2`, and `all` commands invoke `run_batch_100.py`, which keeps each
stage's models resident across videos and then releases them before the next
stage. `run_selected_100.py` remains the legacy per-video/debug runner.

`source-root` must contain `<relative_path>/images/` for every manifest episode.
Use `--method prepare` to materialize all 100 clean MP4 inputs without starting
the GPU models.
SG-Ego receives each planning goal in its captioning prompt and emits only
role-tagged triplets. SVG2 keeps native class-agnostic tracking, selects at most
one native track for each role after semantic structuring, and sends only those
tracks to its relationship stage. Its final stage-6 JSON therefore contains no
distractor tracks or distractor edges.

## Reference annotations and metrics

Precision/recall require independent human reference graphs; predictions from
our method must not be used as ground truth. Put one `task_role_graph_v1` JSON per
episode in a reference directory, named `dataset__episode.json`:

```json
{
  "schema_version": "task_role_graph_v1",
  "frame_count": 30,
  "frames": [
    {
      "frame_index": 0,
      "nodes": ["robot", "manipulated_object", "initial_support", "target"],
      "edges": [["manipulated_object", "on", "initial_support"]]
    }
  ]
}
```

Then evaluate all three methods:

```bash
python3 benchmark/evaluate.py \
  --manifest our_results/random_review_100_seed_20260907.json \
  --reference-root /path/to/human_reference \
  --ours-root our_results \
  --sg-ego-root /path/to/baseline_runs/sg_ego \
  --svg2-root /path/to/baseline_runs/svg2 \
  --output /path/to/baseline_runs/metrics.json
```

The report contains frame-micro and episode-macro precision/recall for visible
role nodes and directed `(subject role, predicate, object role)` triplets. Empty
predictions are handled explicitly. Missing predictions or references make the
command fail rather than silently changing the 100-video evaluation set.

## Three-way visualization

Render synchronized horizontal comparisons (`OUR | SG-EGO | SVG2`) with a
shared role palette, masks/boxes, planning goal, and relevant edges:

```bash
python3 benchmark/visualize_comparison.py \
  --manifest our_results/random_review_100_seed_20260907.json \
  --source-root our_results/scenes \
  --ours-root our_results \
  --sg-ego-root baseline_runs/sg_ego \
  --svg2-root baseline_runs/svg2 \
  --output-root baseline_runs/comparisons
```

Use `--episode dataset/episode_N` for one video. By default, missing baseline
outputs are an error; `--allow-missing` renders them as explicitly marked empty
panels, which is useful for checking the layout before inference finishes.
