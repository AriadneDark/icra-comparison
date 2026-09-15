# Object-only ablation on the human-100 split

The ablation changes only role-object segmentation/tracking. Relation metrics are
therefore intentionally excluded. The evaluated roles are `robot`,
`manipulated_object`, `initial_support`, and `target`.

## Variants

| ID | SAM input | Segmenter | Missing-frame repair |
|---|---|---|---|
| `full` | Qwen-derived text, Qwen boxes select the instance | SAM3 | SAM2.1 from Qwen boxes |
| `qwen_text_sam3` | Qwen-derived text, Qwen boxes select the instance | SAM3 | none |
| `qwen_box_sam3` | Qwen box, no SAM3 text prompt | SAM3 | none |
| `qwen_box_sam2` | Qwen boxes | SAM2.1 | none (SAM2.1 is the primary tracker) |
| `qwen_text_sam31` | Qwen-derived text | SAM3.1 | none |
| `qwen_text_sam31_sam2` | Qwen-derived text | SAM3.1 | SAM2.1 from Qwen boxes |

`full` is reused and must not be recomputed. All other variants reuse its frozen
Qwen `task_spec.json`; this prevents nondeterministic Qwen calls from confounding
the segmenter comparison.

SAM3.1 is a separate integration target. It requires the current upstream SAM3
source, the gated `facebook/sam3.1` checkpoint, and the multiplex predictor API.
It must not be represented by merely changing the checkpoint path in the SAM3
configuration.

## Prepare the three currently supported variants

Set `RUN_ROOT` in the protected segmentation runtime env to the desired ablation
directory. Then create hard-linked inputs (ordinary copies are used across
filesystems):

```bash
python3 benchmark/prepare_object_ablations.py \
  --study-root /home/zimina_sv/scene_graphs_icra/baseline_runs/evaluation_1000 \
  --full-root /home/zimina_sv/scene_graphs_icra/sharerobot_manipulation_gt_1000_sgg_v1 \
  --run-root /home/zimina_sv/object_ablation_runs
```

The Qwen server is not needed because `task_spec.json` is reused. Build the core
image after adding the configs, then run all variants sequentially:

```bash
cd sharerobot_sg/unified_scene_graph_pipeline
export SEGMENTATION_ENV=/datasets/goal_guided_segmentation.runtime.env
./run.sh build
cd ../..
./benchmark/run_object_ablations.sh
```

Outputs and timing logs are written below `$RUN_ROOT`; rerunning the command is
resumable because completed SAM stages are fingerprinted.

## Metrics

Count role predictions per video, role, and frame:

- TP: the role is human-visible and the predicted track follows the correct
  physical object;
- FP: a predicted track is present but follows the wrong object, or predicts a
  role that is not human-visible;
- FN: a human-visible role lacks a correct predicted track;
- precision = TP / (TP + FP);
- recall = TP / (TP + FN).

Report micro precision/recall/F1, macro video-level precision/recall/F1, per-role
scores, correct-track frame coverage, runtime, and peak GPU memory. Use paired
video bootstrap confidence intervals for differences relative to `full`.

The existing human role visibility intervals can be reused. New variant tracks
still need a blind correctness audit because a mask assigned the right role name
may follow the wrong physical instance. Relations and canonical role identities
must not be annotated again.

After the lightweight track audit has produced one JSON file per human video,
run:

```bash
PYTHONPATH=benchmark python3 benchmark/evaluate_object_ablations.py \
  --study-root /home/zimina_sv/scene_graphs_icra/baseline_runs/evaluation_1000 \
  --full-root /home/zimina_sv/scene_graphs_icra/sharerobot_manipulation_gt_1000_sgg_v1 \
  --run-root /home/zimina_sv/object_ablation_runs \
  --audit-root /home/zimina_sv/object_ablation_runs/track_audits \
  --annotator annotator1 \
  --output /home/zimina_sv/object_ablation_runs/object_metrics.json
```

For `full`, the evaluator reuses the existing blind `ours` track verdicts. For
new variants, an audit entry has this shape:

```json
{
  "tracks": {
    "qwen_text_sam3": {
      "robot": {
        "verdict": "yes",
        "correct_intervals": [[0, 29]],
        "uncertain_intervals": []
      }
    }
  }
}
```
