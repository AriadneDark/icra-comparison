# Stage 2: Frame-level graph grounding

This stage converts frame captions (triplets) into grounded frame-level scene graphs.
Triplets from the captioning stage are grounded in the corresponding frame with GroundingDINO and the converted to a frame-level scene graph.

## Usage

To ground the triplets on a single video:
```bash
python -m grounding.main \
    --root data/example/ \
    --input-file 727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
    --captions-version qwen3.5_9b \
    --frame-graphs-version v1
```

To ground the triplets in a set of videos:
```bash
python -m grounding.main \
    --root data/example/ \
    --job-file data/example/input_files/frames.txt \
    --captions-version qwen3.5_9b \
    --frame-graphs-version v1
```

The input file contains a video id and a set of frames to process:
```text
727ffce8-20ec-4111-af26-698eb306e8c7.mp4,0,1,2,3,4,5,6,7,...
...
```

## Output

For each processed video, outputs are saved to:

- `<root>/frame_graphs/<version>/<video_id>.json`
- `<root>/frame_graphs/<version>/<video_id>_mapped.json`

Each frame graph contains object labels, confidences, boxes, relation pairs, and relations.

### Schema of the frame graphs
```json
{
   "video_id": "770b0de4-9f4a-4c51-8b37-3c22fca3e6a6",
   "frame_id": 3678,
   "obj": [
      "person",
      "sofa",
      "main actor"
   ],
   "rel": [
      "in front of",
      "sit on"
   ],
   "confidence": [
      0.48,
      0.22,
      1.00
   ],
   "bbox": [
      [0.31, 0.08, 0.62, 0.84],
      [0.02, 0.60, 0.34, 0.92],
      [0.00, 0.00, 1.00, 1.00]
   ],
   "pair": [
      [0, 1],
      [2, 1]
   ],
   // ...
}
```

- `video_id`: unique identifier of the Ego4d video.
- `frame_id`: frame index at 5 FPS.
- `obj`: labels of the graph nodes.
- `bbox`: bounding boxes of the graph nodes, in `xyxy` format.
- `confidence`: confidence associated to the bounding boxes of the graph nodes.
- `pair`: graph edges encoded in the `(source, target)` format.
- `rel`: labels of the graph edges.