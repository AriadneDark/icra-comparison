# Stage 3: Clip-level graph consolidation over time

This stage merges frame-level scene graphs into window-level video graphs.
For each temporal window, it grounds frame detections with SAM2, tracks objects through time with SAM2 Video, and optionally falls back to DINOv2 feature similarity when IoU matching fails, and outputs one consolidated graph per frames window.

## Usage

To ground the triplets on a single video:
```bash
python -m consolidation.main \
    --root data/example/ \
    --input-file 727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
    --frame-graphs-version v1 \
    --video-graphs-version v1
```

To ground the triplets in a set of videos:
```bash
python -m consolidation.main \
    --root data/example/ \
    --job-file data/example/input_files/windows.txt \
    --frame-graphs-version v1 \
    --video-graphs-version v1
```

The input file contains a video id and a set of frames to process:
```text
727ffce8-20ec-4111-af26-698eb306e8c7.mp4,0-9,10-19,20-29,30-39,...
...
```

## Output

For each processed video, outputs are saved to:

- `<root>/video_graphs/<version>/<video_id>.json`

Each video graph contains object labels, confidences, boxes, relation pairs, and relations.

### Schema of the consolidated graphs

```json
{
   "video_id":"770b0de4-9f4a-4c51-8b37-3c22fca3e6a6",
   "start_frame":3678,
   "end_frame":3682,
   "obj":[
      "person",
      "sofa",
      "main actor",
      // ...
   ],
   "confidence":[
      0.81,
      0.63,
      1.0,
      // ...
   ],
   "bbox":[
      [0.31, 0.08, 0.62, 0.84],
      [0.00, 0.15, 0.17, 0.90],
      [0.00, 0.00, 1.00, 1.00],
      // ...
   ],
   "frame_idx":[
      0,
      4,
      0,
      //...
   ],
   "pair":[
      [2, 4],
      [0, 1],
      [2, 3],
      //...
   ],
   "rel":[
      "hold",
      "in front of",
      "sit on",
      //...
   ],
   "history":[
      [
         {"frame_idx": 0, "label": "person", "obj_idx": 0},
         {"frame_idx": 0, "label": "person", "obj_idx": 0},
         {"frame_idx": 4, "label": "person", "obj_idx": 0}
      ],
      // ...
   ]
}
```

- `video_id`: unique identifier of the Ego4d video.
- `start_frame` and `end_frame`: frame boundaries of the consolidation window.
- `obj`: labels of the graph nodes.
- `bbox`: bounding boxes of the graph nodes, in `xyxy` format at the frame stored in `frame_idx`.
- `confidence`: confidence associated to the bounding boxes of the graph nodes at the frame stored in `frame_idx`..
- `pair`: graph edges encoded in the `(source, target)` format.
- `rel`: labels of the graph edges.
- `history`: this tracks the consolidation history of the graph objects, mapping each consolidated node back to its original source nodes in the frame-level graphs.
