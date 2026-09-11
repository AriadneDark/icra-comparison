"""Consolidate frame scene graphs over temporal windows.

Notes
-----
This module reads per-frame scene-graph predictions, tracks objects across a
temporal window with SAM2 video tracking, and emits one consolidated graph per
window.

The consolidated graph contains a canonical object list, relation triplets, and
track histories aligned to the input window.

Expected directory layout under ``--root``::

    videos/<video_id>.mp4
    frame_graphs/<graphs_version>/<video_id>/graph_frame_XXXXXX.json

Outputs are written under::

    video_graphs/<graphs_output_version>/<video_id>.json
"""

import logging
from collections import Counter, namedtuple
from typing import Optional

import decord
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from transformers import (Sam2Model, Sam2Processor, Sam2VideoModel,
                          Sam2VideoProcessor)

from consolidation.dinov2_fe import DinoV2FE

decord.bridge.set_bridge("torch")

from accelerate import Accelerator

from consolidation.utils import compute_iou_batch, unbatch

torch.set_grad_enabled(False)

# Data type for the history entries in the consolidation process.
# Each history entry refers to a specific instance of an object in a frame, along with its label and confidence score.
HistoryEntry = namedtuple("HistoryEntry", ["frame_idx", "obj_idx_in_frame", "name", "confidence_score"])

device = Accelerator().device

logger = logging.getLogger(__name__)


def frame_level_grounding(
    video_frames: torch.Tensor,
    frame_object_bboxes: list[dict],
    grounding_model: Sam2Model,
    grounding_processor: Sam2Processor,
    frame_object_names: list[dict] | None = None,
) -> tuple[list[Optional[torch.Tensor]], list[list[float]]]:
    """Ground all objects in a window with the SAM2 image model.

    Parameters
    ----------
    video_frames : torch.Tensor
        Tensor with shape ``(T, C, H, W)`` for the current temporal window.
    frame_object_bboxes : list[dict]
        Per-frame mapping from local object index to bounding box coordinates
        ``[x1, y1, x2, y2]``.
    grounding_model : Sam2Model
        SAM2 image model used to convert boxes into masks.
    grounding_processor : Sam2Processor
        Processor used to prepare grounding inputs and decode outputs.
    frame_object_names : list[dict] | None, optional
        Per-frame mapping from local object index to object label. When
        provided, objects labeled ``"main actor"`` bypass SAM2 grounding and
        receive a zero mask with confidence ``1.0``.

    Returns
    -------
    tuple[list[Optional[torch.Tensor]], list[list[float]]]
        A pair ``(frame_grounding_masks, frame_grounding_scores)`` where each
        element is indexed by frame. Mask entries are boolean tensors with shape
        ``(num_objects, H, W)`` or ``None`` when a frame has no objects. Score
        entries are serialized as Python lists for easier downstream writing.

    Notes
    -----
    The ``main actor`` shortcut avoids running SAM2 on the actor track while
    still marking it as grounded.
    """
    frame_grounding_masks: list[Optional[torch.Tensor]] = [None] * len(video_frames)
    frame_grounding_scores: list[Optional[torch.Tensor]] = [None] * len(video_frames)
    frames_with_objects = [idx for idx, bboxes in enumerate(frame_object_bboxes) if len(bboxes) > 0]

    # Iterate over the frame indices that have at least one object to ground
    for frame_idx in frames_with_objects:

        # Extract all the objects that need to be grounded for this frame
        input_frame = video_frames[frame_idx]
        _, frame_h, frame_w = input_frame.shape
        all_keys = sorted(frame_object_bboxes[frame_idx].keys())
        num_objects = len(all_keys)

        # For the main actor, we skip SAM2 grounding and assign a zero mask with score 1.0
        if frame_object_names is not None:
            names_dict = frame_object_names[frame_idx]
            regular_keys = [k for k in all_keys if names_dict.get(k, "") != "main actor"]
            main_actor_keys = [k for k in all_keys if names_dict.get(k, "") == "main actor"]
        else:
            regular_keys = all_keys
            main_actor_keys = []

        # Allocate combined output tensors
        combined_masks = torch.zeros(num_objects, frame_h, frame_w, dtype=torch.bool, device=device)
        combined_scores = torch.zeros(num_objects, device=device)

        # main actor: always grounded — zero mask, score 1.0
        for k in main_actor_keys:
            combined_scores[k] = 1.0

        # Run SAM2 only on non-main actor objects
        if regular_keys:
            input_boxes = [[frame_object_bboxes[frame_idx][k] for k in regular_keys]]

            # Ground all objects in the current frame using SAM2
            frame_inputs = grounding_processor(images=input_frame, input_boxes=input_boxes, return_tensors="pt").to(device)
            frame_output = grounding_model(**frame_inputs, multimask_output=False)
            regular_masks = grounding_processor.post_process_masks(frame_output.pred_masks, frame_inputs["original_sizes"], binarize=True)[0].squeeze(1)
            regular_scores = frame_output.iou_scores.squeeze(0)

            # Ensure 2D tensors when there is only one object
            if regular_masks.dim() == 2:
                regular_masks = regular_masks.unsqueeze(0)
                regular_scores = regular_scores.unsqueeze(0)

            for i, k in enumerate(regular_keys):
                combined_masks[k] = regular_masks[i]
                combined_scores[k] = regular_scores[i]

        frame_grounding_masks[frame_idx] = combined_masks
        frame_grounding_scores[frame_idx] = combined_scores

    # Return the grounding masks and scores for all frames, converting scores to CPU lists for easier serialization
    return frame_grounding_masks, [scores.cpu().tolist() if scores is not None else [] for scores in frame_grounding_scores]


def consolidate(
    video_frames: torch.Tensor,
    graphs: list[dict],
    tracking_model: Sam2VideoModel,
    tracking_processor: Sam2VideoProcessor,
    grounding_model: Sam2Model,
    grounding_processor: Sam2Processor,
    dino_model: DinoV2FE,
    semantic_similarity_threshold: float = 0.8,
    enforce_label_consistency: bool = True,
):
    """Merge per-frame scene graphs into a window-level consolidated graph.

    Parameters
    ----------
    video_frames : torch.Tensor
        Tensor with shape ``(T, C, H, W)`` containing the frames in one window.
    graphs : list[dict]
        Per-frame graph dictionaries with keys such as ``obj``, ``bbox``,
        ``pair``, and ``rel``.
    tracking_model : Sam2VideoModel
        SAM2 video model used for temporal propagation.
    tracking_processor : Sam2VideoProcessor
        Processor responsible for tracker session setup and mask decoding.
    grounding_model : Sam2Model
        SAM2 image model used to ground bounding boxes into masks.
    grounding_processor : Sam2Processor
        Processor used for grounding pre-processing and post-processing.
    dino_model : DinoV2FE
        DINOv2 feature extractor used as a fallback matcher.
    semantic_similarity_threshold : float, optional
        Minimum cosine similarity required for DINO-based fallback matching.
    enforce_label_consistency : bool, optional
        If ``True``, only matches objects whose labels agree between grounding
        and tracking histories.

    Returns
    -------
    dict
        Consolidated graph with canonical object labels, best boxes, relation
        endpoints, relation labels, and per-track observation history.

    Notes
    -----
    The matching pipeline first attempts IoU-based Hungarian assignment and
    then falls back to DINO similarity for unmatched grounded objects.
    """
    assert len(video_frames) == len(graphs), "Number of frames and graphs must be the same"

    _, _, h, w = video_frames.shape

    # Step 1: collect bounding boxes and object names for each frame
    frame_object_bboxes, frame_object_names = [], []  # frame_idx -> (bounding box list, object name list)
    for graph in graphs:
        object_ids = range(len(graph["obj"]))
        frame_object_bboxes.append(dict(zip(object_ids, [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in graph["bbox"]])))
        frame_object_names.append(dict(zip(object_ids, graph["obj"])))

    # Step 2: take the first frame that has at least one detection (other than the main actor)
    assert any(len(bboxes) > 0 for bboxes in frame_object_bboxes), "No detections in any frame of this consolidation window"
    current_start_frame: int = next(
        (idx for idx, bboxes in enumerate(frame_object_bboxes) if len(bboxes) > 0 and not all(frame_object_names[idx][k] == "main actor" for k in bboxes.keys())),
        -1,
    )  # skip frames with only main actor detections if possible
    assert current_start_frame != -1, breakpoint()  # No valid detections (other than main actor) in any frame of this consolidation window

    # Step 3: Extract frame-level grounding masks using SAM2 for all objects in the window
    frame_grounding_masks, frame_grounding_scores = frame_level_grounding(video_frames, frame_object_bboxes, grounding_model, grounding_processor, frame_object_names)
    # and precompute DINO features for all objects across all frames in the window
    # This features will be used for fallback matching when IoU fails to match a grounded object to a track
    dino_boxes = [torch.tensor(list(bboxes.values())) if len(bboxes.values()) else torch.empty((0, 4)) for bboxes in frame_object_bboxes]
    dino_boxes_batch = [idx * torch.ones(len(boxes), dtype=torch.long) for idx, boxes in enumerate(dino_boxes)]
    frame_dino_features = dino_model(video_frames, torch.vstack(dino_boxes).to(device), torch.cat(dino_boxes_batch).to(device))
    frame_dino_features = F.normalize(frame_dino_features, dim=-1)
    frame_dino_features = unbatch(frame_dino_features, torch.cat(dino_boxes_batch))  # List of (N_i, dim) tensors, one per frame, where N_i is the number of objects in frame i

    # Per-track best DINO feature: track_id -> (feature_vector, confidence_score)
    # Initialized after seeding the first frame, updated during matching
    track_features: dict[int, torch.Tensor] = {}

    # Step 4: Start the video-level inference session
    inference_session = tracking_processor.init_video_session(video=video_frames, inference_device=device, dtype=torch.bfloat16)
    video_width, video_height = inference_session.video_width, inference_session.video_height

    # Get precomputed grounded masks for the first frame in the window
    grounding_masks = frame_grounding_masks[current_start_frame]
    assert grounding_masks is not None and len(grounding_masks) > 0, "no objects to ground in the first frame"

    # Add the objects grounded (except the main actor) in the first frame to the tracking model's inference session
    tracking_processor.add_inputs_to_inference_session(
        inference_session,
        frame_idx=current_start_frame,
        input_masks=[mask for idx, mask in enumerate(grounding_masks) if frame_object_names[current_start_frame][idx] != "main actor"],  # main actor doesn't get added to the tracker
        obj_ids=[idx for idx in range(len(grounding_masks)) if frame_object_names[current_start_frame][idx] != "main actor"],  # main actor doesn't get added to the tracker
    )

    # Keep track of the objects across frames in the consolidation window
    # (object_id -> [track history list of (frame_idx, obj_idx_in_frame, confidence_score)])
    history: dict[int, list[HistoryEntry]] = {
        idx: [HistoryEntry(current_start_frame, idx, name, frame_grounding_scores[current_start_frame][idx])] for idx, name in frame_object_names[current_start_frame].items()
    }  # (object_id -> track_id)

    # Keep track of the best DINO feature vector observed for each track to enable DINO-based matching when IoU fails
    for idx in history.keys():
        track_features[idx] = frame_dino_features[current_start_frame][idx]

    # Keep track of the already seen frames
    seen_frames = set()

    while current_start_frame < len(video_frames):

        for tracker_output in tracking_model.propagate_in_video_iterator(inference_session, start_frame_idx=current_start_frame):

            current_frame: int = tracker_output.frame_idx  # type: ignore
            tracking_object_ids: np.ndarray = np.array(tracker_output.object_ids)  # type: ignore

            # Reuse precomputed grounded masks for the current frame
            grounding_masks, grounding_scores = frame_grounding_masks[current_frame], frame_grounding_scores[current_frame]

            # Track the object masks we want to add to the inference session for the current frame, along with their assigned track ids
            masks_to_add, objects_ids_to_add = [], []

            if grounding_masks is None:
                current_start_frame = current_frame + 1
                continue

            if current_frame not in seen_frames:  # Perform this operation exactly once per frame

                tracking_masks = tracking_processor.post_process_masks([tracker_output.pred_masks], original_sizes=[[video_height, video_width]], binarize=True)[0].squeeze(1).to(device)

                logger.debug(f"Frame {current_frame} has {len(grounding_masks)} grounding masks and {len(tracking_masks)} tracking masks")

                # Get the most common object name associated with each tracking mask based on the history of matched grounding masks for that track
                tracking_object_names = {
                    obj_id: Counter([entry.name for entry in history[obj_id] if entry.name is not None]).most_common(1)[0][0] for obj_id in tracker_output.object_ids  # type: ignore
                }

                assert (
                    "main actor" not in tracking_object_names.values()
                ), "main actor should not be associated with any tracking mask since it doesn't get added to the tracker and should be handled separately in the grounding masks"

                # Consolidation substep 1: IoU-based Hungarian matching
                ious = compute_iou_batch(grounding_masks, tracking_masks).cpu().numpy()  # (G, T)
                cost_matrix = 1.0 - ious

                # Consolidation substep 2: Keep track of which grounding masks have been matched to a tracking mask
                grounding_outputs_matched = np.zeros(len(grounding_masks), dtype=bool)
                tracking_outputs_matched = np.zeros(len(tracking_masks), dtype=bool)
                tracklets_matched = np.zeros(len(history), dtype=bool)

                # NOTE: here the main actor has a zero mask
                for g_idx in range(len(grounding_masks)):
                    if frame_object_names[current_frame][g_idx] == "main actor":
                        # check if a main actor is already in the history

                        found = False
                        for entry in history.values():
                            if any(e.name == "main actor" for e in entry):
                                entry.append(HistoryEntry(current_frame, int(g_idx), "main actor", 1.0))
                                masks_to_add.append(None)
                                objects_ids_to_add.append(-1)  # main actor gets a special object id of -1 since it doesn't correspond to a tracklet
                                found = True
                                break

                        if not found:
                            new_id = len(history)
                            history[new_id] = [(HistoryEntry(idx, None, None, None)) for idx in range(current_frame)] + [HistoryEntry(current_frame, int(g_idx), "main actor", 1.0)]
                            masks_to_add.append(None)
                            objects_ids_to_add.append(-1)  # main actor gets a special object id of -1 since it doesn't correspond to a tracklet
                            track_features[new_id] = torch.zeros_like(frame_dino_features[current_frame][g_idx])  # main actor gets a zero feature vector since it doesn't correspond to a tracklet
                            tracklets_matched = np.append(tracklets_matched, True)  # main actor is considered already matched since it doesn't correspond to a tracklet

                        grounding_outputs_matched[g_idx] = True

                # Prevent matching with the main actor track
                for t_idx, entry in history.items():
                    if any(e.name == "main actor" for e in entry):
                        tracklets_matched[t_idx] = True

                cost_matrix[grounding_outputs_matched, :] = 1e6  # set a high cost for already matched grounding masks to prevent them from being matched again
                cost_matrix[:, tracking_outputs_matched] = 1e6  # set a high cost for already matched tracking masks to prevent them from being matched again
                cost_matrix = np.where(ious < 0.5, 1e6, cost_matrix)  # set a high cost for pairs with low IoU to prevent them from being matched

                grounding_idxs, tracking_idxs = linear_sum_assignment(cost_matrix)
                for g_idx, t_idx, track_obj_id, cost in zip(grounding_idxs, tracking_idxs, tracking_object_ids[tracking_idxs], cost_matrix[grounding_idxs, tracking_idxs]):
                    if cost >= 1e6:
                        continue  # skip pairs that were not matched due to low IoU or because the grounding mask was already matched

                    g_label = frame_object_names[current_frame][g_idx]
                    t_label = tracking_object_names[track_obj_id]

                    if enforce_label_consistency and t_label != g_label:
                        continue

                    # print(f"  IoU match: grounding {g_idx} '{g_label}' -> track {track_obj_id} '{t_label}' (IoU={ious[g_idx, track_obj_id].item():.3f})")
                    grounding_outputs_matched[g_idx] = True
                    tracking_outputs_matched[t_idx] = True
                    tracklets_matched[track_obj_id] = True

                    masks_to_add.append(grounding_masks[g_idx][None].float())
                    objects_ids_to_add.append(track_obj_id)
                    history[track_obj_id].append(HistoryEntry(current_frame, int(g_idx), g_label, grounding_scores[g_idx]))

                    # Update track feature if this observation has higher confidence
                    if grounding_scores[g_idx] > max(entry.confidence_score for entry in history[track_obj_id] if entry.confidence_score is not None):
                        track_features[track_obj_id] = frame_dino_features[current_frame][g_idx]

                # Step 2: DINO similarity fallback for IoU-unmatched objects
                if (grounding_outputs_matched == 0).sum() and len(track_features) > 0:

                    # Cosine similarity (features are already L2-normalized)
                    assert all(track_id in track_features for track_id in tracking_object_ids), "All tracking object ids should have an associated DINO feature in track_features"
                    assert sorted(track_features.keys()) == sorted(range(max(track_features.keys()) + 1)), "Track features keys should match tracking object ids"
                    sim_matrix = (frame_dino_features[current_frame] @ torch.stack([track_features[k] for k in sorted(track_features.keys())]).T).cpu().numpy()  # (U, K)

                    dino_cost = 1.0 - sim_matrix
                    dino_cost[sim_matrix < semantic_similarity_threshold] = 1e6  # set a high cost for pairs with low similarity to prevent them from being matched
                    dino_cost[grounding_outputs_matched, :] = 1e6  # set a high cost for already matched grounding masks to prevent them from being matched again
                    dino_cost[:, tracklets_matched] = 1e6  # set a high cost for already matched tracklets to prevent them from being matched again

                    dg_idxs, dt_idxs = linear_sum_assignment(dino_cost)

                    for g_idx, obj_track_id, cost in zip(dg_idxs, dt_idxs, dino_cost[dg_idxs, dt_idxs]):
                        if cost >= 1e6:
                            continue  # skip pairs that were not matched due to low similarity or because the grounding mask or tracklet was already matched

                        g_label = frame_object_names[current_frame][g_idx]
                        t_label = Counter([entry.name for entry in history[obj_track_id] if entry.name is not None]).most_common(1)[0][0]

                        if enforce_label_consistency and t_label != g_label:
                            continue

                        if t_label == "main actor":
                            continue  # main actor should have already been matched and assigned a track (or spawned a new one) in the previous step, so we skip it in the DINO matching

                        # Find the best IoU this grounding mask had with any tracking mask
                        # best_iou: float = ious[g_idx].max().item()
                        # best_iou_track: int = tracker_output.object_ids[ious[g_idx].argmax().item()]  # type: ignore
                        # print(f"  DINO match: grounding {g_idx} '{g_label}' -> track {matched_track_id} '{t_label}' (sim={similarity:.3f}, best IoU was {best_iou:.3f} with track {best_iou_track})")

                        masks_to_add.append(grounding_masks[g_idx][None].float())
                        objects_ids_to_add.append(obj_track_id)
                        history[obj_track_id].append(HistoryEntry(current_frame, int(g_idx), g_label, grounding_scores[g_idx]))
                        if grounding_scores[g_idx] > max(entry.confidence_score for entry in history[obj_track_id] if entry.confidence_score is not None):
                            track_features[obj_track_id] = frame_dino_features[current_frame][g_idx]

                        grounding_outputs_matched[g_idx] = True
                        tracklets_matched[obj_track_id] = True

                # Remaining unmatched objects become new tracks
                if np.any(~grounding_outputs_matched):
                    for g_idx in np.argwhere(~grounding_outputs_matched).flatten():
                        g_label = frame_object_names[current_frame][g_idx]
                        # print(f"  New track for grounding {g_idx} '{g_label}' at frame {current_frame}")
                        new_id = len(history)
                        masks_to_add.append(grounding_masks[g_idx][None].float())
                        objects_ids_to_add.append(new_id)
                        history[new_id] = [(HistoryEntry(idx, None, None, None)) for idx in range(current_frame)] + [HistoryEntry(current_frame, int(g_idx), g_label, grounding_scores[g_idx])]
                        track_features[new_id] = frame_dino_features[current_frame][g_idx]

            assert current_frame in seen_frames or (len(masks_to_add) == len(objects_ids_to_add) == len(grounding_masks)), "All grounded objects must have been matched to a track at this point"

            for g_idx in range(len(grounding_masks)):
                if not any(entry.frame_idx == current_frame and entry.obj_idx_in_frame == g_idx for t_idx in history.keys() for entry in history[t_idx]):
                    g_label = frame_object_names[current_frame][g_idx]
                    logger.warning(f"Grounded object {g_idx} '{g_label}' in frame {current_frame} was not matched to any track!")

            if len(objects_ids_to_add) > 0 and any(obj_id != -1 for obj_id in objects_ids_to_add):
                tracking_processor.add_inputs_to_inference_session(
                    inference_session,
                    frame_idx=current_frame,
                    obj_ids=[obj_id for obj_id in objects_ids_to_add if obj_id != -1],  # main actor doesn't get added to the tracker
                    input_masks=[mask for mask, obj_id in zip(masks_to_add, objects_ids_to_add) if obj_id != -1],  # main actor doesn't get added to the tracker
                )

            # We need to perform overlap checks exactly once per frame
            seen_frames.add(current_frame)

            # 5. The Critical Step: If we added new objects, we MUST restart the generator
            if len(objects_ids_to_add) > 0:
                # Set the start frame to the current frame so the tracker re-evaluates
                # this frame and properly encodes the new objects into memory.
                current_start_frame = current_frame

                break  # Break out of the inner for-loop to trigger the while-loop restart

            # If no new objects were added, update the start frame tracker safely
            current_start_frame = current_frame + 1

    return build_graph_from_history(graphs, history, frame_object_bboxes, w, h)


def build_graph_from_history(graphs: list[dict], matching_history: dict[int, list[HistoryEntry]], frame_object_bboxes, w: float, h: float) -> dict:
    """Build the consolidated graph for one window from track history.

    Parameters
    ----------
    graphs : list[dict]
        Per-frame scene graphs used to recover relation triplets.
    matching_history : dict[int, list[HistoryEntry]]
        Mapping from consolidated track identifier to the corresponding list of
        observations over time.
    frame_object_bboxes : list[dict]
        Per-frame mapping from object index to absolute bounding box.
    w : float
        Frame width used to normalize bounding box coordinates.
    h : float
        Frame height used to normalize bounding box coordinates.

    Returns
    -------
    dict
        Consolidated graph containing object labels, confidence scores, best
        normalized boxes, reference frame indices, relations, and raw history.
    """

    triplets = set()

    for graph_idx, graph in enumerate(graphs):

        for (src, tgt), rel in zip(graph["pair"], graph["rel"]):
            # Find a matching tracklet for subject
            x_track = next(
                (obj_idx for obj_idx, object_history in matching_history.items() if any(entry.frame_idx == graph_idx and entry.obj_idx_in_frame == src for entry in object_history)),
                None,
            )

            # Find a matching tracklet for object
            y_track = next(
                (obj_idx for obj_idx, object_history in matching_history.items() if any(entry.frame_idx == graph_idx and entry.obj_idx_in_frame == tgt for entry in object_history)),
                None,
            )

            if x_track is None or y_track is None:
                print("Bad tracking history entry for graph_idx", graph_idx, "x: ", graph["obj"][src], "y", graph["obj"][tgt])
                print("x_track:", x_track, "y_track:", y_track)
                continue  # one of the two objects was not tracked correctly

            if (x_track, rel, y_track) in triplets:
                continue  # we have already processed this triplet for a previous frame, no need to do it again

            if x_track == y_track:
                continue  # skip self-relations for now

            triplets.add((x_track, rel, y_track))

    # Final graph: build obj names efficiently
    obj = [Counter([entry.name for entry in entries if entry.obj_idx_in_frame is not None]).most_common(1)[0][0] for entries in matching_history.values()]
    confidence = [max([entry.confidence_score for entry in entries if entry.confidence_score is not None]) for entries in matching_history.values()]
    best_frame = [next(((entry.frame_idx, entry.obj_idx_in_frame) for entry in entries if entry.confidence_score == conf), (0, 0)) for entries, conf in zip(matching_history.values(), confidence)]

    history = [[(entry.frame_idx, entry.obj_idx_in_frame, entry.name) for entry in history if entry.obj_idx_in_frame is not None] for history in matching_history.values()]

    roles = []
    for entries in matching_history.values():
        observed_roles = []
        for entry in entries:
            if entry.obj_idx_in_frame is None or entry.frame_idx >= len(graphs):
                continue
            frame_roles = graphs[entry.frame_idx].get("role", [])
            if entry.obj_idx_in_frame < len(frame_roles) and frame_roles[entry.obj_idx_in_frame]:
                observed_roles.append(frame_roles[entry.obj_idx_in_frame])
        roles.append(Counter(observed_roles).most_common(1)[0][0] if observed_roles else None)

    # Normalize bounding boxes to [0, 1]
    best_boxes = [frame_object_bboxes[frame_idx][obj_idx] for frame_idx, obj_idx in best_frame]
    best_boxes = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in best_boxes]

    best_frame = [frame_idx for frame_idx, _ in best_frame]
    pairs = [(x_track, y_track) for x_track, _, y_track in triplets]
    rels = [rel for _, rel, _ in triplets]

    return {
        "obj": obj,
        "role": roles,
        "confidence": confidence,
        "bbox": best_boxes,
        "frame_idx": best_frame,
        "pair": pairs,
        "rel": rels,
        "history": history,
    }
