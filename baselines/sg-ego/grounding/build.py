"""
Build frame-level scene graphs from (subject, relation, object) triplets.
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import itertools
import json
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from grounding.detector import GroundingDINO
from grounding.utils.bbox import compute_area, compute_area_intersection, compute_iou
from grounding.utils.filters import apply_spatial_heuristics, intra_caption_filtering

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

CAMERA_WEARER_LABEL = "##camera_wearer##"
ROLE_NAMES = {"robot", "manipulated_object", "initial_support", "target"}


def _role_from_raw_entity(value: str) -> str | None:
    if "::" not in value:
        return None
    role = value.split("::", 1)[0].strip().lower()
    return role if role in ROLE_NAMES else None


def build_frame_scene_graph(
    captions: List[Tuple[str, str, str]],
    triplets: List[Tuple[str, str, str]],
    detection_results: List[Dict[str, Any]],
    tags: List[Tuple[int, int]],
    relations_mapper: Callable[[list[str]], tuple[torch.Tensor, list[str]]],
    overlap_iou_threshold: float = 0.3,
    use_spatial_heuristics: bool = False,
    double_detection_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Build a frame-level scene graph from grounded (subject, relation, object) triplets.

    The function aligns parsed (subject, relation, object) triplets with detector predictions,
    resolves duplicate/overlapping detections, and produces a compact scene graph
    representation with object nodes and relation edges.

    Parameters
    ----------
    captions : List[Tuple[str, str, str]]
        Raw caption triplets (subject, relation, object) including optional
        instance suffixes (for example ``("picture_frame_1", "on", "wall_1")``).
    triplets : List[Tuple[str, str, str]]
        Parsed/normalized triplets derived from ``captions``
        (for example ``("picture frame", "on", "wall")``).
    detection_results : List[Dict[str, Any]]
        Per-caption detector outputs. Each dict is expected to contain at least
        ``text_labels``, ``scores``, ``boxes``, and ``type`` (subject/object).
    tags : List[Tuple[int, int]]
        Per-triplet ``(subject_tag, object_tag)`` used to preserve instance
        identity when multiple objects share the same label.
    relations_mapper : Callable[[list[str]], tuple[torch.Tensor, list[str]]]
        Optional mapper that aligns relation strings to a closed-set vocabulary.
        Used only when ``use_spatial_heuristics`` is enabled.
    overlap_iou_threshold : float, optional
        IoU threshold used when matching detections that share label and tag.
    use_spatial_heuristics : bool, optional
        Whether to prune candidate relations using geometry-aware heuristics.
    double_detection_threshold : float, optional
        IoU threshold for suppressing likely duplicate detections across
        different triplets, regardless of label.

    Returns
    -------
    Dict[str, Any]
        Frame graph with keys:

        - ``obj``: list of object labels.
        - ``confidence``: confidence score for each object node.
        - ``bbox``: normalized boxes for each object node.
        - ``pair``: ``(subject_idx, object_idx)`` edges.
        - ``rel``: relation label for each edge in ``pair``.
        - ``grounded_from``: original raw caption triplet that generated each
          edge.

    Notes
    -----
    The grounding pipeline applies three filtering stages:

    1. Intra-caption duplicate suppression.
    2. Cross-caption high-overlap suppression.
    3. Node-level merge against already accepted objects.

    The special label ``##camera_wearer##`` is handled as a special single instance node.
    """

    assert len(captions) == len(triplets) == len(detection_results) == len(tags), "Length of captions, triplets, detection_results and tags must be the same"

    objects, confidences, bboxes, object_roles = [], [], [], []
    # Stores (raw_caption, (subject_idx, relation, object_idx)) before
    # de-duplication into the final graph structure.
    grounded_triplets = list()

    # Build a dictionary mapping all the relations in the captions to their corresponding entry in the closed set
    rels = [rel for _, rel, _ in triplets]
    mapped_rels = dict(zip(rels, relations_mapper(rels)[1] if relations_mapper is not None else rels)) if len(rels) > 0 else {}

    # Filter 1: Within each triplet, filter out overlapping detections for the subject and/or the object. The goal of this part is to
    # filter multiple detections for the same triplet that have significant overlap.
    # Example: "cup on table" -> GDino detects cup twice, but one is inside the other or one is much more confident than the other.
    detection_results = intra_caption_filtering(captions, detection_results)

    for idx_triplet, (raw_caption, (subj, _, obj), rel, (subj_tag, obj_tag), dets) in enumerate(zip(captions, triplets, rels, tags, detection_results)):

        # Collect all the nodes that refer to either the subject or the object in the triplet
        subj_idxs, obj_idxs = set(), set()

        # Keep track of nodes first introduced while processing this caption.
        # This is currently informational and not consumed downstream.
        added_indices = []

        # Filter 2: Skip overlapping detections across different triplets (introduced in v8)
        for label, score, bbox, det_type in zip(dets["text_labels"], dets["scores"], dets["boxes"], dets["type"]):
            # Iterate over the valid detections for this triplet

            tag = subj_tag if det_type == "subject" else obj_tag
            raw_entity = raw_caption[0] if det_type == "subject" else raw_caption[2]
            role = _role_from_raw_entity(raw_entity)

            if label != CAMERA_WEARER_LABEL:

                # Check whether another triplet already explains this region
                # with higher confidence and high IoU.
                skip = False
                for idx_other_triplet, (*_, (other_subj_tag, other_obj_tag), other_triplet_det) in enumerate(zip(triplets, rels, tags, detection_results)):
                    # Iterate over all the other triplets

                    if idx_other_triplet == idx_triplet:
                        continue

                    for other_label, other_score, other_bbox, other_det_type in zip(
                        other_triplet_det["text_labels"], other_triplet_det["scores"], other_triplet_det["boxes"], other_triplet_det["type"]
                    ):
                        # Iterate over all the detections for the other triplet
                        other_tag = other_subj_tag if other_det_type == "subject" else other_obj_tag

                        if compute_iou(bbox, other_bbox) > double_detection_threshold and score < other_score:
                            # they have different labels or (iv) the label is the same but they have different tags
                            if other_label != label or other_tag != tag:
                                skip = True

                # If any of the previous conditions is true, we skip this detection as it is likely to be a duplicate detection of the same object with a lower confidence score
                if skip:
                    continue

            # At this point the detection is valid, i.e., it does not overlap with any other detection

            # Filter 3: Find potential existing matches among the already added objects (avoid node duplicates)
            # We match a new object with an already existing node if they have the same label and either:
            #  - the new object is contained in the candidate object or vice versa
            #    (i.e., one of the two bounding boxes has a large intersection over union with the other)
            #    (in this case the tag is ignored)
            #  - same tag and low iou
            #  - possibly different tags but high iou threshold (0.5)
            candidate = None
            for i, ((candidate_label, candidate_tag), candidate_bbox) in enumerate(zip(objects, bboxes)):
                if label != candidate_label:
                    continue

                if label == CAMERA_WEARER_LABEL:
                    # Camera wearer has no associated tags
                    candidate = i
                    break

                # Compute the area of the new objects' bbox and the (potential) candidate
                area_bbox, area_other, area_intersection = compute_area(bbox), compute_area(candidate_bbox), compute_area_intersection(bbox, candidate_bbox)

                # Check if one of the objects is contained inside the other
                is_contained = area_intersection / area_bbox >= 0.5 or area_intersection / area_other >= 0.5
                # Check if the object and the (potential) candidate have the same tracking tag
                is_same_tag = candidate_tag == tag

                # Find a node with the same label and sufficient IoU overlap and better confidence
                if is_contained or (is_same_tag and compute_iou(bbox, bboxes[i]) > overlap_iou_threshold) or (compute_iou(bbox, bboxes[i]) > 0.5):
                    candidate = i if candidate is None or confidences[i] > confidences[candidate] else candidate  # if multiple matches, we take the one with the highest confidence

            # Conservative suppression: if the detection strongly overlaps with
            # any existing node and no match was found above, ignore it.
            if candidate is None and any(compute_iou(bbox, bboxes[i]) > double_detection_threshold for i in range(len(objects))):
                continue

            # No candidate found -> add a new object
            if candidate is None:
                objects.append((label, tag))
                confidences.append(score)
                bboxes.append(bbox)
                object_roles.append(role)
                candidate = len(objects) - 1

                added_indices.append(candidate)
            else:
                # We found a candidate, update the bbox if the new bbox is more confident
                if score > confidences[candidate]:
                    confidences[candidate] = score
                    bboxes[candidate] = bbox
                if object_roles[candidate] is None:
                    object_roles[candidate] = role

            if label == subj or (label == CAMERA_WEARER_LABEL and raw_caption[0].lower() == "camera_wearer"):
                subj_idxs.add(candidate)
            elif label == obj or (label == CAMERA_WEARER_LABEL and raw_caption[2].lower() == "camera_wearer"):
                obj_idxs.add(candidate)

        if raw_caption[0].lower() == "camera_wearer" and len(subj_idxs) == 0:
            if CAMERA_WEARER_LABEL not in [obj for obj, _ in objects]:
                objects.append((CAMERA_WEARER_LABEL, -1))
                confidences.append(1.0)
                bboxes.append([0.0, 0.0, 1.0, 1.0])
                object_roles.append("robot")
                candidate = len(objects) - 1

                added_indices.append(candidate)
            subj_idxs.add([obj for obj, _ in objects].index(CAMERA_WEARER_LABEL))
            pass

        if raw_caption[2].lower() == "camera_wearer" and len(obj_idxs) == 0:
            if CAMERA_WEARER_LABEL not in [obj for obj, _ in objects]:
                objects.append((CAMERA_WEARER_LABEL, -1))
                confidences.append(1.0)
                bboxes.append([0.0, 0.0, 1.0, 1.0])
                object_roles.append("robot")
                candidate = len(objects) - 1

                added_indices.append(candidate)
            obj_idxs.add([obj for obj, _ in objects].index(CAMERA_WEARER_LABEL))
            pass

        # If we don't have at least one detection for the subject and one for the object, we skip this triplet (i.e., we don't add any relation)
        candidate_triplets = set(itertools.product(subj_idxs, [rel], obj_idxs))

        # Optionally, apply spatial filtering heuristics ("on", "in", "left of", "right of") to prune candidate triplets that are not spatially consistent
        # (e.g., "cup on table" but the detected cup is not on the detected table)
        if use_spatial_heuristics:
            valid_candidate_triplets = apply_spatial_heuristics(candidate_triplets, bboxes, mapped_rels)
        else:
            valid_candidate_triplets = candidate_triplets

        assert all(idx_subj < len(objects) and idx_obj < len(objects) for idx_subj, _, idx_obj in valid_candidate_triplets), "Candidate triplet indices should be valid indices of the objects list"

        for idx_subj, rel, idx_obj in valid_candidate_triplets:
            grounded_triplets.append((raw_caption, (idx_subj, rel, idx_obj)))

    # Remove any duplicate relations (i.e., same subject and object and same relation)
    objects = [obj for obj, _ in objects]  # Strip the tracking number

    # Filter unique (subj, rel, obj) triplets
    seen_triplets = set()
    pairs, rels = [], []
    grounded_from = []

    for raw_caption, (idx_subj, rel, idx_obj) in grounded_triplets:
        if (idx_subj, rel, idx_obj) not in seen_triplets:
            seen_triplets.add((idx_subj, rel, idx_obj))
            pairs.append((idx_subj, idx_obj))
            rels.append(rel)
            grounded_from.append(raw_caption)

    # Ensure that all indices in pairs are valid indices of output_objects
    assert all(0 <= idx < len(objects) for pair in pairs for idx in pair), "All indices in pairs must be valid indices of output_objects"

    # Remove leaf nodes: remove any object that is not part of any pair, and update the indices in pairs accordingly
    # (e.g., if we remove the object at index 0, we need to decrease by 1 all the indices in pairs that are greater than 0)
    output_objects, output_confidences, output_bboxes, output_roles = [], [], [], []
    for idx_object in range(len(objects))[::-1]:
        obj, conf, bbox = objects[idx_object], confidences[idx_object], bboxes[idx_object]

        if not any(idx_object == idx for pair in pairs for idx in pair):
            # this object was not actually used in any relation, we can skip it

            for idx_pair in range(len(pairs)):
                if pairs[idx_pair][0] > idx_object:
                    pairs[idx_pair] = (pairs[idx_pair][0] - 1, pairs[idx_pair][1])
                if pairs[idx_pair][1] > idx_object:
                    pairs[idx_pair] = (pairs[idx_pair][0], pairs[idx_pair][1] - 1)

            continue

        output_objects.insert(0, obj)
        output_confidences.insert(0, conf)
        output_bboxes.insert(0, bbox)
        output_roles.insert(0, object_roles[idx_object])

    assert len(output_objects) == len(output_confidences) == len(output_bboxes), "Length of objects, confidences and bboxes must be the same"
    assert len(pairs) == len(rels), "Length of pairs and relations must be the same"
    # Ensure that all indices in pairs are valid indices of output_objects
    assert all(0 <= idx < len(output_objects) for pair in pairs for idx in pair), "All indices in pairs must be valid indices of output_objects"
    # Ensure that each object is part of at least one pair
    assert all((idx in [subj for subj, _ in pairs]) or (idx in [obj for _, obj in pairs]) for idx in range(len(output_objects))), "All output_objects must be part of at least one pair"
    assert len([obj for obj in output_objects if obj == "##camera_wearer##"]) <= 1, "There should be at most one camera wearer in the graph"
    assert all(role in ROLE_NAMES for role in output_roles), "Goal-guided output contains an unassigned or irrelevant role"

    return {"obj": output_objects, "role": output_roles, "confidence": output_confidences, "bbox": output_bboxes, "pair": pairs, "rel": rels, "grounded_from": grounded_from}


def build_video_scene_graphs(
    video_dataloader: DataLoader,
    detector: GroundingDINO,
    output_path: str,
    objects_mapper: Callable[[list[str]], tuple[torch.Tensor, list[str]]],
    relations_mapper: Callable[[list[str]], tuple[torch.Tensor, list[str]]],
    det_batch_size: int = 8,
    overlap_iou_threshold: float = 0.2,
    use_spatial_heuristics: bool = False,
    double_detection_threshold: float = 0.8,
):
    """Build and persist frame-level graphs for all frames in a video.

    For each frame, the function grounds all caption triplets through the
    detector, builds an unmapped graph (open vocabulary), and builds a mapped
    graph where object/relation labels are projected to the configured
    closed-set vocabularies.

    Parameters
    ----------
    video_dataloader : DataLoader
        Batches of frame indices, images, raw captions, parsed triplets, and
        instance tags.
    detector : DINOdetector
        Detector used to ground subject/object mentions for each triplet.
    output_path : str
        Output prefix. Two files are produced:

        - ``{output_path}.json`` for open-vocabulary graphs.
        - ``{output_path}_mapped.json`` for closed-set mapped graphs.
    objects_mapper : Callable[[list[str]], tuple[torch.Tensor, list[str]]]
        Mapper that projects object labels to a closed vocabulary.
    relations_mapper : Callable[[list[str]], tuple[torch.Tensor, list[str]]]
        Mapper that projects relation labels to a closed vocabulary.
    det_batch_size : int, optional
        Batch size used when calling the detector.
    overlap_iou_threshold : float, optional
        Node-merge IoU threshold passed to :func:`build_graph`.
    use_spatial_heuristics : bool, optional
        Whether spatial consistency checks are enabled in :func:`build_graph`.
    double_detection_threshold : float, optional
        Cross-triplet duplicate-suppression threshold passed to
        :func:`build_graph`.

    Returns
    -------
    None
        Results are written to disk; the function does not return in-memory
        graph structures.
    """

    video_mapped_graphs = dict()
    video_unmapped_graphs = dict()

    num_empty_graphs, total_num_graphs = 0, 0

    # Iterate over batches of frames
    # Each frame may contain an arbitrary number of captions. Hence we proceed in three steps:
    #  1) we extract the triplets for all captions in the batch
    #  2) we run the detector on all pairs (frame, caption) in the batch, and then we group the results by frame
    #  3) for each frame, we build the graph by matching the triplets with the detected objects
    pbar = tqdm(video_dataloader, total=len(video_dataloader), desc="Processing frames", leave=False)
    for batch in pbar:

        assert len(batch.index) == len(batch.triplets), "Batch size mismatch between indices and captions"

        # Step 1: compute how many triplets belong to each frame in the batch.
        counts = [len(triplets) for triplets in batch.triplets]

        # Step 2: Extract detection results (batched)
        flat_images = [image for image, triplets in zip(batch.image, batch.triplets) for _ in triplets]
        flat_triplets = [triplet for triplets in batch.triplets for triplet in triplets]

        assert len(flat_triplets) == len(flat_images) == sum(counts), "Total number of triplets must match total number of captions in the batch"

        detection_results = []
        for i in range(0, len(flat_images), det_batch_size):
            images_slice, triplets_slice = flat_images[i : i + det_batch_size], flat_triplets[i : i + det_batch_size]
            detection_results.extend(detector.run(images_slice, triplets_slice))

        # detection_results = detector.run(flat_images, flat_triplets)
        detection_results = [detection_results[sum(counts[:i]) : sum(counts[: i + 1])] for i in range(len(counts))]

        assert len(detection_results) == len(batch.triplets), "Total number of detection results must match total number of triplets in the batch"
        assert all(len(det) == len(triplets) for det, triplets in zip(detection_results, batch.triplets)), "Number of detection results per triplet must match number of triplets in the batch"

        # Step 3: Build graphs from triplets and detection results
        for idx, captions, triplets, tags, det_results in zip(batch.index, batch.captions, batch.triplets, batch.tags, detection_results):
            video_unmapped_graphs[idx] = build_frame_scene_graph(
                captions,
                triplets,
                det_results,
                overlap_iou_threshold=overlap_iou_threshold,
                tags=tags,
                relations_mapper=relations_mapper,
                use_spatial_heuristics=use_spatial_heuristics,
                double_detection_threshold=double_detection_threshold,
            )

            # Map the graph over the closed set
            mapped_objects = [
                mapped_obj if og_obj != CAMERA_WEARER_LABEL else "main actor" for mapped_obj, og_obj in zip(objects_mapper(video_unmapped_graphs[idx]["obj"])[1], video_unmapped_graphs[idx]["obj"])
            ]
            video_mapped_graphs[idx] = {
                "obj": mapped_objects,
                "rel": relations_mapper(video_unmapped_graphs[idx]["rel"])[1],
                **{k: video_unmapped_graphs[idx][k] for k in video_unmapped_graphs[idx] if k not in ["obj", "rel"]},
            }

            video_unmapped_graphs[idx]["obj"] = [obj if obj != CAMERA_WEARER_LABEL else "main actor" for obj in video_unmapped_graphs[idx]["obj"]]

            num_empty_graphs += 1 if len(video_unmapped_graphs[idx]["obj"]) == 0 else 0
            total_num_graphs += 1

        pbar.set_description("Processing frames (empty graphs: %d / %d, %.2f%%)" % (num_empty_graphs, total_num_graphs, num_empty_graphs / total_num_graphs * 100))

    print(f"Number of empty graphs: {num_empty_graphs} / {total_num_graphs} ({num_empty_graphs / total_num_graphs:.2%})")

    # Save the entire video graphs
    logger.info(f"Saving video graph to {output_path}.json...")
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(video_unmapped_graphs, f, indent=2)

    logger.info(f"Saving video mapped graph to {output_path}_mapped.json...")
    with open(f"{output_path}_mapped.json", "w", encoding="utf-8") as f:
        json.dump(video_mapped_graphs, f, indent=2)
