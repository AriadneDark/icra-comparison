from typing import Dict, List, Set

from grounding.utils.bbox import compute_iou, compute_area, compute_area_intersection

CAMERA_WEARER_LABEL = "##camera_wearer##"


def apply_spatial_heuristics(candidate_triplets: Set[tuple], bboxes: List[List[float]], relations_mapping: Dict[str, str]):
    """Apply spatial heuristics to validate "in", "on", "left", and "right" relations based on the bounding boxes of the subject and object.

    Parameters
    ----------
    candidate_triplets : List[tuple]
        List of candidate grounded triplets in the format (idx_subj, rel, idx_obj)
    bboxes : List[List[float]]
        List of bounding boxes in the format [y1, x1, y2, x2]
    relations_mapping : Dict[str, str]
        Dictionary mapping relation labels to the closed set in order to apply the spatial filtering heuristics (e.g., {"in": "in", "on": "on", "left of": "left", "right of": "right", ...})

    Returns
    -------
    Set[tuple]
        Set of valid candidate triplets after applying spatial heuristics
    """

    valid_candidate_triplets = set()

    for idx_subj, rel, idx_obj in candidate_triplets:

        if relations_mapping[rel] == "in":
            # check that the at least half of the bbox of the object is included in the bbox of the subject
            bbox_subj, bbox_obj = bboxes[idx_subj], bboxes[idx_obj]
            area_within = max(0, min(bbox_subj[2], bbox_obj[2]) - max(bbox_subj[0], bbox_obj[0])) * max(0, min(bbox_subj[3], bbox_obj[3]) - max(bbox_subj[1], bbox_obj[1]))
            area_subj = (bbox_subj[2] - bbox_subj[0]) * (bbox_subj[3] - bbox_subj[1])
            if area_within / area_subj >= 0.5:
                valid_candidate_triplets.add((idx_subj, rel, idx_obj))

        elif relations_mapping[rel] == "right":
            # check that the object is to the right of the subject
            bbox_subj, bbox_obj = bboxes[idx_subj], bboxes[idx_obj]
            center_subj = (bbox_subj[0] + bbox_subj[2]) / 2
            center_obj = (bbox_obj[0] + bbox_obj[2]) / 2
            if center_subj >= center_obj:
                valid_candidate_triplets.add((idx_subj, rel, idx_obj))

        elif relations_mapping[rel] == "left":
            # check that the object is to the left of the subject
            bbox_subj, bbox_obj = bboxes[idx_subj], bboxes[idx_obj]
            center_subj = (bbox_subj[0] + bbox_subj[2]) / 2
            center_obj = (bbox_obj[0] + bbox_obj[2]) / 2
            if center_subj <= center_obj:
                valid_candidate_triplets.add((idx_subj, rel, idx_obj))

        elif relations_mapping[rel] == "on":
            # check that the object is on the subject
            bbox_subj, bbox_obj = bboxes[idx_subj], bboxes[idx_obj]
            # enlarge bbox_obj by 10% in all directions to account for small detection errors
            bbox_obj_enlarged = [
                max(bbox_obj[0] - 0.1 * (bbox_obj[2] - bbox_obj[0]), 0.0),
                max(bbox_obj[1] - 0.1 * (bbox_obj[3] - bbox_obj[1]), 0.0),
                min(bbox_obj[2] + 0.1 * (bbox_obj[2] - bbox_obj[0]), 1.0),
                min(bbox_obj[3] + 0.1 * (bbox_obj[3] - bbox_obj[1]), 1.0),
            ]
            if compute_iou(bbox_subj, bbox_obj_enlarged) > 0:
                valid_candidate_triplets.add((idx_subj, rel, idx_obj))

        else:
            valid_candidate_triplets.add((idx_subj, rel, idx_obj))

    return valid_candidate_triplets


def intra_caption_filtering(captions, detection_results):
    """Apply a set of filtering heuristics to prune the set of detected entities for each triplet.
    
    For each caption, we iterate over all the its detections obtained from GroudingDINO and we filter out detections that are likely to be duplicates of the same object.
    Given a detection d_i for the same caption, we keep detection d_i if no other detection d_j for the same caption satisfies at least one of the following conditions:
     - d_j has a significantly higher confidence score than d_i (threshold is chose arbitrarily)
     - d_j has a large iou with d_i and a higher confidence score than d_i
     - d_j has a large iou with d_i and a smaller bounding box area than d_i (i.e., d_j is mostly contained in d_i) and a higher confidence score than d_i.
    
    Parameters
    ----------
    captions : list of tuple
        List of triplets (subject, relation, object) for each image.
    detection_results : list of dict
        List of detection results for each image, where each dict contains keys "text_labels", "scores", "boxes", and "type".

    Returns
    -------
    list of dict
        Filtered detection results for each image.
    """
    
    filtered_detection_results = []
    
    # Iterate over the captions and their corresponding detection results for each caption
    for (raw_caption, triplet_dets) in zip(captions, detection_results):
        
        # For each caption, keep only the detections that satisfy all the constraints
    
        # Accumulate here the valid detections that survive the filtering heuristics, and that will be used for graph construction
        text_labels, scores, boxes, det_types = [], [], [], []
        
        # Iterate over all the detections for this caption (i.e., for this triplet)
        for det_idx, (det_label, det_score, det_bbox, det_type) in enumerate(zip(triplet_dets["text_labels"], triplet_dets["scores"], triplet_dets["boxes"], triplet_dets["type"])):
            
            # Camera wearer shortcut: replace the detected bounding box with the full frame
            is_cw = (det_type == "subject" and raw_caption[0].lower() == "camera_wearer") or (det_type == "object" and raw_caption[2].lower() == "camera_wearer")
            if is_cw:
                text_labels.append(CAMERA_WEARER_LABEL)
                scores.append(1.0)
                boxes.append([0.0, 0.0, 1.0, 1.0])
                det_types.append(det_type)
                continue

            # Iterate over all the other detections
            invalid = False
            for other_det_idx, (other_det_label, other_det_score, other_det_bbox, other_det_type) in enumerate(zip(triplet_dets["text_labels"], triplet_dets["scores"], triplet_dets["boxes"], triplet_dets["type"])):
                
                # Skip the same detection (pointless)
                if det_idx == other_det_idx:
                    continue

                # Skip detections with the same label
                if det_label != other_det_label:
                    continue
                
                # Skip detections with different types (e.g., one is subject and the other is object)
                if det_type != other_det_type:
                    continue

                # Compute the area for this bbox and the other bbox, and the area of their intersection
                area_bbox, area_other, area_intersection = compute_area(det_bbox), compute_area(other_det_bbox), compute_area_intersection(det_bbox, other_det_bbox)
                
                assert area_bbox > 0 and area_other > 0, f"Invalid bounding box with zero area: {det_bbox} or {other_det_bbox}"
                assert area_intersection >= 0, f"Invalid intersection area: {area_intersection} for bboxes {det_bbox} and {other_det_bbox}"
                assert area_intersection <= area_bbox and area_intersection <= area_other, f"Invalid intersection area: {area_intersection} greater than bbox area {area_bbox} or other bbox area {area_other}"
                
                if (
                    # No intersection but one or much more confident than the other (threshold is chose arbitrarily)
                    (other_det_score > det_score + 0.05)
                    # One detection is overlapped with the other
                    or (compute_iou(det_bbox, other_det_bbox) >= 0.5 and other_det_score > det_score)
                    # One of the detections is mostly contained in the other -> take the detection corresponding to the smallest bbox
                    or ((area_intersection / area_bbox >= 0.5) and other_det_score > det_score)
                ):
                    invalid = True
                    break

            if not invalid:
                text_labels.append(det_label)
                scores.append(det_score)
                boxes.append(det_bbox)
                det_types.append(det_type)
                
        filtered_detection_results.append({
            "text_labels": text_labels,
            "scores": scores,
            "boxes": boxes,
            "type": det_types
        })
        
    return filtered_detection_results