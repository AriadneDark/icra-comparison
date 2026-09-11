"""Bounding box utilities for filtering heuristics in build.py"""

import torch

from typing import List, Union


def compute_iou(box1: Union[torch.Tensor, List[float]], box2: Union[torch.Tensor, List[float]]):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    The bounding boxes must be in the format [y1, x1, y2, x2]
    """
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    # Compute intersection area
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_min = max(x1_min, x2_min)
    intersect_y_max = min(y1_max, y2_max)
    intersect_x_max = min(x1_max, x2_max)

    if intersect_y_max <= intersect_y_min or intersect_x_max <= intersect_x_min:
        return 0.0

    intersect_area = (intersect_y_max - intersect_y_min) * (intersect_x_max - intersect_x_min)

    # Compute union area
    box1_area = (y1_max - y1_min) * (x1_max - x1_min)
    box2_area = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = box1_area + box2_area - intersect_area
    return intersect_area / union_area if union_area > 0 else 0.0


def compute_area(box: Union[torch.Tensor, List[float]]):
    """Compute the area of a bounding box given in the format xyxy"""
    y_min, x_min, y_max, x_max = box
    return max(0, y_max - y_min) * max(0, x_max - x_min)


def compute_area_intersection(box1: Union[torch.Tensor, List[float]], box2: Union[torch.Tensor, List[float]]):
    """Compute the area of intersection between two bounding boxes given in the xyxy format"""
    return max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
