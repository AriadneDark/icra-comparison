#!/usr/bin/env python3
"""Evaluate frame-level task-role tracks on the frozen human-100 videos."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from hybrid_common import ROLE_ORDER, fact_id, frames_from_intervals, load_json, write_json


def scene_path(root: Path, relative_path: str, name: str) -> Path:
    candidates = (root / "scenes" / relative_path / name, root / relative_path / name)
    return next((path for path in candidates if path.exists()), candidates[0])


def visible_tracks(path: Path) -> dict[str, set[int]]:
    payload = load_json(path)
    result: dict[str, set[int]] = defaultdict(set)
    for track in payload.get("tracks", []):
        role = track.get("role")
        if role not in ROLE_ORDER:
            continue
        for frame in track.get("frames", []):
            if frame.get("status") == "visible":
                result[role].add(int(frame["frame_index"]))
    return result


def add(first: tuple[int, int, int], second: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def scores(counts: tuple[int, int, int]) -> dict[str, float | int]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def full_track_truth(annotation: dict[str, Any], role: str, frame_count: int) -> tuple[set[int], set[int]]:
    claim = annotation.get("claim_labels", {}).get(fact_id("role", role, "ours"))
    if not claim or claim.get("label") == "uncertain":
        return set(), set(range(frame_count))
    if claim.get("label") == "no":
        return set(), set()
    return frames_from_intervals(claim.get("intervals", []), frame_count), set()


def audited_track_truth(
    audit: dict[str, Any], variant: str, role: str, frame_count: int,
) -> tuple[set[int], set[int]]:
    value = audit.get("tracks", {}).get(variant, {}).get(role)
    if not isinstance(value, dict) or value.get("verdict") not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Missing audit for {variant}/{role}")
    if value["verdict"] == "uncertain":
        return set(), set(range(frame_count))
    if value["verdict"] == "no":
        return set(), set()
    return (
        frames_from_intervals(value.get("correct_intervals", []), frame_count),
        frames_from_intervals(value.get("uncertain_intervals", []), frame_count),
    )


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--full-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--variants", nargs="+", default=[
        "full", "qwen_text_sam3", "qwen_box_sam3", "qwen_box_sam2",
        "qwen_text_sam31", "qwen_text_sam31_sam2",
    ])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    study = load_json(args.study_root / "study_manifest.json")
    records = [r for r in study["episodes"] if r["evaluation_split"].startswith("human_")]
    totals = {variant: (0, 0, 0) for variant in args.variants}
    role_totals = {
        variant: {role: (0, 0, 0) for role in ROLE_ORDER} for variant in args.variants
    }
    video_counts: dict[str, dict[str, tuple[int, int, int]]] = {}
    skipped: list[str] = []

    for record in records:
        video_id, frame_count = record["video_id"], int(record["frame_count"])
        annotation_path = args.study_root / "human_annotations" / args.annotator / f"{video_id}.json"
        audit_path = args.audit_root / args.annotator / f"{video_id}.json"
        if not annotation_path.exists():
            skipped.append(f"{video_id}: missing human annotation")
            continue
        annotation = load_json(annotation_path)
        if not annotation.get("complete"):
            skipped.append(f"{video_id}: human annotation is unfinished")
            continue
        audit = load_json(audit_path) if audit_path.exists() else {}
        per_variant: dict[str, tuple[int, int, int]] = {}
        per_variant_roles: dict[str, dict[str, tuple[int, int, int]]] = {}
        try:
            for variant in args.variants:
                root = args.full_root if variant == "full" else args.run_root / variant
                tracks_path = scene_path(root, record["relative_path"], "tracks.json")
                predicted = visible_tracks(tracks_path)
                variant_counts = (0, 0, 0)
                variant_role_counts: dict[str, tuple[int, int, int]] = {}
                for role in ROLE_ORDER:
                    role_gt = annotation.get("roles", {}).get(role, {})
                    if role_gt.get("status") == "ambiguous":
                        continue
                    ground_truth = (
                        frames_from_intervals(role_gt.get("visible_intervals", []), frame_count)
                        if role_gt.get("status") == "present" else set()
                    )
                    correct, uncertain = (
                        full_track_truth(annotation, role, frame_count)
                        if variant == "full"
                        else audited_track_truth(audit, variant, role, frame_count)
                    )
                    counts = [0, 0, 0]
                    for frame_index in set(range(frame_count)) - uncertain:
                        has_prediction = frame_index in predicted.get(role, set())
                        matched = has_prediction and frame_index in correct and frame_index in ground_truth
                        counts[0] += int(matched)
                        counts[1] += int(has_prediction and not matched)
                        counts[2] += int(frame_index in ground_truth and not matched)
                    value = tuple(counts)  # type: ignore[assignment]
                    variant_role_counts[role] = value
                    variant_counts = add(variant_counts, value)
                per_variant[variant] = variant_counts
                per_variant_roles[variant] = variant_role_counts
        except (FileNotFoundError, ValueError) as error:
            if not args.allow_incomplete:
                raise
            skipped.append(f"{video_id}: {error}")
            continue
        for variant, counts in per_variant.items():
            totals[variant] = add(totals[variant], counts)
            for role, value in per_variant_roles[variant].items():
                role_totals[variant][role] = add(role_totals[variant][role], value)
        video_counts[video_id] = per_variant

    if not video_counts:
        raise SystemExit("No fully annotated videos could be evaluated")
    result: dict[str, Any] = {
        "schema_version": "object_ablation_metrics_v1",
        "scope": list(ROLE_ORDER),
        "unit": "video-role-frame",
        "evaluated_video_count": len(video_counts),
        "skipped": skipped,
        "methods": {},
    }
    rng = random.Random(args.seed)
    video_ids = sorted(video_counts)
    for variant in args.variants:
        method = scores(totals[variant])
        method["per_role"] = {role: scores(role_totals[variant][role]) for role in ROLE_ORDER}
        macro_values = [scores(video_counts[v][variant]) for v in video_ids]
        method["macro"] = {
            metric: sum(float(value[metric]) for value in macro_values) / len(macro_values)
            for metric in ("precision", "recall", "f1")
        }
        if variant != "full" and "full" in args.variants:
            samples = {metric: [] for metric in ("precision", "recall", "f1")}
            for _ in range(args.bootstrap_samples):
                drawn = [rng.choice(video_ids) for _ in video_ids]
                variant_total = full_total = (0, 0, 0)
                for video_id in drawn:
                    variant_total = add(variant_total, video_counts[video_id][variant])
                    full_total = add(full_total, video_counts[video_id]["full"])
                variant_scores, full_scores = scores(variant_total), scores(full_total)
                for metric in samples:
                    samples[metric].append(float(variant_scores[metric]) - float(full_scores[metric]))
            method["delta_vs_full"] = {
                metric: {
                    "point": float(method[metric]) - float(scores(totals["full"])[metric]),
                    "bootstrap_95_ci": [percentile(values, 0.025), percentile(values, 0.975)],
                }
                for metric, values in samples.items()
            }
        result["methods"][variant] = method
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
