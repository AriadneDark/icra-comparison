#!/usr/bin/env python3
"""Compute micro/macro precision and recall on the four task roles only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:  # supports both ``python benchmark/evaluate.py`` and ``python -m benchmark.evaluate``
    from .normalize import ROLES, load_json, normalize_ours, normalize_sgego, normalize_svg2
except ImportError:
    from normalize import ROLES, load_json, normalize_ours, normalize_sgego, normalize_svg2


def _counts(predicted: set, reference: set) -> tuple[int, int, int]:
    return len(predicted & reference), len(predicted - reference), len(reference - predicted)


def _scores(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def compare_episode(prediction: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if prediction["frame_count"] != reference["frame_count"]:
        raise ValueError(
            f"frame_count mismatch: prediction={prediction['frame_count']}, "
            f"reference={reference['frame_count']}"
        )
    totals = {"nodes": [0, 0, 0], "triplets": [0, 0, 0]}
    for pred_frame, ref_frame in zip(prediction["frames"], reference["frames"]):
        p_nodes = set(pred_frame.get("nodes", [])) & ROLES
        r_nodes = set(ref_frame.get("nodes", [])) & ROLES
        p_edges = {tuple(x) for x in pred_frame.get("edges", []) if x[0] in ROLES and x[2] in ROLES}
        r_edges = {tuple(x) for x in ref_frame.get("edges", []) if x[0] in ROLES and x[2] in ROLES}
        for key, values in (("nodes", _counts(p_nodes, r_nodes)), ("triplets", _counts(p_edges, r_edges))):
            totals[key] = [a + b for a, b in zip(totals[key], values)]
    return {key: _scores(*values) for key, values in totals.items()}


def _prediction_path(root: Path, method: str, relative_path: str) -> Path:
    dataset, scene = relative_path.split("/", 1)
    stem = f"{dataset}__{scene}"
    if method == "ours":
        candidates = [root / "scene_graphs" / f"{stem}.json", root / "scenes" / relative_path / "scene_graph.json"]
    elif method == "sg_ego":
        candidates = [root / "video_graphs" / "goal_roles" / f"{stem}.json", root / f"{stem}.json"]
    else:
        candidates = [root / stem / "stage6_scene_graph.json", root / f"{stem}.json"]
    return next((p for p in candidates if p.exists()), candidates[0])


def _normalize(method: str, data: dict[str, Any], frame_count: int) -> dict[str, Any]:
    if method == "ours":
        return normalize_ours(data)
    if method == "sg_ego":
        return normalize_sgego(data, frame_count)
    return normalize_svg2(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-root", required=True,
                        help="Directory of human task_role_graph_v1 files named dataset__scene.json")
    parser.add_argument("--ours-root", required=True)
    parser.add_argument("--sg-ego-root", required=True)
    parser.add_argument("--svg2-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    if len(manifest.get("episodes", [])) != 100:
        raise SystemExit("The benchmark manifest must contain exactly 100 episodes")
    roots = {"ours": Path(args.ours_root), "sg_ego": Path(args.sg_ego_root), "svg2": Path(args.svg2_root)}
    rows, missing = [], []
    aggregate = {m: {k: [0, 0, 0] for k in ("nodes", "triplets")} for m in roots}
    macro = {m: {k: [] for k in ("nodes", "triplets")} for m in roots}
    for episode in manifest["episodes"]:
        rel = episode["relative_path"]
        stem = rel.replace("/", "__")
        ref_path = Path(args.reference_root) / f"{stem}.json"
        if not ref_path.exists():
            missing.append({"episode": rel, "method": "reference", "path": str(ref_path)})
            continue
        reference = load_json(ref_path)
        for method, root in roots.items():
            pred_path = _prediction_path(root, method, rel)
            if not pred_path.exists():
                missing.append({"episode": rel, "method": method, "path": str(pred_path)})
                continue
            result = compare_episode(_normalize(method, load_json(pred_path), int(episode["frame_count"])), reference)
            row = {"episode": rel, "method": method}
            for kind in ("nodes", "triplets"):
                row.update({f"{kind}_{key}": value for key, value in result[kind].items()})
                macro[method][kind].append((result[kind]["precision"], result[kind]["recall"]))
                aggregate[method][kind] = [a + result[kind][b] for a, b in zip(aggregate[method][kind], ("tp", "fp", "fn"))]
            rows.append(row)

    summary = {}
    for method in roots:
        summary[method] = {}
        for kind in ("nodes", "triplets"):
            micro = _scores(*aggregate[method][kind])
            values = macro[method][kind]
            summary[method][kind] = {
                "micro": micro,
                "macro_precision": sum(x[0] for x in values) / len(values) if values else None,
                "macro_recall": sum(x[1] for x in values) / len(values) if values else None,
                "evaluated_episodes": len(values),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "missing": missing}, indent=2), encoding="utf-8")
    if rows:
        with open(output.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if missing:
        raise SystemExit(f"Incomplete benchmark: {len(missing)} artifacts missing; see {output}")


if __name__ == "__main__":
    main()
