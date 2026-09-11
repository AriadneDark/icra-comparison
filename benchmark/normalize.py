"""Normalize our method, adapted SG-Ego, and adapted SVG2 to one frame schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROLES = frozenset({"robot", "manipulated_object", "initial_support", "target"})


def _predicate(value: str) -> str:
    value = re.sub(r"[_-]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", value)


def _empty_frames(count: int) -> list[dict[str, Any]]:
    return [{"frame_index": i, "nodes": [], "edges": []} for i in range(count)]


def _dedupe(frame: dict[str, Any]) -> None:
    frame["nodes"] = sorted(set(frame["nodes"]))
    frame["edges"] = [list(x) for x in sorted({tuple(x) for x in frame["edges"]})]


def normalize_ours(data: dict[str, Any]) -> dict[str, Any]:
    frames = _empty_frames(int(data["frame_count"]))
    for source in data.get("frames", []):
        idx = int(source["frame_index"])
        if not 0 <= idx < len(frames):
            continue
        visible = {
            n.get("role") or n.get("entity_id")
            for n in source.get("nodes", [])
            if n.get("status", "visible") == "visible"
        } & ROLES
        frames[idx]["nodes"].extend(visible)
        for edge in source.get("state_edges", []):
            subj, obj = edge.get("subject"), edge.get("object")
            if subj in ROLES and obj in ROLES:
                frames[idx]["edges"].append([subj, _predicate(edge.get("relation", "")), obj])
        for action in source.get("actions", []):
            subj, obj = action.get("actor"), action.get("object")
            if subj in ROLES and obj in ROLES:
                frames[idx]["edges"].append([subj, _predicate(action.get("action", "")), obj])
        _dedupe(frames[idx])
    return {"schema_version": "task_role_graph_v1", "frame_count": len(frames), "frames": frames}


def normalize_sgego(data: dict[str, Any], frame_count: int) -> dict[str, Any]:
    """Expand SG-Ego window graphs over their half-open frame windows."""
    frames = _empty_frames(frame_count)
    for window, graph in data.items():
        match = re.fullmatch(r"(\d+)-(\d+)", window)
        if not match or not isinstance(graph, dict):
            continue
        start, end = map(int, match.groups())
        objects = graph.get("obj", [])
        roles = graph.get("role", [])
        role_by_index = {
            i: role for i, role in enumerate(roles)
            if role in ROLES and i < len(objects)
        }
        edges = []
        for pair, relation in zip(graph.get("pair", []), graph.get("rel", [])):
            if len(pair) != 2:
                continue
            subj, obj = role_by_index.get(int(pair[0])), role_by_index.get(int(pair[1]))
            if subj in ROLES and obj in ROLES:
                edges.append([subj, _predicate(relation), obj])
        for idx in range(max(0, start), min(frame_count, end)):
            frames[idx]["nodes"].extend(role_by_index.values())
            frames[idx]["edges"].extend(edges)
            _dedupe(frames[idx])
    return {"schema_version": "task_role_graph_v1", "frame_count": frame_count, "frames": frames}


def normalize_svg2(data: dict[str, Any]) -> dict[str, Any]:
    frame_count = int(data["total_frames"])
    frames = _empty_frames(frame_count)
    role_by_id: dict[int, str] = {}
    for obj in data.get("objects", []):
        role = obj.get("role")
        if role not in ROLES:
            continue
        oid = int(obj["object_id"])
        role_by_id[oid] = role
        active = obj.get("trajectory", {}).get("frames", [])
        for idx in active:
            if 0 <= int(idx) < frame_count:
                frames[int(idx)]["nodes"].append(role)

    relations = data.get("relationships", {})
    sampled = relations.get("sampled_frame_indices", [])
    for kind in ("temporal", "spatial"):
        for relation in relations.get(kind, []):
            if len(relation) < 4:
                continue
            subj = role_by_id.get(int(relation[0]))
            obj = role_by_id.get(int(relation[2]))
            if subj not in ROLES or obj not in ROLES:
                continue
            for interval in relation[3]:
                if len(interval) != 2 or not sampled:
                    continue
                lo, hi = sorted((int(interval[0]), int(interval[1])))
                if lo >= len(sampled):
                    continue
                start = int(sampled[lo])
                end = int(sampled[min(hi, len(sampled) - 1)])
                for idx in range(max(0, start), min(frame_count - 1, end) + 1):
                    frames[idx]["edges"].append([subj, _predicate(relation[1]), obj])
    for frame in frames:
        _dedupe(frame)
    return {"schema_version": "task_role_graph_v1", "frame_count": frame_count, "frames": frames}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

