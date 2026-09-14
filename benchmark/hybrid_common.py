"""Shared utilities for human-calibrated evaluation of task-role scene graphs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .normalize import ROLES, normalize_ours, normalize_sgego, normalize_svg2
except ImportError:
    from normalize import ROLES, normalize_ours, normalize_sgego, normalize_svg2


ROLE_ORDER = ("robot", "manipulated_object", "initial_support", "target")
METHODS = ("ours", "sg_ego", "svg2")
STRICT_ONTOLOGY_KEY = "\0drop_unknown"


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def resolve_study_path(study_root: str | Path, value: str | Path) -> Path:
    """Resolve a frozen-study resource, including legacy host-absolute paths."""
    root = Path(study_root).resolve()
    path = Path(value)
    if not path.is_absolute():
        return root / path
    if path.exists():
        return path
    fallback = root / path.name
    return fallback if fallback.exists() else path


def episode_stem(relative_path: str) -> str:
    return relative_path.replace("/", "__")


def normalize_text(value: str) -> str:
    value = re.sub(r"[_-]+", " ", str(value).strip().casefold())
    return re.sub(r"\s+", " ", value)


def load_ontology(path: str | Path | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if path is None:
        return aliases
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Predicate ontology must be a JSON object")
    if payload.get("_drop_unknown") is True:
        aliases[STRICT_ONTOLOGY_KEY] = "true"
    for canonical, values in payload.items():
        if canonical.startswith("_"):
            continue
        if not isinstance(values, list):
            raise ValueError(f"Aliases for predicate {canonical!r} must be a list")
        aliases[normalize_text(canonical)] = normalize_text(canonical)
        for value in values:
            aliases[normalize_text(value)] = normalize_text(canonical)
    return aliases


def canonical_predicate(value: str, ontology: dict[str, str]) -> str:
    normalized = normalize_text(value)
    if normalized in ontology:
        return ontology[normalized]
    return "" if ontology.get(STRICT_ONTOLOGY_KEY) == "true" else normalized


def ontology_predicates(ontology: dict[str, str]) -> list[str]:
    """Return the closed set of canonical predicates exposed to the judge."""
    return sorted({value for key, value in ontology.items() if key != STRICT_ONTOLOGY_KEY})


def intervals_from_frames(frame_ids: Iterable[int]) -> list[list[int]]:
    values = sorted(set(int(value) for value in frame_ids))
    if not values:
        return []
    intervals: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            intervals.append([start, previous])
            start = value
        previous = value
    intervals.append([start, previous])
    return intervals


def frames_from_intervals(intervals: Iterable[Iterable[int]], frame_count: int) -> set[int]:
    frames: set[int] = set()
    for interval in intervals:
        values = list(interval)
        if len(values) != 2:
            continue
        start, end = sorted((int(values[0]), int(values[1])))
        frames.update(range(max(0, start), min(frame_count - 1, end) + 1))
    return frames


def collapse_graph(graph: dict[str, Any], ontology: dict[str, str]) -> dict[str, Any]:
    role_frames: dict[str, list[int]] = defaultdict(list)
    relation_frames: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for frame in graph.get("frames", []):
        frame_index = int(frame["frame_index"])
        for role in frame.get("nodes", []):
            if role in ROLES:
                role_frames[role].append(frame_index)
        for edge in frame.get("edges", []):
            if len(edge) != 3 or edge[0] not in ROLES or edge[2] not in ROLES:
                continue
            # The four role nodes denote distinct task entities. Self-edges do
            # not express a useful role-to-role fact and only burden annotation.
            if edge[0] == edge[2]:
                continue
            predicate = canonical_predicate(edge[1], ontology)
            if not predicate:
                continue
            key = (edge[0], predicate, edge[2])
            relation_frames[key].append(frame_index)
    return {
        "frame_count": int(graph["frame_count"]),
        "role_intervals": {role: intervals_from_frames(ids) for role, ids in role_frames.items()},
        "relations": [
            {"subject": s, "predicate": p, "object": o, "intervals": intervals_from_frames(ids)}
            for (s, p, o), ids in sorted(relation_frames.items())
        ],
    }


def method_paths(root: Path, method: str, relative_path: str) -> dict[str, Path]:
    stem = episode_stem(relative_path)
    if method == "ours":
        candidates = [
            root / "scene_graphs" / f"{stem}.json",
            root / "scenes" / relative_path / "scene_graph.json",
        ]
        return {"graph": next((path for path in candidates if path.exists()), candidates[0])}
    if method == "sg_ego":
        return {
            "graph": root / "video_graphs" / "goal_roles" / f"{stem}.json",
            "frames": root / "frame_graphs" / "goal_roles" / f"{stem}.json",
        }
    candidates = [root / stem / "stage6_scene_graph.json", root / f"{stem}.json"]
    return {"graph": next((path for path in candidates if path.exists()), candidates[0])}


def _majority_role_labels(method: str, data: dict[str, Any], frame_data: dict[str, Any] | None) -> dict[str, str]:
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    if method == "ours":
        for frame in data.get("frames", []):
            for node in frame.get("nodes", []):
                role = node.get("role") or node.get("entity_id")
                label = str(node.get("canonical_name") or "").strip()
                if role in ROLES and label:
                    labels[role][label] += 1
    elif method == "sg_ego" and frame_data is not None:
        for frame in frame_data.values():
            if not isinstance(frame, dict):
                continue
            for role, label in zip(frame.get("role", []), frame.get("obj", [])):
                if role in ROLES and str(label).strip():
                    labels[role][str(label).strip()] += 1
    elif method == "svg2":
        for obj in data.get("objects", []):
            role, label = obj.get("role"), str(obj.get("name") or "").strip()
            if role in ROLES and label:
                labels[role][label] += 1
    return {role: counts.most_common(1)[0][0] for role, counts in labels.items() if counts}


def _sample_evidence(entries: list[dict[str, Any]], maximum: int = 5) -> list[dict[str, Any]]:
    if len(entries) <= maximum:
        return entries
    indices = sorted({round(index * (len(entries) - 1) / (maximum - 1)) for index in range(maximum)})
    return [entries[index] for index in indices]


def _role_evidence(method: str, data: dict[str, Any], frame_data: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if method == "ours":
        for frame in data.get("frames", []):
            for node in frame.get("nodes", []):
                role = node.get("role") or node.get("entity_id")
                box = node.get("bbox_xyxy")
                if role in ROLES and box and len(box) == 4:
                    evidence[role].append({
                        "frame_index": int(frame["frame_index"]), "bbox": box, "normalized": False,
                    })
    elif method == "sg_ego" and frame_data is not None:
        for frame_index, frame in frame_data.items():
            if not isinstance(frame, dict):
                continue
            for role, box in zip(frame.get("role", []), frame.get("bbox", [])):
                if role in ROLES and box and len(box) == 4:
                    evidence[role].append({
                        "frame_index": int(frame_index), "bbox": box, "normalized": True,
                    })
    elif method == "svg2":
        for obj in data.get("objects", []):
            role = obj.get("role")
            if role not in ROLES:
                continue
            for frame_index, box in obj.get("trajectory", {}).get("boxes", {}).items():
                if box and len(box) == 4:
                    evidence[role].append({
                        "frame_index": int(frame_index), "bbox": box, "normalized": False,
                    })
    return {
        role: _sample_evidence(sorted(values, key=lambda item: item["frame_index"]))
        for role, values in evidence.items()
    }


def load_method_output(
    method: str, root: Path, relative_path: str, frame_count: int, ontology: dict[str, str]
) -> dict[str, Any] | None:
    paths = method_paths(root, method, relative_path)
    if not paths["graph"].exists():
        return None
    data = load_json(paths["graph"])
    frame_data = load_json(paths["frames"]) if paths.get("frames") and paths["frames"].exists() else None
    if method == "ours":
        normalized = normalize_ours(data)
    elif method == "sg_ego":
        normalized = normalize_sgego(data, frame_count)
    else:
        normalized = normalize_svg2(data)
    collapsed = collapse_graph(normalized, ontology)
    collapsed["role_labels"] = _majority_role_labels(method, data, frame_data)
    collapsed["role_evidence"] = _role_evidence(method, data, frame_data)
    collapsed["source_path"] = str(paths["graph"])
    return collapsed


def fact_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join((kind, *parts)).encode("utf-8")).hexdigest()[:12]
    return f"{kind[0]}_{digest}"


def build_candidates(method_outputs: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    roles: list[dict[str, Any]] = []
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    method_facts: dict[str, dict[str, list[str]]] = {
        method: {"roles": [], "relations": []} for method in METHODS
    }
    for method, output in method_outputs.items():
        if output is None:
            continue
        for role, label in output.get("role_labels", {}).items():
            # Keep each method's physical track separate even when text labels
            # coincide (e.g. two different plates both named "plate").
            record = {
                "id": fact_id("role", role, method), "kind": "role", "role": role,
                "label": label, "sources": [method],
                "source_intervals": {method: output.get("role_intervals", {}).get(role, [])},
                "evidence_boxes": output.get("role_evidence", {}).get(role, []),
            }
            roles.append(record)
            # Agreement sampling uses a semantic key, while the annotation
            # claim id remains method-specific to preserve physical identity.
            method_facts[method]["roles"].append(f"role:{role}:{normalize_text(label)}")
        for relation in output.get("relations", []):
            key = (relation["subject"], relation["predicate"], relation["object"])
            record = relations.setdefault(key, {
                "id": fact_id("relation", *key), "kind": "relation",
                "subject": key[0], "predicate": key[1], "object": key[2],
                "sources": [], "source_intervals": {},
            })
            record["sources"].append(method)
            record["source_intervals"][method] = relation["intervals"]
            method_facts[method]["relations"].append(f"relation:{'|'.join(key)}")
    facts = sorted([*roles, *relations.values()], key=lambda value: value["id"])
    return {"facts": facts, "method_facts": method_facts}


def disagreement_score(method_facts: dict[str, dict[str, list[str]]]) -> float:
    sets = {
        method: set(values["roles"]) | set(values["relations"])
        for method, values in method_facts.items()
    }
    distances = []
    for index, first in enumerate(METHODS):
        for second in METHODS[index + 1:]:
            union = sets[first] | sets[second]
            distances.append(1.0 - len(sets[first] & sets[second]) / len(union) if union else 0.0)
    return sum(distances) / len(distances)


def parse_model_object(content: str) -> dict[str, Any]:
    """Parse a JSON/Python/YAML-style object from a model response."""
    text = content.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    starts = [index for index, char in enumerate(text) if char == "{"]
    candidates = [text]
    if starts and "}" in text:
        candidates.insert(0, text[starts[0]:text.rfind("}") + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, dict):
                    return value
            except Exception as exc:
                last_error = exc
    if last_error:
        raise last_error
    raise ValueError("Model response does not contain an object")


def task_family(goal: str) -> str:
    value = normalize_text(goal)
    patterns = (
        ("pour", ("pour", "decant")),
        ("open_close", ("open", "close", "drawer", "door")),
        ("stack", ("stack",)),
        ("clean", ("wipe", "clean", "sweep")),
        ("press", ("press", "push button", "switch")),
        ("pick_place", ("pick", "place", "put", "move", "transfer", "insert")),
    )
    for family, words in patterns:
        if any(word in value for word in words):
            return family
    return "other"
