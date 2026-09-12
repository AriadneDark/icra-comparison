#!/usr/bin/env python3
"""Create proposer/verifier VLM references for a prepared hybrid evaluation study."""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageDraw

try:
    from .hybrid_common import (
        ROLE_ORDER, canonical_predicate, episode_stem, fact_id, frames_from_intervals,
        intervals_from_frames, load_json, load_ontology, normalize_text, parse_model_object, write_json,
    )
except ImportError:
    from hybrid_common import (
        ROLE_ORDER, canonical_predicate, episode_stem, fact_id, frames_from_intervals,
        intervals_from_frames, load_json, load_ontology, normalize_text, parse_model_object, write_json,
    )


def sample_indices(count: int, maximum: int) -> list[int]:
    if maximum < 2:
        return [0] if count else []
    if count <= maximum:
        return list(range(count))
    return sorted({round(index * (count - 1) / (maximum - 1)) for index in range(maximum)})


def encode_image(path: Path, overlays: list[tuple[str, dict[str, Any]]] | None = None) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if overlays:
            draw = ImageDraw.Draw(image)
            line_width = max(2, image.width // 200)
            for row, (claim_id, evidence) in enumerate(overlays):
                box = [float(value) for value in evidence["bbox"]]
                if evidence.get("normalized"):
                    box = [
                        box[0] * image.width, box[1] * image.height,
                        box[2] * image.width, box[3] * image.height,
                    ]
                draw.rectangle(tuple(box), outline=(255, 220, 40), width=line_width)
                text = claim_id
                y = min(max(0, int(box[1]) - 16), max(0, image.height - 16))
                # Stagger labels when several methods provide overlapping tracks.
                y = min(image.height - 16, y + (row % 3) * 16)
                draw.rectangle((int(box[0]), y, int(box[0]) + 8 * len(text) + 4, y + 15), fill=(15, 15, 15))
                draw.text((int(box[0]) + 2, y + 1), text, fill=(255, 255, 255))
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def visual_content(
    prompt: str, images: list[Path], indices: list[int],
    overlay_by_frame: dict[int, list[tuple[str, dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame_index, path in zip(indices, images):
        content.extend([
            {"type": "text", "text": f"Original frame index: {frame_index}"},
            {"type": "image_url", "image_url": {
                "url": encode_image(path, (overlay_by_frame or {}).get(frame_index))
            }},
        ])
    return content


def model_request_args(model: str) -> dict[str, Any]:
    folded = model.casefold()
    if "gemma-4" in folded or "gemma4" in folded:
        return {
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    if "qwen" not in folded:
        return {"temperature": 0.0, "max_tokens": 4096}
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 4096,
        "extra_body": {"top_k": 0, "chat_template_kwargs": {"enable_thinking": False}},
    }


def is_local_endpoint(base_url: str) -> bool:
    """Return true for the unauthenticated judge endpoints used by this repo."""
    return (urlparse(base_url).hostname or "").casefold() in {
        "127.0.0.1", "localhost", "host.docker.internal", "gemma-judge",
    }


def messages_for_model(model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use Gemma 4's recommended image-before-text multimodal ordering."""
    if "gemma-4" not in model.casefold() and "gemma4" not in model.casefold():
        return messages
    prepared: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            prepared.append(message)
            continue
        images = [item for item in content if item.get("type") == "image_url"]
        other = [item for item in content if item.get("type") != "image_url"]
        prepared.append({**message, "content": images + other})
    return prepared


def request_object(client: Any, model: str, messages: list[dict[str, Any]], retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    malformed: str | None = None
    for attempt in range(retries):
        try:
            request_messages = messages_for_model(model, messages)
            if malformed is not None:
                request_messages = [
                    {
                        "role": "system",
                        "content": "Repair malformed JSON. Return only one valid JSON object and no Markdown.",
                    },
                    {"role": "user", "content": malformed},
                ]
            response = client.chat.completions.create(
                model=model, messages=request_messages, **model_request_args(model)
            )
            content = response.choices[0].message.content
            try:
                return parse_model_object(content)
            except Exception:
                malformed = content
                raise
        except Exception as exc:
            last_error = exc
            print(f"API attempt {attempt + 1}/{retries} failed: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"VLM request failed after {retries} attempts") from last_error


def proposer_prompt(goal: str, frame_count: int, sampled_frames: list[int] | None = None) -> str:
    sampled = sampled_frames if sampled_frames is not None else list(range(frame_count))
    return f"""Analyze this robot-manipulation video using only visible evidence.

Planning goal: {goal}
The original video has {frame_count} frames numbered 0 through {frame_count - 1}.
You are shown only original frames: {sampled}.

Identify only these task roles: robot, manipulated_object, initial_support, target.
Then list visually supported directed relations between those roles. Use concise predicates and
list only shown frame numbers where each fact is directly visible. Never infer unshown frames.
Do not infer that the goal succeeded merely from its text.

Return exactly one JSON object:
{{
  "roles": [
    {{"role": "robot", "label": "robot arm", "visible_frames": [0, 3, 6]}}
  ],
  "relations": [
    {{"subject": "robot", "predicate": "holding", "object": "manipulated_object",
      "frames": [10, 13]}}
  ]
}}
Use an empty list when nothing is visually supported. No Markdown or explanation."""


def sanitize_proposal(
    payload: dict[str, Any], frame_count: int, ontology: dict[str, str],
    sampled_frames: list[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(sampled_frames if sampled_frames is not None else range(frame_count))

    def direct_frames(item: dict[str, Any], key: str, legacy_key: str) -> set[int]:
        values = item.get(key)
        if isinstance(values, list):
            return {
                int(value) for value in values
                if isinstance(value, int) and not isinstance(value, bool) and int(value) in allowed
            }
        # Read old responses defensively, but retain only frames actually shown.
        return frames_from_intervals(item.get(legacy_key, []), frame_count) & allowed

    roles = []
    for item in payload.get("roles", []):
        if not isinstance(item, dict) or item.get("role") not in ROLE_ORDER:
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        frames = direct_frames(item, "visible_frames", "visible_intervals")
        roles.append({
            "role": item["role"], "label": label,
            "visible_intervals": _bounded_intervals(frames, frame_count),
        })
    relations = []
    for item in payload.get("relations", []):
        if not isinstance(item, dict):
            continue
        subject, obj = item.get("subject"), item.get("object")
        if subject not in ROLE_ORDER or obj not in ROLE_ORDER or subject == obj:
            continue
        predicate = canonical_predicate(str(item.get("predicate") or ""), ontology)
        if not predicate:
            continue
        frames = direct_frames(item, "frames", "intervals")
        relations.append({
            "subject": subject, "predicate": predicate, "object": obj,
            "intervals": _bounded_intervals(frames, frame_count),
        })
    return {"roles": roles, "relations": relations}


def _bounded_intervals(frames: set[int], frame_count: int) -> list[list[int]]:
    return intervals_from_frames(frame for frame in frames if 0 <= frame < frame_count)


def merge_proposal(base: dict[str, Any], proposal: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    facts = {fact["id"]: dict(fact) for fact in base["facts"]}
    for role in proposal["roles"]:
        key = (role["role"], normalize_text(role["label"]))
        identifier = fact_id("role", *key)
        record = facts.setdefault(identifier, {
            "id": identifier, "kind": "role", "role": role["role"], "label": role["label"],
            "sources": [], "source_intervals": {},
        })
        if "vlm_proposer" not in record["sources"]:
            record["sources"].append("vlm_proposer")
        record["source_intervals"]["vlm_proposer"] = role["visible_intervals"]
    for relation in proposal["relations"]:
        key = (relation["subject"], relation["predicate"], relation["object"])
        identifier = fact_id("relation", *key)
        record = facts.setdefault(identifier, {
            "id": identifier, "kind": "relation", "subject": key[0], "predicate": key[1],
            "object": key[2], "sources": [], "source_intervals": {},
        })
        if "vlm_proposer" not in record["sources"]:
            record["sources"].append("vlm_proposer")
        record["source_intervals"]["vlm_proposer"] = relation["intervals"]
    return {**base, "facts": sorted(facts.values(), key=lambda value: value["id"])}


def verifier_prompt(
    goal: str, facts: list[dict[str, Any]], frame_count: int, selected_frames: set[int]
) -> str:
    blind = []
    for fact in facts:
        if fact["kind"] == "role":
            claim = {
                "id": fact["id"], "kind": "role", "role": fact["role"], "label": fact["label"],
                "evidence_boxes": [
                    value for value in fact.get("evidence_boxes", [])
                    if int(value.get("frame_index", -1)) in selected_frames
                ],
            }
        else:
            claim = {
                "id": fact["id"], "kind": "relation", "subject": fact["subject"],
                "predicate": fact["predicate"], "object": fact["object"],
            }
        blind.append(claim)
    return f"""Independently verify atomic claims against the supplied robot-manipulation frames.
Method identities are hidden. Planning goal is provided only to define the task roles.
For role claims with evidence_boxes, judge whether the indicated physical track has that role.
The same claim id is drawn in yellow on the corresponding supplied frame; bbox coordinates are
[x1,y1,x2,y2], and normalized=true means coordinates are fractions of image size.
Role claims without boxes are independent proposer hypotheses about role existence.

Planning goal:
{goal}

Do not assume the demonstrated goal succeeded. For every claim return yes, no, or uncertain.
Judge every claim separately at every supplied original frame. Do not infer any unshown frame.
The only allowed frame keys are {sorted(selected_frames)}.
Return exactly one JSON object and no Markdown:
{{"verdicts": [{{"claim_id": "...", "frame_verdicts":
{{"0": "no", "3": "uncertain", "6": "yes"}}}}]}}

Claims:
{json_dumps(blind)}"""


def verifier_visual_plan(
    facts: list[dict[str, Any]], frame_count: int, maximum: int
) -> tuple[list[int], dict[int, list[tuple[str, dict[str, Any]]]]]:
    """Balance temporal coverage with frames that identify candidate physical tracks."""
    if frame_count <= 0 or maximum <= 0:
        return [], {}
    base_budget = min(maximum, max(2, maximum // 2))
    selected = set(sample_indices(frame_count, base_budget))
    evidence_by_frame: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for fact in facts:
        if fact.get("kind") != "role":
            continue
        for evidence in fact.get("evidence_boxes", []):
            frame_index = int(evidence.get("frame_index", -1))
            if 0 <= frame_index < frame_count:
                evidence_by_frame.setdefault(frame_index, []).append((fact["id"], evidence))
    ranked = sorted(
        evidence_by_frame,
        key=lambda index: (len({claim for claim, _ in evidence_by_frame[index]}), -index),
        reverse=True,
    )
    for frame_index in ranked:
        if len(selected) >= min(maximum, frame_count):
            break
        selected.add(frame_index)
    indices = sorted(selected)
    return indices, {index: evidence_by_frame.get(index, []) for index in indices}


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


def sanitize_verdicts(
    payload: dict[str, Any], facts: list[dict[str, Any]], sampled_frames: list[int]
) -> list[dict[str, Any]]:
    known = {fact["id"] for fact in facts}
    by_id: dict[str, dict[str, Any]] = {}
    for item in payload.get("verdicts", []):
        if not isinstance(item, dict) or item.get("claim_id") not in known:
            continue
        raw = item.get("frame_verdicts", {})
        raw = raw if isinstance(raw, dict) else {}
        frame_verdicts = {}
        for frame_index in sampled_frames:
            label = normalize_text(raw.get(str(frame_index), raw.get(frame_index, "uncertain")))
            frame_verdicts[str(frame_index)] = label if label in {"yes", "no", "uncertain"} else "uncertain"
        labels = list(frame_verdicts.values())
        aggregate = "yes" if "yes" in labels else ("no" if labels and set(labels) == {"no"} else "uncertain")
        by_id[item["claim_id"]] = {
            "claim_id": item["claim_id"], "label": aggregate,
            "frame_verdicts": frame_verdicts,
        }
    return [by_id.get(fact["id"], {
        "claim_id": fact["id"], "label": "uncertain",
        "frame_verdicts": {str(index): "uncertain" for index in sampled_frames},
    }) for fact in facts]


def reference_from_verdicts(
    facts: list[dict[str, Any]], verdicts: list[dict[str, Any]], frame_count: int,
    sampled_frames: list[int],
) -> dict[str, Any]:
    frames = {index: {"frame_index": index, "nodes": [], "edges": []} for index in sampled_frames}
    fact_by_id = {fact["id"]: fact for fact in facts}
    for verdict in verdicts:
        fact = fact_by_id[verdict["claim_id"]]
        for value, label in verdict.get("frame_verdicts", {}).items():
            index = int(value)
            if label != "yes" or index not in frames:
                continue
            if fact["kind"] == "role":
                frames[index]["nodes"].append(fact["role"])
            else:
                frames[index]["edges"].append([fact["subject"], fact["predicate"], fact["object"]])
    result = []
    for frame in frames.values():
        frame["nodes"] = sorted(set(frame["nodes"]))
        frame["edges"] = [list(value) for value in sorted({tuple(value) for value in frame["edges"]})]
        result.append(frame)
    return {
        "schema_version": "sampled_task_role_graph_v1", "frame_count": frame_count,
        "evaluated_frame_indices": sampled_frames, "frames": result,
    }


def process_episode(
    client: Any, model: str, record: dict[str, Any], study_root: Path, source_root: Path,
    ontology: dict[str, str], max_frames: int, retries: int, overwrite: bool,
) -> str:
    stem = record["video_id"]
    output_path = study_root / "vlm" / f"{stem}.json"
    reference_path = study_root / "vlm_references" / f"{stem}.json"
    if output_path.exists() and reference_path.exists() and not overwrite:
        if load_json(output_path).get("schema_version") == "hybrid_vlm_assessment_v2":
            return f"SKIP {stem}"
    base = load_json(study_root / record["candidate_path"])
    images = sorted((source_root / record["relative_path"] / "images").glob("frame_*.png"))
    if len(images) != int(record["frame_count"]):
        raise ValueError(f"Expected {record['frame_count']} frames for {record['relative_path']}, got {len(images)}")
    indices = sample_indices(len(images), max_frames)
    selected = [images[index] for index in indices]
    proposal_raw = request_object(client, model, [{
        "role": "user",
        "content": visual_content(
            proposer_prompt(record["planning_goal"], len(images), indices), selected, indices
        ),
    }], retries)
    proposal = sanitize_proposal(proposal_raw, len(images), ontology, indices)
    enriched = merge_proposal(base, proposal)
    verifier_indices, overlays = verifier_visual_plan(enriched["facts"], len(images), max_frames)
    verifier_images = [images[index] for index in verifier_indices]
    verifier_raw = request_object(client, model, [{
        "role": "user",
        "content": visual_content(
            verifier_prompt(
                record["planning_goal"], enriched["facts"], len(images), set(verifier_indices)
            ),
            verifier_images, verifier_indices, overlays,
        ),
    }], retries)
    verdicts = sanitize_verdicts(verifier_raw, enriched["facts"], verifier_indices)
    output = {
        "schema_version": "hybrid_vlm_assessment_v2",
        "video_id": stem,
        "model": model,
        "judge_independent_of_qwen_generators": "qwen" not in model.casefold(),
        "sampled_frame_indices": indices,
        "verifier_frame_indices": verifier_indices,
        "proposal": proposal,
        "facts": enriched["facts"],
        "method_facts": enriched["method_facts"],
        "verdicts": verdicts,
    }
    write_json(output_path, output)
    write_json(
        reference_path,
        reference_from_verdicts(enriched["facts"], verdicts, len(images), verifier_indices),
    )
    return f"OK {stem} ({len(enriched['facts'])} facts)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", default=os.environ.get("EVAL_VLM_MODEL") or os.environ.get("SVG2_API_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("EVAL_VLM_BASE_URL") or os.environ.get("SVG2_API_BASE_URL"))
    parser.add_argument("--api-key-env", default="EVAL_VLM_API_KEY")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--split", choices=("all", "human", "vlm_only"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        raise SystemExit("Set EVAL_VLM_BASE_URL (and EVAL_VLM_MODEL, or use --model auto)")
    key = os.environ.get(args.api_key_env)
    if not key and args.api_key_env == "EVAL_VLM_API_KEY":
        key = os.environ.get("SVG2_API_KEY")
    if not key and is_local_endpoint(args.base_url):
        key = "local"
    if not key:
        raise SystemExit(f"Set {args.api_key_env} (or SVG2_API_KEY for the default fallback)")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=180.0, max_retries=0)
    model = args.model
    if model == "auto":
        available = [entry.id for entry in client.models.list().data]
        if not available:
            raise SystemExit(f"No models reported by {args.base_url}")
        model = available[0]
        print(f"Using model discovered from local server: {model}", flush=True)
    if not model:
        raise SystemExit("Set EVAL_VLM_MODEL or pass --model auto")
    study_root = Path(args.study_root).resolve()
    study = load_json(study_root / "study_manifest.json")
    records = study["episodes"]
    if args.split == "human":
        records = [record for record in records if record["evaluation_split"].startswith("human_")]
    elif args.split == "vlm_only":
        records = [record for record in records if record["evaluation_split"] == "vlm_only"]
    if args.limit is not None:
        records = records[:args.limit]
    ontology = load_ontology(study["ontology"])
    source_root = Path(args.source_root).resolve()

    def run(record: dict[str, Any]) -> str:
        return process_episode(
            client, model, record, study_root, source_root, ontology,
            args.max_frames, args.retries, args.overwrite,
        )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, result in enumerate(pool.map(run, records), 1):
                print(f"[{index:04d}/{len(records):04d}] {result}", flush=True)
    else:
        for index, record in enumerate(records, 1):
            print(f"[{index:04d}/{len(records):04d}] {run(record)}", flush=True)


if __name__ == "__main__":
    main()
