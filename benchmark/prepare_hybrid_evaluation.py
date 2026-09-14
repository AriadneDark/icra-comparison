#!/usr/bin/env python3
"""Build a blind candidate pool and select human-gold videos for hybrid evaluation."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .hybrid_common import (
        METHODS, build_candidates, disagreement_score, episode_stem, load_json,
        load_method_output, load_ontology, task_family, write_json,
    )
except ImportError:
    from hybrid_common import (
        METHODS, build_candidates, disagreement_score, episode_stem, load_json,
        load_method_output, load_ontology, task_family, write_json,
    )


def proportional_stratified_sample(
    records: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample proportionally while keeping every stratum identifiable for PPI."""
    if size >= len(records):
        return list(records)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[(record["dataset_name"], record["task_family"])].append(record)
    for values in strata.values():
        rng.shuffle(values)

    exact = {key: size * len(values) / len(records) for key, values in strata.items()}
    # Two observations permit within-stratum uncertainty estimation. Singleton
    # strata are censused and therefore have no sampling uncertainty.
    minimum = {key: min(2, len(values)) for key, values in strata.items()}
    if sum(minimum.values()) > size:
        raise ValueError(
            f"human-primary size {size} is too small for PPI coverage of {len(strata)} strata; "
            f"need at least {sum(minimum.values())}"
        )
    allocation = dict(minimum)
    remaining = size - sum(allocation.values())
    while remaining:
        available = [key for key in strata if allocation[key] < len(strata[key])]
        if not available:
            break
        key = max(
            available,
            key=lambda value: (exact[value] - allocation[value], len(strata[value]), value),
        )
        allocation[key] += 1
        remaining -= 1
    return [record for key in sorted(strata) for record in strata[key][:allocation[key]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ours-root", required=True)
    parser.add_argument("--sg-ego-root", required=True)
    parser.add_argument("--svg2-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ontology", default=str(Path(__file__).with_name("predicate_ontology.json")))
    parser.add_argument("--human-size", type=int, default=100)
    parser.add_argument("--challenge-size", type=int, default=20)
    parser.add_argument("--double-annotation-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise SystemExit("Manifest has no episodes")
    if not 0 <= args.challenge_size <= args.human_size <= len(episodes):
        raise SystemExit("Require 0 <= challenge-size <= human-size <= number of videos")
    if not 0 <= args.double_annotation_size <= args.human_size:
        raise SystemExit("Require 0 <= double-annotation-size <= human-size")

    roots = {
        "ours": Path(args.ours_root).resolve(),
        "sg_ego": Path(args.sg_ego_root).resolve(),
        "svg2": Path(args.svg2_root).resolve(),
    }
    ontology = load_ontology(args.ontology)
    output_root = Path(args.output_root).resolve()
    candidate_root = output_root / "candidates"
    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for index, episode in enumerate(episodes, 1):
        relative_path = episode["relative_path"]
        stem = episode_stem(relative_path)
        frame_count = int(episode["frame_count"])
        method_outputs = {
            method: load_method_output(method, roots[method], relative_path, frame_count, ontology)
            for method in METHODS
        }
        absent = [method for method, output in method_outputs.items() if output is None]
        for method in absent:
            missing.append({"episode": relative_path, "method": method})
        if absent and not args.allow_missing:
            continue
        candidates = build_candidates(method_outputs)
        candidate_payload = {
            "schema_version": "hybrid_candidate_pool_v1",
            "video_id": stem,
            "relative_path": relative_path,
            "planning_goal": episode["planning_goal"],
            "frame_count": frame_count,
            "method_outputs": method_outputs,
            **candidates,
        }
        candidate_path = candidate_root / f"{stem}.json"
        write_json(candidate_path, candidate_payload)
        record = {
            **episode,
            "video_id": stem,
            "task_family": task_family(episode["planning_goal"]),
            "candidate_path": str(candidate_path.relative_to(output_root)),
            "disagreement_score": round(disagreement_score(candidates["method_facts"]), 6),
            "missing_methods": absent,
        }
        records.append(record)
        print(f"[{index:04d}/{len(episodes):04d}] {relative_path}: disagreement={record['disagreement_score']:.3f}")

    if len(records) < args.human_size:
        raise SystemExit(
            f"Only {len(records)} complete candidate records, fewer than --human-size={args.human_size}; "
            "use --allow-missing only if incomplete methods are intentional"
        )

    rng = random.Random(args.seed)
    primary_size = args.human_size - args.challenge_size
    primary = proportional_stratified_sample(records, primary_size, rng)
    primary_ids = {record["video_id"] for record in primary}
    remaining = [record for record in records if record["video_id"] not in primary_ids]
    challenge = sorted(
        remaining, key=lambda record: (record["disagreement_score"], record["video_id"]), reverse=True
    )[:args.challenge_size]
    challenge_ids = {record["video_id"] for record in challenge}
    double_ids = {
        record["video_id"] for record in rng.sample(primary + challenge, args.double_annotation_size)
    }
    stratum_sizes = Counter((record["dataset_name"], record["task_family"]) for record in records)
    primary_stratum_sizes = Counter(
        (record["dataset_name"], record["task_family"]) for record in primary
    )
    for record in records:
        stratum = (record["dataset_name"], record["task_family"])
        record["primary_inclusion_probability"] = (
            primary_stratum_sizes[stratum] / stratum_sizes[stratum]
        )
        if record["video_id"] in primary_ids:
            record["evaluation_split"] = "human_primary"
        elif record["video_id"] in challenge_ids:
            record["evaluation_split"] = "human_challenge"
        else:
            record["evaluation_split"] = "vlm_only"
        record["double_annotation"] = record["video_id"] in double_ids

    study = {
        "schema_version": "hybrid_evaluation_study_v1",
        "source_manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "selection": {
            "total": len(records),
            "human_total": args.human_size,
            "human_primary": primary_size,
            "human_challenge": args.challenge_size,
            "double_annotation": args.double_annotation_size,
            "strategy": "proportional dataset/task stratification plus high-disagreement challenge set",
            "ppi_design": "stratified probability sample with at least two primary videos per non-singleton stratum",
        },
        "ontology": str(Path(args.ontology).resolve()),
        "roots": {method: str(root) for method, root in roots.items()},
        "missing": missing,
        "episodes": records,
    }
    write_json(output_root / "study_manifest.json", study)
    write_json(output_root / "human_100_manifest.json", {
        "schema_version": "benchmark_manifest_v1",
        "source_manifest": str(Path(args.manifest).resolve()),
        "selection_seed": args.seed,
        "episodes": [
            record for record in records
            if record["evaluation_split"].startswith("human_")
        ],
    })
    (output_root / "human_annotations").mkdir(parents=True, exist_ok=True)
    (output_root / "vlm").mkdir(parents=True, exist_ok=True)
    print(
        f"Prepared {len(records)} videos: {primary_size} primary human, "
        f"{len(challenge)} challenge, {len(records) - args.human_size} VLM-only"
    )
    print(f"Study manifest: {output_root / 'study_manifest.json'}")
    print(f"Human manifest: {output_root / 'human_100_manifest.json'}")


if __name__ == "__main__":
    main()
