#!/usr/bin/env python3
"""Report human-gold and human-calibrated VLM metrics for three graph methods."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .hybrid_common import METHODS, ROLE_ORDER, canonical_predicate, load_json, load_ontology, write_json
except ImportError:
    from hybrid_common import METHODS, ROLE_ORDER, canonical_predicate, load_json, load_ontology, write_json


def scores(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def add_counts(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def interval_iou(first: list[list[int]], second: list[list[int]]) -> float | None:
    a = {frame for start, end in first for frame in range(min(start, end), max(start, end) + 1)}
    b = {frame for start, end in second for frame in range(min(start, end), max(start, end) + 1)}
    if not a and not b:
        return None
    return len(a & b) / len(a | b) if a | b else None


def human_episode_counts(
    facts: list[dict[str, Any]], annotation: dict[str, Any], ontology: dict[str, str]
) -> dict[str, dict[str, tuple[float, float, float] | list[float]]]:
    labels = annotation.get("claim_labels", {})
    result: dict[str, dict[str, Any]] = {
        method: {"roles": (0.0, 0.0, 0.0), "relations": (0.0, 0.0, 0.0), "temporal_iou": []}
        for method in METHODS
    }
    role_status = {
        role: annotation.get("roles", {}).get(role, {}).get("status", "ambiguous")
        for role in ROLE_ORDER
    }
    for method in METHODS:
        for role in ROLE_ORDER:
            if role_status[role] not in {"present", "not_applicable"}:
                continue
            predictions = [
                fact for fact in facts
                if fact["kind"] == "role" and fact["role"] == role and method in fact.get("sources", [])
            ]
            accepted = any(labels.get(fact["id"], {}).get("label") == "yes" for fact in predictions)
            expected = role_status[role] == "present"
            if accepted and expected:
                value = (1.0, 0.0, 0.0)
            elif predictions and expected:
                value = (0.0, 1.0, 1.0)
            elif not predictions and expected:
                value = (0.0, 0.0, 1.0)
            elif predictions and not expected:
                value = (0.0, 1.0, 0.0)
            else:
                value = (0.0, 0.0, 0.0)
            result[method]["roles"] = add_counts(result[method]["roles"], value)

        relation_facts = [fact for fact in facts if fact["kind"] == "relation"]
        predicted = {fact["id"] for fact in relation_facts if method in fact.get("sources", [])}
        true_ids = {fact["id"] for fact in relation_facts if labels.get(fact["id"], {}).get("label") == "yes"}
        false_ids = {fact["id"] for fact in relation_facts if labels.get(fact["id"], {}).get("label") == "no"}
        missing = [
            item for item in annotation.get("missing_relations", [])
            if item.get("subject") in ROLE_ORDER and item.get("object") in ROLE_ORDER
            and canonical_predicate(item.get("predicate", ""), ontology)
        ]
        relation_counts = (
            float(len(predicted & true_ids)),
            float(len(predicted & false_ids)),
            float(len(true_ids - predicted) + len(missing)),
        )
        result[method]["relations"] = relation_counts
        for fact in relation_facts:
            if fact["id"] not in predicted & true_ids:
                continue
            source_intervals = fact.get("source_intervals", {}).get(method, [])
            reference_intervals = labels[fact["id"]].get("intervals", [])
            value = interval_iou(source_intervals, reference_intervals)
            if value is not None:
                result[method]["temporal_iou"].append(value)
    return result


def calibration_table(
    records: list[dict[str, Any]], study_root: Path, annotation_root: Path
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record["evaluation_split"] != "human_primary":
            continue
        annotation_path = annotation_root / f"{record['video_id']}.json"
        vlm_path = study_root / "vlm" / f"{record['video_id']}.json"
        if not annotation_path.exists() or not vlm_path.exists():
            continue
        annotation, vlm = load_json(annotation_path), load_json(vlm_path)
        if not annotation.get("complete"):
            continue
        fact_kind = {fact["id"]: fact["kind"] for fact in vlm["facts"]}
        human = annotation.get("claim_labels", {})
        for verdict in vlm.get("verdicts", []):
            human_label = human.get(verdict["claim_id"], {}).get("label")
            if human_label not in {"yes", "no"}:
                continue
            kind = fact_kind.get(verdict["claim_id"])
            vlm_label = verdict.get("label", "uncertain")
            if kind not in {"role", "relation"} or vlm_label not in {"yes", "no", "uncertain"}:
                continue
            counts[(kind, vlm_label)][0 if human_label == "yes" else 1] += 1
            confusion[kind][f"human_{human_label}__vlm_{vlm_label}"] += 1

    priors = {"yes": (1.6, 0.4), "no": (0.4, 1.6), "uncertain": (1.0, 1.0)}
    table: dict[str, dict[str, dict[str, float]]] = {"role": {}, "relation": {}}
    for kind in table:
        for label in ("yes", "no", "uncertain"):
            positive, negative = counts[(kind, label)]
            alpha, beta = priors[label]
            table[kind][label] = {
                "p_true": (positive + alpha) / (positive + negative + alpha + beta),
                "human_true": positive,
                "human_false": negative,
            }
    return table, {kind: dict(values) for kind, values in confusion.items()}


def vlm_episode_counts(
    vlm: dict[str, Any], calibration: dict[str, dict[str, dict[str, float]]],
    relation_pool_coverage: float, role_pool_coverage: float,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    facts = vlm["facts"]
    verdict_by_id = {item["claim_id"]: item for item in vlm.get("verdicts", [])}
    probability = {}
    for fact in facts:
        label = verdict_by_id.get(fact["id"], {}).get("label", "uncertain")
        probability[fact["id"]] = calibration[fact["kind"]][label]["p_true"]
    result = {method: {"roles": (0.0, 0.0, 0.0), "relations": (0.0, 0.0, 0.0)} for method in METHODS}
    role_facts = [fact for fact in facts if fact["kind"] == "role"]
    relation_facts = [fact for fact in facts if fact["kind"] == "relation"]
    for method in METHODS:
        role_counts = (0.0, 0.0, 0.0)
        for role in ROLE_ORDER:
            universe = [fact for fact in role_facts if fact["role"] == role]
            predictions = [fact for fact in universe if method in fact.get("sources", [])]
            truth = max((probability[fact["id"]] for fact in universe), default=0.0)
            correct = max((probability[fact["id"]] for fact in predictions), default=0.0)
            role_counts = add_counts(role_counts, (
                correct,
                max(0.0, float(bool(predictions)) - correct),
                max(0.0, truth - correct)
                + (truth * (1.0 / role_pool_coverage - 1.0) if 0 < role_pool_coverage < 1 else 0.0),
            ))
        result[method]["roles"] = role_counts

        predicted = [fact for fact in relation_facts if method in fact.get("sources", [])]
        predicted_ids = {fact["id"] for fact in predicted}
        tp = sum(probability[fact["id"]] for fact in predicted)
        fp = sum(1.0 - probability[fact["id"]] for fact in predicted)
        fn = sum(probability[fact["id"]] for fact in relation_facts if fact["id"] not in predicted_ids)
        observed_truth = sum(probability[fact["id"]] for fact in relation_facts)
        if 0 < relation_pool_coverage < 1:
            fn += observed_truth * (1.0 / relation_pool_coverage - 1.0)
        result[method]["relations"] = (tp, fp, fn)
    return result


def bootstrap_ci(per_video: list[tuple[float, float, float]], seed: int, samples: int = 2000) -> dict[str, list[float]]:
    if not per_video:
        return {"precision": [0.0, 0.0], "recall": [0.0, 0.0], "f1": [0.0, 0.0]}
    rng = random.Random(seed)
    values = {key: [] for key in ("precision", "recall", "f1")}
    for _ in range(samples):
        picked = [rng.choice(per_video) for _ in per_video]
        aggregate = (sum(x[0] for x in picked), sum(x[1] for x in picked), sum(x[2] for x in picked))
        result = scores(*aggregate)
        for key in values:
            values[key].append(result[key])
    return {
        key: [sorted(data)[int(0.025 * samples)], sorted(data)[int(0.975 * samples) - 1]]
        for key, data in values.items()
    }


def agreement(first: Path, second: Path, records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    pairs = []
    for record in records:
        a, b = first / f"{record['video_id']}.json", second / f"{record['video_id']}.json"
        if not a.exists() or not b.exists():
            continue
        la, lb = load_json(a).get("claim_labels", {}), load_json(b).get("claim_labels", {})
        for identifier in la.keys() & lb.keys():
            if la[identifier].get("label") in {"yes", "no"} and lb[identifier].get("label") in {"yes", "no"}:
                pairs.append((la[identifier]["label"], lb[identifier]["label"]))
    if not pairs:
        return {"items": 0, "agreement": None, "cohen_kappa": None}
    observed = sum(a == b for a, b in pairs) / len(pairs)
    pa = sum(a == "yes" for a, _ in pairs) / len(pairs)
    pb = sum(b == "yes" for _, b in pairs) / len(pairs)
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"items": len(pairs), "agreement": observed, "cohen_kappa": kappa}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--second-annotator")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    study_root = Path(args.study_root).resolve()
    study = load_json(study_root / "study_manifest.json")
    ontology = load_ontology(study["ontology"])
    annotation_root = study_root / "human_annotations" / args.annotator
    records = study["episodes"]
    calibration, confusion = calibration_table(records, study_root, annotation_root)

    human_per_video: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        split: {f"{method}_{kind}": [] for method in METHODS for kind in ("roles", "relations")}
        for split in ("human_primary", "human_challenge", "human_all")
    }
    temporal: dict[str, dict[str, list[float]]] = {
        split: {method: [] for method in METHODS}
        for split in ("human_primary", "human_challenge", "human_all")
    }
    complete_human = 0
    accepted_relations = missing_relations = 0
    present_roles = covered_roles = 0
    for record in records:
        if not record["evaluation_split"].startswith("human_"):
            continue
        annotation_path = annotation_root / f"{record['video_id']}.json"
        vlm_path = study_root / "vlm" / f"{record['video_id']}.json"
        if not annotation_path.exists() or not vlm_path.exists():
            continue
        annotation, vlm = load_json(annotation_path), load_json(vlm_path)
        if not annotation.get("complete"):
            continue
        complete_human += 1
        counts = human_episode_counts(vlm["facts"], annotation, ontology)
        # Estimate pool coverage only on the probability-sampled primary split.
        # The disagreement-selected challenge split is intentionally non-representative.
        if record["evaluation_split"] == "human_primary":
            accepted_relations += sum(
                label.get("label") == "yes"
                for identifier, label in annotation.get("claim_labels", {}).items()
                if any(
                    fact["id"] == identifier and fact["kind"] == "relation"
                    for fact in vlm["facts"]
                )
            )
            missing_relations += len(annotation.get("missing_relations", []))
            for role in ROLE_ORDER:
                if annotation.get("roles", {}).get(role, {}).get("status") != "present":
                    continue
                present_roles += 1
                covered_roles += any(
                    fact["kind"] == "role" and fact["role"] == role
                    and annotation.get("claim_labels", {}).get(fact["id"], {}).get("label") == "yes"
                    for fact in vlm["facts"]
                )
        for method in METHODS:
            for kind in ("roles", "relations"):
                value = counts[method][kind]
                human_per_video[record["evaluation_split"]][f"{method}_{kind}"].append(value)
                human_per_video["human_all"][f"{method}_{kind}"].append(value)
            temporal[record["evaluation_split"]][method].extend(counts[method]["temporal_iou"])
            temporal["human_all"][method].extend(counts[method]["temporal_iou"])

    relation_pool_coverage = (
        accepted_relations / (accepted_relations + missing_relations)
        if accepted_relations + missing_relations else 1.0
    )
    role_pool_coverage = covered_roles / present_roles if present_roles else 1.0
    human_summary: dict[str, Any] = {}
    for split, values in human_per_video.items():
        human_summary[split] = {}
        for method in METHODS:
            human_summary[split][method] = {}
            for kind in ("roles", "relations"):
                items = values[f"{method}_{kind}"]
                total = tuple(sum(value[index] for value in items) for index in range(3))
                human_summary[split][method][kind] = {
                    **scores(*total), "videos": len(items),
                    "bootstrap_95_ci": bootstrap_ci(items, args.seed),
                }
            human_summary[split][method]["mean_temporal_iou"] = (
                sum(temporal[split][method]) / len(temporal[split][method])
                if temporal[split][method] else None
            )

    vlm_totals = {method: {kind: (0.0, 0.0, 0.0) for kind in ("roles", "relations")} for method in METHODS}
    vlm_videos = 0
    for record in records:
        if record["evaluation_split"] != "vlm_only":
            continue
        path = study_root / "vlm" / f"{record['video_id']}.json"
        if not path.exists():
            continue
        counts = vlm_episode_counts(
            load_json(path), calibration, relation_pool_coverage, role_pool_coverage
        )
        vlm_videos += 1
        for method in METHODS:
            for kind in ("roles", "relations"):
                vlm_totals[method][kind] = add_counts(vlm_totals[method][kind], counts[method][kind])
    vlm_summary = {
        method: {kind: scores(*vlm_totals[method][kind]) for kind in ("roles", "relations")}
        for method in METHODS
    }
    for method in METHODS:
        vlm_summary[method]["videos"] = vlm_videos

    full_summary: dict[str, Any] = {}
    for method in METHODS:
        full_summary[method] = {}
        for kind in ("roles", "relations"):
            human_items = human_per_video["human_all"][f"{method}_{kind}"]
            human_total = tuple(sum(value[index] for value in human_items) for index in range(3))
            combined = add_counts(human_total, vlm_totals[method][kind])
            full_summary[method][kind] = scores(*combined)
        full_summary[method]["videos"] = complete_human + vlm_videos

    second_root = study_root / "human_annotations" / args.second_annotator if args.second_annotator else None
    expected_human = sum(record["evaluation_split"].startswith("human_") for record in records)
    expected_vlm_only = sum(record["evaluation_split"] == "vlm_only" for record in records)
    warnings = [
        "VLM-calibrated metrics are model-based estimates, not direct ground-truth precision/recall.",
        "Human-primary is the unbiased headline subset; human-challenge must be reported separately.",
    ]
    if complete_human < expected_human:
        warnings.append(
            f"Only {complete_human}/{expected_human} human tasks are complete; incomplete human videos are omitted."
        )
    if vlm_videos < expected_vlm_only:
        warnings.append(
            f"Only {vlm_videos}/{expected_vlm_only} VLM-only tasks are complete; missing videos are omitted."
        )
    if relation_pool_coverage < 0.95 or role_pool_coverage < 0.95:
        warnings.append(
            "Candidate-pool coverage below 0.95 makes calibrated recall sensitive to the missing-fact correction."
        )
    report = {
        "schema_version": "hybrid_evaluation_report_v1",
        "human_complete_videos": complete_human,
        "vlm_only_complete_videos": vlm_videos,
        "candidate_pool_coverage_on_human": {
            "roles": role_pool_coverage,
            "relations": relation_pool_coverage,
        },
        "calibration": calibration,
        "vlm_human_confusion": confusion,
        "human_metrics": human_summary,
        "vlm_calibrated_metrics": vlm_summary,
        "full_hybrid_metrics": full_summary,
        "inter_annotator": agreement(annotation_root, second_root, records) if second_root else None,
        "warnings": warnings,
    }
    output = Path(args.output)
    write_json(output, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "split", "method", "kind", "precision", "recall", "f1", "videos"])
        for split, methods in human_summary.items():
            for method, kinds in methods.items():
                for kind in ("roles", "relations"):
                    item = kinds[kind]
                    writer.writerow(["human", split, method, kind, item["precision"], item["recall"], item["f1"], item["videos"]])
        for method, kinds in vlm_summary.items():
            for kind in ("roles", "relations"):
                item = kinds[kind]
                writer.writerow(["vlm_calibrated", "vlm_only", method, kind, item["precision"], item["recall"], item["f1"], kinds["videos"]])
        for method, kinds in full_summary.items():
            for kind in ("roles", "relations"):
                item = kinds[kind]
                writer.writerow(["hybrid", "full", method, kind, item["precision"], item["recall"], item["f1"], kinds["videos"]])
    print(f"Wrote {output} and {output.with_suffix('.csv')}")
    print(
        f"Human complete: {complete_human}; VLM-only complete: {vlm_videos}; "
        f"pool coverage roles={role_pool_coverage:.3f}, relations={relation_pool_coverage:.3f}"
    )


if __name__ == "__main__":
    main()
