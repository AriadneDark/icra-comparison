#!/usr/bin/env python3
"""Summarize the amount of human work in a frozen evaluation study."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0


def summarize(root: Path) -> dict[str, object]:
    study = json.loads((root / "study_manifest.json").read_text(encoding="utf-8"))
    rows: list[tuple[str, int, int]] = []
    predicates: Counter[str] = Counter()
    for episode in study["episodes"]:
        payload = json.loads((root / episode["candidate_path"]).read_text(encoding="utf-8"))
        role_count = sum(fact["kind"] == "role" for fact in payload["facts"])
        relations = [fact for fact in payload["facts"] if fact["kind"] == "relation"]
        predicates.update(fact["predicate"] for fact in relations)
        rows.append((episode["evaluation_split"], role_count, len(relations)))

    def group(name: str, selected: list[tuple[str, int, int]]) -> dict[str, object]:
        totals = [roles + relations for _, roles, relations in selected]
        return {
            "videos": len(selected),
            "roles": sum(row[1] for row in selected),
            "relations": sum(row[2] for row in selected),
            "facts": sum(totals),
            "mean_facts_per_video": round(statistics.mean(totals), 2) if totals else 0,
            "median_facts_per_video": statistics.median(totals) if totals else 0,
            "p90_facts_per_video": percentile(totals, 0.9),
            "max_facts_per_video": max(totals, default=0),
        }

    human = [row for row in rows if row[0] in {"human_primary", "human_challenge"}]
    return {
        "study_root": str(root.resolve()),
        "all": group("all", rows),
        "human_100": group("human_100", human),
        "predicate_types": len(predicates),
        "top_predicates": predicates.most_common(30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_roots", nargs="+")
    args = parser.parse_args()
    for value in args.study_roots:
        print(json.dumps(summarize(Path(value)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
