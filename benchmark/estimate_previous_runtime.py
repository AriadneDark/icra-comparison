#!/usr/bin/env python3
"""Estimate an earlier benchmark runtime from artifact modification times.

This is necessarily an approximation: model loading happens before the first
artifact is written, and filesystem mtimes cannot distinguish computation from
an intentional pause. The script reports both the full calendar span and
separate apparent sessions split at large gaps.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SG_STAGES = {
    "caption": "sg_ego/captions/goal_roles/{stem}.json",
    "grounding": "sg_ego/frame_graphs/goal_roles/{stem}.json",
    "consolidation": "sg_ego/video_graphs/goal_roles/{stem}.json",
}

SVG2_STAGES = {
    "stage1_masks": "svg2/{stem}/stage1_masks.json",
    "stage2_tracks": "svg2/{stem}/stage2_tracks.json",
    "stage3_cleanup": "svg2/{stem}/stage3_tracks_clean.json",
    "stage4_caption": "svg2/{stem}/stage4_descriptions.json",
    "stage5_structure": "svg2/{stem}/stage5_scene_graph.json",
    "stage6_relationships": "svg2/{stem}/stage6_scene_graph.json",
}


def stamp(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def duration(value: float) -> str:
    seconds = max(0, round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def split_sessions(values: list[float], gap_seconds: float) -> list[list[float]]:
    if not values:
        return []
    sessions = [[values[0]]]
    for value in values[1:]:
        if value - sessions[-1][-1] > gap_seconds:
            sessions.append([])
        sessions[-1].append(value)
    return sessions


def collect(
    work_root: Path, stems: list[str], stages: dict[str, str], gap_seconds: float
) -> dict[str, Any]:
    stage_summary: dict[str, Any] = {}
    all_files: list[Path] = []
    for stage, template in stages.items():
        expected = [work_root / template.format(stem=stem) for stem in stems]
        existing = [path for path in expected if path.is_file()]
        all_files.extend(existing)
        times = sorted(path.stat().st_mtime for path in existing)
        stage_summary[stage] = {
            "complete": len(existing),
            "expected": len(expected),
            "first_write": stamp(times[0]) if times else None,
            "last_write": stamp(times[-1]) if times else None,
            "write_span_seconds": times[-1] - times[0] if len(times) > 1 else 0.0,
        }

    timeline = sorted((path.stat().st_mtime, path) for path in set(all_files))
    sessions = split_sessions([value for value, _ in timeline], gap_seconds)
    final_stage = next(reversed(stages))
    result: dict[str, Any] = {
        "artifact_count": len(timeline),
        "final_outputs": stage_summary[final_stage]["complete"],
        "expected_videos": len(stems),
        "first_artifact": stamp(timeline[0][0]) if timeline else None,
        "last_artifact": stamp(timeline[-1][0]) if timeline else None,
        "calendar_span_seconds": timeline[-1][0] - timeline[0][0] if len(timeline) > 1 else 0.0,
        "sessions": [
            {
                "first_artifact": stamp(values[0]),
                "last_artifact": stamp(values[-1]),
                "span_seconds": values[-1] - values[0] if len(values) > 1 else 0.0,
                "artifact_count": len(values),
            }
            for values in sessions
        ],
        "stages": stage_summary,
    }
    result["session_spans_seconds"] = sum(item["span_seconds"] for item in result["sessions"])
    return result


def print_method(name: str, value: dict[str, Any]) -> None:
    print(f"\n{name}")
    print(f"  final outputs: {value['final_outputs']}/{value['expected_videos']}")
    print(f"  artifacts:     {value['artifact_count']}")
    if not value["artifact_count"]:
        print("  No matching artifacts found.")
        return
    print(f"  first write:   {value['first_artifact']}")
    print(f"  last write:    {value['last_artifact']}")
    print(f"  calendar span: {duration(value['calendar_span_seconds'])}")
    if len(value["sessions"]) > 1:
        print(
            f"  session spans: {duration(value['session_spans_seconds'])} "
            f"across {len(value['sessions'])} apparent sessions"
        )
        for index, session in enumerate(value["sessions"], 1):
            print(
                f"    {index}: {session['first_artifact']} -> {session['last_artifact']} "
                f"({duration(session['span_seconds'])}, {session['artifact_count']} artifacts)"
            )
    print("  stages:")
    for stage, item in value["stages"].items():
        span = duration(item["write_span_seconds"])
        print(f"    {stage:<22} {item['complete']:>4}/{item['expected']:<4} write span {span}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work-root", default="baseline_runs")
    parser.add_argument(
        "--session-gap-hours", type=float, default=2.0,
        help="Start a new apparent run session when adjacent artifact writes are farther apart.",
    )
    parser.add_argument("--json-output", help="Optionally save the machine-readable report.")
    args = parser.parse_args()
    if args.session_gap_hours <= 0:
        raise SystemExit("--session-gap-hours must be positive")

    manifest_path = Path(args.manifest).resolve()
    work_root = Path(args.work_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise SystemExit("Manifest has no episodes")
    stems = [str(item["relative_path"]).replace("/", "__") for item in episodes]
    if len(stems) != len(set(stems)):
        raise SystemExit("Manifest contains duplicate relative_path values")

    gap_seconds = args.session_gap_hours * 3600
    report = {
        "schema_version": "artifact_runtime_estimate_v1",
        "manifest": str(manifest_path),
        "work_root": str(work_root),
        "videos": len(stems),
        "session_gap_hours": args.session_gap_hours,
        "methods": {
            "sg_ego": collect(work_root, stems, SG_STAGES, gap_seconds),
            "svg2": collect(work_root, stems, SVG2_STAGES, gap_seconds),
        },
        "limitations": [
            "Times are inferred from artifact mtimes, not measured process runtime.",
            "Model loading before the first write is absent, so uninterrupted runs are underestimated.",
            "Calendar span includes pauses; session spans exclude gaps above the configured threshold.",
        ],
    }

    dated_methods = [
        value for value in report["methods"].values()
        if value["first_artifact"] is not None and value["last_artifact"] is not None
    ]
    if dated_methods:
        overall_start = min(
            datetime.fromisoformat(value["first_artifact"]).timestamp() for value in dated_methods
        )
        overall_end = max(
            datetime.fromisoformat(value["last_artifact"]).timestamp() for value in dated_methods
        )
        report["overall_calendar_span"] = {
            "first_artifact": stamp(overall_start),
            "last_artifact": stamp(overall_end),
            "span_seconds": overall_end - overall_start,
        }
    else:
        report["overall_calendar_span"] = None

    print(f"Manifest: {manifest_path} ({len(stems)} videos)")
    print(f"Work root: {work_root}")
    print(f"Session gap: {args.session_gap_hours:g} hours")
    print_method("SG-Ego", report["methods"]["sg_ego"])
    print_method("SVG2", report["methods"]["svg2"])
    if report["overall_calendar_span"]:
        overall = report["overall_calendar_span"]
        print("\nOverall SG-Ego + SVG2 calendar span")
        print(f"  first write: {overall['first_artifact']}")
        print(f"  last write:  {overall['last_artifact']}")
        print(f"  span:        {duration(overall['span_seconds'])}")
    print("\nThis is an artifact-time estimate, not an exact process timer.")

    if args.json_output:
        output = Path(args.json_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
