#!/usr/bin/env python3
"""Materialize the human-100 SAM ablations while reusing frozen Qwen artifacts."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any


VARIANTS = ("qwen_text_sam3", "qwen_box_sam3", "qwen_box_sam2")
UPSTREAM_FILES = ("input.json", "task_spec.json", "qwen_entities_audit.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_scene(root: Path, relative_path: str) -> Path:
    candidates = (root / "scenes" / relative_path, root / relative_path)
    return next((path for path in candidates if (path / "task_spec.json").exists()), candidates[0])


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def materialize_scene(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in UPSTREAM_FILES:
        path = source / name
        if path.exists():
            link_or_copy(path, target / name)
    if not (target / "input.json").exists() or not (target / "task_spec.json").exists():
        raise FileNotFoundError(f"Missing frozen prepare/Qwen artifacts in {source}")
    frames = sorted((source / "frames").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No prepared frames in {source / 'frames'}")
    for frame in frames:
        link_or_copy(frame, target / "frames" / frame.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--full-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    manifest = load_json(args.study_root / "human_100_manifest.json")
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise SystemExit("human_100_manifest.json has no episodes")
    compact_manifest = {
        "schema_version": "benchmark_manifest_v1",
        "source_manifest": str(args.study_root / "human_100_manifest.json"),
        "episodes": [
            {
                "relative_path": record["relative_path"],
                "planning_goal": record["planning_goal"],
                "frame_count": int(record["frame_count"]),
            }
            for record in episodes
        ],
    }
    write_json(args.run_root / "human_100_manifest.json", compact_manifest)

    for index, record in enumerate(episodes, 1):
        source = source_scene(args.full_root, record["relative_path"])
        for variant in args.variants:
            materialize_scene(source, args.run_root / variant / record["relative_path"])
        print(f"[{index:03d}/{len(episodes):03d}] {record['relative_path']}")
    write_json(args.run_root / "ablation_manifest.json", {
        "schema_version": "object_ablation_manifest_v1",
        "full_root": str(args.full_root.resolve()),
        "human_manifest": str((args.run_root / "human_100_manifest.json").resolve()),
        "variants": list(args.variants),
        "episode_count": len(episodes),
        "qwen_policy": "reuse frozen task_spec.json and qwen_entities_audit.json from Full",
    })
    print(f"Prepared {len(episodes)} scenes for {len(args.variants)} variants in {args.run_root}")


if __name__ == "__main__":
    main()
