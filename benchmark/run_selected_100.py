#!/usr/bin/env python3
"""Run adapted SG-Ego/SVG2 over the fixed 100-episode manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def make_video(images: Path, output: Path, fps: float, dry_run: bool) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(images / "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ], Path.cwd(), dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True,
                        help="Contains <relative_path>/images and planning_goal.json")
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--method", choices=("prepare", "sg_ego", "svg2", "both"), default="both",
                        help="Use prepare to materialize MP4 inputs without loading any models.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episode", help="Run one manifest relative_path (smoke test).")
    parser.add_argument("--limit", type=int, help="Run only the first N manifest episodes.")
    parser.add_argument("--expected-count", type=int,
                        help="Optional manifest-size guard (for example 100 or 1000).")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sg_repo, svg_repo = repo / "baselines" / "sg-ego", repo / "baselines" / "svg2"
    source_root, work_root = Path(args.source_root).resolve(), Path(args.work_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise SystemExit("Manifest has no episodes")
    if args.expected_count is not None and len(episodes) != args.expected_count:
        raise SystemExit(f"Expected exactly {args.expected_count} episodes, got {len(episodes)}")
    if args.episode:
        episodes = [ep for ep in episodes if ep["relative_path"] == args.episode]
        if not episodes:
            raise SystemExit(f"Episode not found in manifest: {args.episode}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        episodes = episodes[:args.limit]

    missing = []
    for ep in episodes:
        scene = source_root / ep["relative_path"]
        if not (scene / "images").is_dir():
            missing.append(str(scene / "images"))
    if missing:
        raise SystemExit(f"Missing input image directories ({len(missing)}), first: {missing[0]}")

    sg_data = work_root / "sg_ego"
    # SG-Ego's grounding entry point assumes its versioned output directory
    # already exists. Create the complete layout before any stage starts.
    for directory in (
        sg_data / "videos",
        sg_data / "captions" / "goal_roles",
        sg_data / "frame_graphs" / "goal_roles",
        sg_data / "video_graphs" / "goal_roles",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    total = len(episodes)
    for number, ep in enumerate(episodes, 1):
        stem = ep["relative_path"].replace("/", "__")
        scene = source_root / ep["relative_path"]
        video = work_root / "videos" / f"{stem}.mp4"
        make_video(scene / "images", video, args.fps, args.dry_run)
        goal = str(ep["planning_goal"])
        print(f"[{number:03d}/{total:03d}] {stem}", flush=True)

        if args.method in ("sg_ego", "both"):
            target = sg_data / "videos" / video.name
            if not args.dry_run and not target.exists():
                shutil.copy2(video, target)
            caption = sg_data / "captions" / "goal_roles" / f"{stem}.json"
            frame_graph = sg_data / "frame_graphs" / "goal_roles" / f"{stem}.json"
            video_graph = sg_data / "video_graphs" / "goal_roles" / f"{stem}.json"
            if args.dry_run or not caption.exists():
                run([sys.executable, "-m", "captioning.main", "--input-path", str(video),
                     "--output-path", str(sg_data / "captions" / "goal_roles"),
                     "--planning-goal", goal], sg_repo, args.dry_run)
            if not args.dry_run and not caption.exists():
                raise RuntimeError(f"SG-Ego captioning did not produce {caption}")
            if args.dry_run or not frame_graph.exists():
                run([sys.executable, "-m", "grounding.main", "--root", str(sg_data),
                     "--input-file", video.name, "--captions-version", "goal_roles",
                     "--frame-graphs-version", "goal_roles"], sg_repo, args.dry_run)
            if not args.dry_run and not frame_graph.exists():
                raise RuntimeError(f"SG-Ego grounding did not produce {frame_graph}")
            if args.dry_run or not video_graph.exists():
                run([sys.executable, "-m", "consolidation.main", "--root", str(sg_data),
                     "--input-file", video.name, "--frame-graphs-version", "goal_roles",
                     "--video-graphs-version", "goal_roles", "--window-size", str(ep["frame_count"]),
                     "--stride", str(ep["frame_count"])], sg_repo, args.dry_run)
            if not args.dry_run and not video_graph.exists():
                raise RuntimeError(f"SG-Ego consolidation did not produce {video_graph}")

        if args.method in ("svg2", "both"):
            svg_output = work_root / "svg2" / stem / "stage6_scene_graph.json"
            if args.dry_run or not svg_output.exists():
                run([sys.executable, "pipeline/svg2_pipeline.py", "--video", str(video),
                     "--output-dir", str(work_root / "svg2"), "--planning-goal", goal],
                    svg_repo, args.dry_run)


if __name__ == "__main__":
    main()
