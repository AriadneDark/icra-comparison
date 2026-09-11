#!/usr/bin/env python3
"""Quality-preserving stage-wise batch runner for the selected benchmark videos."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from run_selected_100 import make_video


LOG = logging.getLogger("benchmark.batch")


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def load_jobs(args: argparse.Namespace) -> tuple[Path, Path, list[dict]]:
    source_root = Path(args.source_root).resolve()
    work_root = Path(args.work_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise SystemExit("Manifest has no episodes")
    if args.expected_count is not None and len(episodes) != args.expected_count:
        raise SystemExit(
            f"Expected exactly {args.expected_count} manifest episodes, got {len(episodes)}"
        )
    if args.episode:
        episodes = [ep for ep in episodes if ep["relative_path"] == args.episode]
        if not episodes:
            raise SystemExit(f"Episode not found in manifest: {args.episode}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        episodes = episodes[:args.limit]

    jobs = []
    for ep in episodes:
        scene = source_root / ep["relative_path"]
        images = scene / "images"
        if not images.is_dir():
            raise SystemExit(f"Missing input image directory: {images}")
        stem = ep["relative_path"].replace("/", "__")
        video = work_root / "videos" / f"{stem}.mp4"
        make_video(images, video, args.fps, args.dry_run)
        jobs.append({
            "stem": stem,
            "video": video,
            "goal": str(ep["planning_goal"]),
            "frame_count": int(ep["frame_count"]),
        })
    return source_root, work_root, jobs


def require_outputs(paths: list[Path], stage: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{stage} did not produce {len(missing)} outputs; first: {missing[0]}")


def run_sg_ego(repo: Path, work_root: Path, jobs: list[dict], args: argparse.Namespace) -> None:
    sg_root = work_root / "sg_ego"
    captions = sg_root / "captions" / "goal_roles"
    frame_graphs = sg_root / "frame_graphs" / "goal_roles"
    video_graphs = sg_root / "video_graphs" / "goal_roles"
    for directory in (
        sg_root / "videos",
        sg_root / "logs",
        captions,
        frame_graphs,
        video_graphs,
        work_root / "jobs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        target = sg_root / "videos" / job["video"].name
        if not args.dry_run and not target.exists():
            shutil.copy2(job["video"], target)

    pending = [job for job in jobs if not (captions / f'{job["stem"]}.json').exists()]
    if pending:
        LOG.info("SG-Ego caption batch: %d videos; loading Qwen once", len(pending))
        if args.dry_run:
            print(f"[dry-run] caption {len(pending)} videos with one model load")
        else:
            sys.path.insert(0, str(repo))
            from captioning.main import load_captioning_resources, main as caption_one

            resources = load_captioning_resources(args.sg_caption_model)
            for index, job in enumerate(pending, 1):
                LOG.info("SG-Ego caption [%d/%d] %s", index, len(pending), job["stem"])
                caption_one(Namespace(
                    input_path=str(job["video"]), output_path=str(captions),
                    planning_goal=job["goal"], planning_goal_json=None,
                    prompt_file=str(repo / "captioning" / "prompt.txt"),
                    model_name=args.sg_caption_model,
                    max_new_tokens=192, temperature=0.7, top_p=0.8,
                    repetition_penalty=1.0, batch_size=args.sg_caption_batch_size,
                    num_workers=args.workers,
                ), resources=resources)
            del resources
            release_cuda()
    if not args.dry_run:
        require_outputs([captions / f'{job["stem"]}.json' for job in jobs], "SG-Ego captioning")

    pending = [job for job in jobs if not (frame_graphs / f'{job["stem"]}.json').exists()]
    if pending:
        job_file = work_root / "jobs" / "sg_grounding.txt"
        if not args.dry_run:
            job_file.write_text("".join(f'{job["video"].name}\n' for job in pending), encoding="utf-8")
        run([
            sys.executable, "-m", "grounding.main", "--root", str(sg_root),
            "--job-file", str(job_file), "--captions-version", "goal_roles",
            "--frame-graphs-version", "goal_roles", "--num-workers", str(args.workers),
        ], repo, args.dry_run)
    if not args.dry_run:
        require_outputs([frame_graphs / f'{job["stem"]}.json' for job in jobs], "SG-Ego grounding")

    pending = [job for job in jobs if not (video_graphs / f'{job["stem"]}.json').exists()]
    if pending:
        job_file = work_root / "jobs" / "sg_consolidation.txt"
        if not args.dry_run:
            job_file.write_text(
                "".join(f'{job["video"].name},0-{job["frame_count"]}\n' for job in pending),
                encoding="utf-8",
            )
        run([
            sys.executable, "-m", "consolidation.main", "--root", str(sg_root),
            "--job-file", str(job_file), "--frame-graphs-version", "goal_roles",
            "--video-graphs-version", "goal_roles",
        ], repo, args.dry_run)
    if not args.dry_run:
        require_outputs([video_graphs / f'{job["stem"]}.json' for job in jobs], "SG-Ego consolidation")


def release_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_svg2(repo: Path, work_root: Path, jobs: list[dict], args: argparse.Namespace) -> None:
    output_root = work_root / "svg2"
    output_root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not (output_root / job["stem"] / "stage6_scene_graph.json").exists()]
    if not pending:
        LOG.info("SVG2: all %d selected outputs already exist", len(jobs))
        return
    if args.dry_run:
        for stage in range(1, 7):
            print(f"[dry-run] SVG2 stage {stage} over {len(pending)} videos with one resource load")
        return

    sys.path.insert(0, str(repo))
    from pipeline.svg2_pipeline import Pipeline, artifact_path, build_arg_parser, config_from_args

    parser = build_arg_parser()
    for stage in range(1, 7):
        shared_resources: dict = {}
        stage_jobs = []
        for job in pending:
            parsed = parser.parse_args([
                "--video", str(job["video"]), "--output-dir", str(output_root),
                "--planning-goal", job["goal"], "--start-stage", str(stage),
                "--end-stage", str(stage),
            ])
            cfg = config_from_args(parsed)
            if not artifact_path(cfg, stage).exists():
                stage_jobs.append((job, cfg))
        LOG.info("SVG2 stage %d: %d pending videos", stage, len(stage_jobs))
        for index, (job, cfg) in enumerate(stage_jobs, 1):
            LOG.info("SVG2 stage %d [%d/%d] %s", stage, index, len(stage_jobs), job["stem"])
            Pipeline(cfg, shared_resources=shared_resources).run()
        shared_resources.clear()
        release_cuda()

    require_outputs(
        [output_root / job["stem"] / "stage6_scene_graph.json" for job in jobs],
        "SVG2",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--method", choices=("prepare", "sg_ego", "svg2"), required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--episode")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int,
                        help="Optional manifest-size guard (for example 100 or 1000).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sg-caption-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sg-caption-batch-size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo = Path(__file__).resolve().parents[1]
    _, work_root, jobs = load_jobs(args)
    LOG.info("Selected %d videos", len(jobs))
    if args.method == "prepare":
        return
    if args.method == "sg_ego":
        run_sg_ego(repo / "baselines" / "sg-ego", work_root, jobs, args)
    else:
        run_svg2(repo / "baselines" / "svg2", work_root, jobs, args)


if __name__ == "__main__":
    main()
