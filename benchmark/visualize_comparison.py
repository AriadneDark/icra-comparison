#!/usr/bin/env python3
"""Render synchronized OUR | SG-EGO | SVG2 three-panel comparison videos."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .normalize import load_json, normalize_ours, normalize_sgego, normalize_svg2
except ImportError:
    from normalize import load_json, normalize_ours, normalize_sgego, normalize_svg2


ROLE_ORDER = ("robot", "manipulated_object", "initial_support", "target")
# BGR colors, shared by all three methods.
ROLE_COLORS = {
    "robot": (70, 210, 255),
    "manipulated_object": (80, 220, 80),
    "initial_support": (255, 155, 50),
    "target": (220, 80, 220),
}
METHOD_TITLES = {"ours": "OUR", "sg_ego": "SG-EGO", "svg2": "SVG2"}


def _ours_graph(root: Path, relative_path: str) -> Path:
    stem = relative_path.replace("/", "__")
    flat = root / "scene_graphs" / f"{stem}.json"
    return flat if flat.exists() else root / "scenes" / relative_path / "scene_graph.json"


def _sg_paths(root: Path, relative_path: str) -> tuple[Path, Path]:
    stem = relative_path.replace("/", "__")
    return (
        root / "frame_graphs" / "goal_roles" / f"{stem}.json",
        root / "video_graphs" / "goal_roles" / f"{stem}.json",
    )


def _svg_graph(root: Path, relative_path: str) -> Path:
    stem = relative_path.replace("/", "__")
    nested = root / stem / "stage6_scene_graph.json"
    return nested if nested.exists() else root / f"{stem}.json"


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    active = mask > 0
    if not active.any():
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[active] = (0.62 * image[active] + 0.38 * color_array).astype(np.uint8)
    contours, _ = cv2.findContours(active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)


def _label_box(
    image: np.ndarray, box: list[float] | None, role: str, name: str = "", normalized: bool = False
) -> None:
    if not box or len(box) != 4:
        return
    h, w = image.shape[:2]
    values = list(map(float, box))
    if normalized:
        values = [values[0] * w, values[1] * h, values[2] * w, values[3] * h]
    x1, y1, x2, y2 = [int(round(x)) for x in values]
    x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
    y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
    color = ROLE_COLORS[role]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text = role if not name else f"{role}: {name}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    label_x = max(0, min(x1, w - tw - 6))
    top = max(0, y1 - th - 7)
    cv2.rectangle(image, (label_x, top), (min(w - 1, label_x + tw + 6), y1), color, -1)
    cv2.putText(image, text, (label_x + 3, max(th + 1, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (15, 15, 15), 1, cv2.LINE_AA)


def _draw_ours(image: np.ndarray, graph_frame: dict[str, Any], mask_root: Path, frame_idx: int) -> None:
    for node in graph_frame.get("nodes", []):
        role = node.get("role") or node.get("entity_id")
        if role not in ROLE_COLORS or node.get("status", "visible") != "visible":
            continue
        mask_path = mask_root / role / f"frame_{frame_idx:06d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _overlay_mask(image, mask, ROLE_COLORS[role])
        _label_box(image, node.get("bbox_xyxy"), role, node.get("canonical_name", ""))


def _draw_sgego(image: np.ndarray, graph_frame: dict[str, Any]) -> None:
    objects, roles, boxes = graph_frame.get("obj", []), graph_frame.get("role", []), graph_frame.get("bbox", [])
    for idx, role in enumerate(roles):
        if role in ROLE_COLORS and idx < len(boxes):
            name = str(objects[idx]) if idx < len(objects) else ""
            _label_box(image, boxes[idx], role, name, normalized=True)


def _decode_svg_mask(rle: dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    try:
        from pycocotools import mask as mask_utils
        encoded = {"size": rle["size"], "counts": rle["counts"]}
        mask = mask_utils.decode(encoded)
        return mask[..., 0] if mask.ndim == 3 else mask
    except (ImportError, ValueError, TypeError):
        return None


def _draw_svg2(image: np.ndarray, objects: list[dict[str, Any]], frame_idx: int) -> None:
    for obj in objects:
        role = obj.get("role")
        if role not in ROLE_COLORS:
            continue
        trajectory = obj.get("trajectory", {})
        masks = trajectory.get("masks", [])
        if frame_idx < len(masks):
            mask = _decode_svg_mask(masks[frame_idx])
            if mask is not None:
                _overlay_mask(image, mask, ROLE_COLORS[role])
        box = trajectory.get("boxes", {}).get(str(frame_idx))
        _label_box(image, box, role, str(obj.get("name", "")))


def _edge_lines(canonical: dict[str, Any] | None, frame_idx: int) -> list[str]:
    if canonical is None or frame_idx >= len(canonical.get("frames", [])):
        return ["result missing"]
    edges = canonical["frames"][frame_idx].get("edges", [])
    return [f"{s} --{p}--> {o}" for s, p, o in edges] or ["no relevant edges"]


def _make_panel(
    image: np.ndarray, method: str, goal: str, edges: list[str], panel_width: int, missing: bool
) -> np.ndarray:
    image_height = max(1, round(image.shape[0] * panel_width / image.shape[1]))
    resized = cv2.resize(image, (panel_width, image_height), interpolation=cv2.INTER_AREA)
    header, footer = 96, 138
    panel = np.full((header + image_height + footer, panel_width, 3), 25, dtype=np.uint8)
    panel[header:header + image_height] = resized
    cv2.putText(panel, METHOD_TITLES[method], (16, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, (245, 245, 245), 2, cv2.LINE_AA)
    status = "MISSING" if missing else "task roles only"
    cv2.putText(panel, status, (16, 57), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (80, 80, 255) if missing else (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Goal: {goal}"[:72], (16, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (170, 170, 170), 1, cv2.LINE_AA)
    y = header + image_height + 23
    for line in edges[:5]:
        for wrapped in textwrap.wrap(line, width=54) or [""]:
            cv2.putText(panel, wrapped, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.39, (225, 225, 225), 1, cv2.LINE_AA)
            y += 18
            if y >= panel.shape[0] - 8:
                break
    return panel


def render_episode(
    episode: dict[str, Any], source_root: Path, ours_root: Path, sg_root: Path,
    svg_root: Path, output: Path, fps: float, panel_width: int, allow_missing: bool,
) -> None:
    relative_path = episode["relative_path"]
    image_dir = source_root / relative_path / "images"
    images = sorted(image_dir.glob("frame_*.png"))
    if len(images) != int(episode["frame_count"]):
        raise ValueError(f"Expected {episode['frame_count']} frames in {image_dir}, found {len(images)}")

    ours_path = _ours_graph(ours_root, relative_path)
    sg_frame_path, sg_video_path = _sg_paths(sg_root, relative_path)
    svg_path = _svg_graph(svg_root, relative_path)
    required = {"ours": ours_path, "sg_ego": sg_video_path, "svg2": svg_path}
    missing = {method for method, path in required.items() if not path.exists()}
    if not allow_missing and missing:
        details = ", ".join(f"{m}: {required[m]}" for m in sorted(missing))
        raise FileNotFoundError(f"Missing comparison results for {relative_path}: {details}")

    ours_data = load_json(ours_path) if ours_path.exists() else None
    sg_frames = load_json(sg_frame_path) if sg_frame_path.exists() else {}
    sg_video = load_json(sg_video_path) if sg_video_path.exists() else None
    svg_data = load_json(svg_path) if svg_path.exists() else None
    canonical = {
        "ours": normalize_ours(ours_data) if ours_data else None,
        "sg_ego": normalize_sgego(sg_video, len(images)) if sg_video else None,
        "svg2": normalize_svg2(svg_data) if svg_data else None,
    }

    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot read {images[0]}")
    panel_height = 96 + round(first.shape[0] * panel_width / first.shape[1]) + 138
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (panel_width * 3, panel_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {output}")
    try:
        for frame_idx, image_path in enumerate(images):
            raw = cv2.imread(str(image_path))
            if raw is None:
                raise ValueError(f"Cannot read {image_path}")
            method_images = {method: raw.copy() for method in METHOD_TITLES}
            if ours_data:
                _draw_ours(method_images["ours"], ours_data["frames"][frame_idx],
                           ours_root / "scenes" / relative_path / "masks", frame_idx)
            if sg_frame_path.exists():
                _draw_sgego(method_images["sg_ego"], sg_frames.get(str(frame_idx), {}))
            if svg_data:
                _draw_svg2(method_images["svg2"], svg_data.get("objects", []), frame_idx)
            panels = [
                _make_panel(method_images[method], method, episode["planning_goal"],
                            _edge_lines(canonical[method], frame_idx), panel_width, method in missing)
                for method in ("ours", "sg_ego", "svg2")
            ]
            combined = np.concatenate(panels, axis=1)
            writer.write(combined)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--ours-root", required=True)
    parser.add_argument("--sg-ego-root", required=True)
    parser.add_argument("--svg2-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode", help="Render only this manifest relative_path")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--panel-width", type=int, default=520)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Render missing result panels explicitly instead of failing.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    episodes = manifest.get("episodes", [])
    if args.episode:
        episodes = [ep for ep in episodes if ep["relative_path"] == args.episode]
        if not episodes:
            raise SystemExit(f"Episode not found in manifest: {args.episode}")
    output_root = Path(args.output_root).resolve()
    for index, episode in enumerate(episodes, 1):
        stem = episode["relative_path"].replace("/", "__")
        output = output_root / f"{stem}.mp4"
        print(f"[{index:03d}/{len(episodes):03d}] {episode['relative_path']} -> {output}", flush=True)
        render_episode(
            episode, Path(args.source_root).resolve(), Path(args.ours_root).resolve(),
            Path(args.sg_ego_root).resolve(), Path(args.svg2_root).resolve(), output,
            args.fps, args.panel_width, args.allow_missing,
        )


if __name__ == "__main__":
    main()
