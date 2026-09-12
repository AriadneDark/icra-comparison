#!/usr/bin/env python3
"""Time one representative multimodal request to the local evaluation judge."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from run_vlm_evaluation import model_request_args, proposer_prompt, sample_indices


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def encode_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def load_sampled_frames(source: Path, maximum: int) -> tuple[list[int], list[Image.Image], int]:
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES)
        if not paths:
            raise RuntimeError(f"No images found under {source}")
        indices = sample_indices(len(paths), maximum)
        images: list[Image.Image] = []
        for index in indices:
            with Image.open(paths[index]) as image:
                images.append(image.convert("RGB"))
        return indices, images, len(paths)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video reports no frames: {source}")
    indices = sample_indices(frame_count, maximum)
    images = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} from {source}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return indices, images, frame_count


def usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="MP4 file or directory containing ordered frames")
    parser.add_argument("--goal", required=True, help="Planning goal supplied to the judge")
    parser.add_argument("--base-url", default="http://gemma-judge:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()

    source = Path(args.video).resolve()
    if not source.exists():
        raise SystemExit(f"Input does not exist: {source}")
    if args.frames < 1:
        raise SystemExit("--frames must be positive")

    from openai import OpenAI

    client = OpenAI(api_key="local", base_url=args.base_url, timeout=args.timeout, max_retries=0)
    model = args.model
    discovery_started = time.perf_counter()
    if model == "auto":
        available = [entry.id for entry in client.models.list().data]
        if not available:
            raise SystemExit(f"No models reported by {args.base_url}")
        model = available[0]
    discovery_seconds = time.perf_counter() - discovery_started

    preprocessing_started = time.perf_counter()
    indices, images, frame_count = load_sampled_frames(source, args.frames)
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": encode_image(image)}}
        for image in images
    ]
    content.extend(
        {"type": "text", "text": f"Image {position} is original frame {frame_index}."}
        for position, frame_index in enumerate(indices, 1)
    )
    content.append({"type": "text", "text": proposer_prompt(args.goal, frame_count)})
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    request_args = model_request_args(model)
    request_args["max_tokens"] = args.max_output_tokens
    request_started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        **request_args,
    )
    request_seconds = time.perf_counter() - request_started
    completion_tokens = usage_value(response.usage, "completion_tokens")
    summary = {
        "model": model,
        "video": str(source),
        "video_frame_count": frame_count,
        "sampled_frame_indices": indices,
        "sampled_images": len(images),
        "model_discovery_seconds": round(discovery_seconds, 3),
        "preprocessing_seconds": round(preprocessing_seconds, 3),
        "request_seconds": round(request_seconds, 3),
        "total_seconds": round(discovery_seconds + preprocessing_seconds + request_seconds, 3),
        "prompt_tokens": usage_value(response.usage, "prompt_tokens"),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": (
            round(completion_tokens / request_seconds, 3)
            if completion_tokens is not None and request_seconds > 0 else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n--- model response ---", flush=True)
    print(response.choices[0].message.content, flush=True)


if __name__ == "__main__":
    main()
