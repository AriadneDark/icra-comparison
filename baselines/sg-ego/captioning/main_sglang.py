import argparse
import asyncio
import base64
import io
import json
import logging
import os
from os import path as osp

import decord
import torch
from PIL import Image
from decord import VideoReader, cpu
from openai import AsyncOpenAI
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm

from captioning.utils import parse_output
from utils.logging import configure_logging

decord.bridge.set_bridge("torch")

logger = logging.getLogger(__name__)


def tensor_to_base64(image_tensor: torch.Tensor) -> str:
    """Convert a CHW uint8 tensor to a base64-encoded PNG string."""
    pil_img = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class FramesDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading video frames from a video file.
    Each item is a (frame_idx, base64_image_string) pair.
    """

    def __init__(self, video_path: str, resolution: int = 384):
        super().__init__()
        self.video_path = video_path
        self.resolution = resolution

        self.vr = None
        self.num_frames = len(VideoReader(self.video_path, ctx=cpu(0)))

    def __len__(self) -> int:
        return self.num_frames

    def _get_frame(self, idx: int) -> torch.Tensor:
        if self.vr is None:
            # This should be instantiated only once in the worker process, not the main process
            self.vr = VideoReader(self.video_path, ctx=cpu(0))

        image = self.vr[idx].permute(2, 0, 1)  # (C, H, W)
        return resize(image, [self.resolution])

    def __getitem__(self, idx: int):
        return idx, tensor_to_base64(self._get_frame(idx))


def collate_fn(batch):
    idxs = [item[0] for item in batch]
    b64s = [item[1] for item in batch]
    return idxs, b64s


async def describe_frame(client: AsyncOpenAI, model: str, image_b64: str, prompt: str) -> tuple[str, int | None]:
    """Call the SGLang server for a single frame, with thinking disabled."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=256,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "repetition_penalty": 1.0,
        },
    )

    completion_tokens = response.usage.completion_tokens if response.usage is not None else None
    return response.choices[0].message.content, completion_tokens  # type: ignore


async def process_batch(client: AsyncOpenAI, model: str, frame_idxs: list, b64_frames: list, prompt: str) -> tuple[dict, int, int]:
    """Run a batch of frames concurrently — SGLang batches them server-side."""

    tasks = [describe_frame(client, model, b64, prompt) for b64 in b64_frames]
    outputs = await asyncio.gather(*tasks)

    results = {}
    total_output_tokens = 0
    counted_frames = 0

    for frame_idx, (output_text, completion_tokens) in zip(frame_idxs, outputs):
        triplets = parse_output(output_text)
        results[int(frame_idx)] = {"raw": output_text, "triplets": triplets}
        if completion_tokens is not None:
            total_output_tokens += completion_tokens
            counted_frames += 1

    return results, total_output_tokens, counted_frames


def main(args):
    logger.info("SG-Ego Stage 1: Captions generation (w/ SGLang)")
    logger.info("Video path: %s", args.video_path)
    logger.info("Output path: %s", args.output_path)

    logger.info("Reading the prompt for caption generation...")
    with open(args.prompt_file, "r") as f:
        prompt = f.read()

    logger.info("Preparing the dataset and dataloader...")
    dataset = FramesDataset(args.video_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    logger.info("Preparing the OpenAI-compatible client...")
    client = AsyncOpenAI(base_url=f"http://{args.sglang_host}:{args.sglang_port}/v1", api_key="EMPTY")

    all_results = {}
    total_output_tokens = 0
    counted_frames = 0

    async def run():
        nonlocal total_output_tokens, counted_frames

        progress_bar = tqdm(dataloader, desc=f"Generating captions for video {args.video_path}...")
        for frame_idxs, b64_frames in progress_bar:
            batch_results, batch_output_tokens, batch_counted_frames = await process_batch(client, args.model_name, frame_idxs, b64_frames, prompt)
            all_results.update(batch_results)
            total_output_tokens += batch_output_tokens
            counted_frames += batch_counted_frames

            if counted_frames > 0:
                progress_bar.set_postfix(avg_out_tokens=f"{total_output_tokens / counted_frames:.1f}")

    asyncio.run(run())

    # Once all frames are processed, save the results to a JSON file
    os.makedirs(args.output_path, exist_ok=True)
    out_file = osp.join(args.output_path, osp.splitext(osp.basename(args.video_path))[0] + ".json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    if counted_frames > 0:
        logger.info("Average output tokens/frame: %.1f", total_output_tokens / counted_frames)
    else:
        logger.warning("No completion token usage was returned by the SGLang server.")

    logger.info("Saved %s frame results to %s", len(all_results), out_file)


if __name__ == "__main__":
    configure_logging("captioning")

    arg_parser = argparse.ArgumentParser(description="SG-Ego Stage 1: Frame-level caption generation")

    # LLM model (only Qwen/Qwen3.5-9B was tested with this configuration)
    arg_parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-9B", help="The model to use for caption generation.")

    arg_parser.add_argument("--video-path", type=str, required=True, help="Path to an input video file.")
    arg_parser.add_argument("--output-path", type=str, required=True, help="Path to the output directory where results will be saved.")
    arg_parser.add_argument("--batch-size", type=int, default=128, help="Number of frames to process in each batch.")
    arg_parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for video loading.")

    # SGLang server port
    arg_parser.add_argument("--sglang-host", type=str, default="localhost", help="Host where the SGLang server is running.")
    arg_parser.add_argument("--sglang-port", type=int, default=30000, help="Port where the SGLang server is running.")

    # Path to the text file containing the prompt to use for caption generation.
    arg_parser.add_argument(
        "--prompt-file",
        type=str,
        default="captioning/prompt.txt",
        help="Path to a text file containing the prompt to use for caption generation. If not provided, a default prompt will be used.",
    )

    args = arg_parser.parse_args()

    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, sort_keys=True))

    main(args)
