import argparse
import json
import logging
import os
import os.path as osp
from pathlib import Path

import torch
from decord import VideoReader, cpu
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from captioning.utils import parse_output
from utils.logging import configure_logging

logger = logging.getLogger(__name__)

ROLE_NAMES = ("robot", "manipulated_object", "initial_support", "target")


def load_planning_goal(goal: str | None, goal_json: str | None) -> str:
    """Load the required task goal from a literal string or a JSON file."""
    if bool(goal) == bool(goal_json):
        raise ValueError("Provide exactly one of --planning-goal or --planning-goal-json")
    if goal_json:
        with open(goal_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        goal = payload.get("planning_goal") if isinstance(payload, dict) else None
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("planning_goal must be a non-empty string")
    return goal.strip()


def goal_guided_prompt(base_prompt: str, goal: str) -> str:
    """Constrain SG-Ego triplets to the four task roles.

    The ``role::visual_name_N`` syntax preserves a machine-readable role while
    GroundingDINO receives only ``visual name`` (see ``grounding/dataset.py``).
    """
    return f"""{base_prompt.rstrip()}

PLANNING GOAL: {goal}

TASK-RELEVANCE CONSTRAINT (overrides exhaustiveness instructions above):
- Output triplets involving ONLY these semantic roles: robot, manipulated_object,
  initial_support (where the manipulated object starts), and target (its intended
  destination/support/receptacle). Ignore every other object even if salient.
- Use an entity only when it is visually present. A role may be absent.
- Every subject and object MUST use the syntax role::visual_name_N, where role is
  exactly one of {', '.join(ROLE_NAMES)}, visual_name is a short visually groundable
  noun phrase, and N is a stable instance number. Example:
  (manipulated_object::red_block_1, on, initial_support::table_1)
- The same physical entity may fill two roles; keep the role required by the goal
  and never invent a duplicate node merely to fill all four roles.
- Do not output an entity or relation outside this four-role induced subgraph.
"""


def keep_role_triplets(triplets: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Enforce the four-role postcondition even when the model ignores the prompt."""
    prefixes = tuple(f"{role}::" for role in ROLE_NAMES)
    return [t for t in triplets if t[0].startswith(prefixes) and t[2].startswith(prefixes)]


class FramesDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading inputs from either:
    - a video file
    - a directory containing frame images
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, input_path: str, prompt: str, resolution: int = 384, skip_frames=None):
        super().__init__()

        # If i move this outside, everything crashes...
        import decord

        decord.bridge.set_bridge("torch")

        self.input_path = input_path
        self.prompt = prompt

        self.resolution = resolution

        # Keep track of the frame indices to skip
        self.skip_frames = {str(x) for x in (skip_frames or set())}
        self.vr = None

        self.source_type, self.items = self.prepare_items()

    def prepare_items(self) -> tuple[str, list[dict]]:

        if os.path.isdir(self.input_path):
            self.source_type = "frames_dir"
            frame_paths = sorted([p for p in Path(self.input_path).iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS])

            if not frame_paths:
                raise ValueError(f"No image frames found in directory: {self.input_path}")

            items = [
                {"source_idx": i, "frame_path": str(p), "output_id": p.stem} for i, p in enumerate(frame_paths) if p.stem not in self.skip_frames and str(i) not in self.skip_frames
            ]
            return "frames_dir", items

        elif os.path.isfile(self.input_path):
            # Preload the video reader if the source is a video file
            num_frames = len(VideoReader(self.input_path, ctx=cpu(0)))
            items = [
                {"source_idx": i, "output_id": str(i)}
                for i in range(num_frames)
                if str(i) not in self.skip_frames and str(i) not in self.skip_frames
            ]
            return "video", items

        else:
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

    def __len__(self):
        return len(self.items)

    def _get_frame(self, idx: int) -> torch.Tensor:
        """Get the idx-th frame from the video / frames directory.


        Parameters
        ----------
        idx : int
            The index of the frame to retrieve.

        Returns
        -------
        torch.Tensor
            The image tensor of the frame, resized to the specified resolution.
        """
        item = self.items[idx]

        if self.source_type == "video":
            if self.vr is None:
                self.vr = VideoReader(self.input_path)
            image = self.vr[item["source_idx"]].permute(2, 0, 1)
        else:
            image = read_image(item["frame_path"], mode=ImageReadMode.RGB)

        return resize(image, self.resolution)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[str, list[dict]]:
        """Get the idx-th frame from the video / frames directory.
        This method directly returns a conversation object that can be fed to the LLM preprocessor.

        Parameters
        ----------
        idx : int
            The index of the frame to retrieve.

        Returns
        -------
        tuple[str, list[dict]]
            A tuple containing the output ID and the conversation object for the frame.
        """
        item = self.items[idx]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._get_frame(idx)},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        return item["output_id"], conversation


def collate_fn(batch):
    return [idx for idx, _ in batch], [messages for _, messages in batch]


def count_generated_tokens(token_ids: torch.Tensor, eos_token_id: int | list[int] | None) -> int:
    if eos_token_id is None:
        return len(token_ids)

    eos_token_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

    for idx, token_id in enumerate(token_ids.tolist()):
        if token_id in eos_token_ids:
            return idx

    return len(token_ids)


def load_captioning_resources(model_name: str):
    """Load the captioning processor/model once for single or batch inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("SG-Ego captioning requires a visible CUDA GPU")
    logger.info("Preparing the model and processor for '%s'...", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    return processor, model, "cuda"


def main(args, resources=None):
    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Input path: %s", args.input_path)
    logger.info("Output path: %s", args.output_path)

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute() and not prompt_path.is_file():
        # Programmatic callers (for example the benchmark batch runner) do not
        # necessarily execute with the SG-Ego repository as their cwd.
        repo_relative_prompt = Path(__file__).resolve().parents[1] / prompt_path
        if repo_relative_prompt.is_file():
            prompt_path = repo_relative_prompt
    logger.info("Reading the prompt for caption generation from %s...", prompt_path)
    with prompt_path.open("r", encoding="utf-8") as f:
        prompt = f.read()
    planning_goal = load_planning_goal(args.planning_goal, args.planning_goal_json)
    prompt = goal_guided_prompt(prompt, planning_goal)

    if resources is None:
        resources = load_captioning_resources(args.model_name)
    processor, model, device = resources

    logger.info("")
    logger.info("Preparing the dataset and dataloader...")
    existing_caption_ids = {Path(f).stem for f in os.listdir(args.output_path) if f.endswith(".json")}
    dataset = FramesDataset(args.input_path, prompt=prompt, skip_frames=existing_caption_ids)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if len(dataset) == 0:
        logger.info("All captions already exist. Exiting.")
        return

    logger.info("")
    logger.info("Starting caption generation...")

    all_captions = {}

    processed_frames = 0
    total_output_tokens = 0
    total_triplets = 0

    progress_bar = tqdm(dataloader, desc="Generating captions...")

    for item_ids, batch in progress_bar:

        # pre-process the batch data
        batch = processor.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            enable_thinking=False
        ).to(device)
        input_token_counts = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()

        # generate the captions
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            model.model.rope_deltas = None
            generated_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=True,
            )

        # decode the generated captions and save them to the output directory
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print(output_text)

        generated_token_counts = [count_generated_tokens(token_ids, processor.tokenizer.eos_token_id) for token_ids in generated_ids_trimmed]
        parsed_outputs = [keep_role_triplets(parse_output(caption)) for caption in output_text]

        processed_frames += len(item_ids)
        total_output_tokens += sum(generated_token_counts)
        total_triplets += sum(len(parsed_triplets) for parsed_triplets in parsed_outputs)

        avg_output_tokens = total_output_tokens / processed_frames
        avg_triplets = total_triplets / processed_frames
        progress_bar.set_postfix(
            avg_out_tokens=f"{avg_output_tokens:.1f}",
            avg_triplets=f"{avg_triplets:.2f}",
        )

        for item_id, caption, parsed_triplets in zip(item_ids, output_text, parsed_outputs):
            all_captions[item_id] = {"raw": caption, "parsed": parsed_triplets}

    # Once all frames are processed, save the results to a JSON file
    os.makedirs(args.output_path, exist_ok=True)
    out_file = osp.join(args.output_path, osp.splitext(osp.basename(args.input_path))[0] + ".json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2)

    logger.info(
        "Generation complete! Average output tokens/frame: %.1f | Average triplets/frame: %.2f",
        total_output_tokens / processed_frames,
        total_triplets / processed_frames,
    )


if __name__ == "__main__":
    configure_logging("captioning")

    arg_parser = argparse.ArgumentParser(description="SG-Ego Stage 1: Frame-level caption generation")

    # Input video or frames directory
    arg_parser.add_argument("--input-path", type=str, help="Path to an input video file or frames dir.")

    # LLM model (only Qwen/Qwen3.5-9B was tested with this configuration)
    arg_parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-9B", help="The model to use for caption generation.")

    # Generation parameters
    arg_parser.add_argument("--output-path", type=str, default="output_captions", help="Path to the output captions.")
    arg_parser.add_argument("--max-new-tokens", type=int, default=192, help="Maximum number of new tokens to generate for each caption.")
    arg_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for caption generation.")
    arg_parser.add_argument("--top-p", type=float, default=0.8, help="Top-p (nucleus) sampling parameter for caption generation.")
    arg_parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for caption generation.")

    arg_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for caption generation.")
    arg_parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for loading video frames or images.")

    goal_group = arg_parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--planning-goal", type=str, help="Natural-language planning goal.")
    goal_group.add_argument("--planning-goal-json", type=str, help="JSON file containing a planning_goal string.")

    # Path to the text file containing the prompt to use for caption generation.
    arg_parser.add_argument(
        "--prompt-file",
        type=str,
        default="captioning/prompt.txt",
        help="Path to a text file containing the prompt to use for caption generation. If not provided, a default prompt will be used.",
    )

    args = arg_parser.parse_args()

    logger.info("##################################################")
    logger.info("# SG-Ego Stage 1: Frame-level caption generation #")
    logger.info("##################################################")
    logger.info("")
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, sort_keys=True))

    main(args)
