"""SG-Ego Stage 3: Consolidation of frame-level graphs into video-level graphs"""

import json
import logging
import os
from argparse import ArgumentParser
from collections import namedtuple
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import json_repair
import torch
from tqdm.auto import tqdm
from transformers import (Sam2Model, Sam2Processor, Sam2VideoModel,
                          Sam2VideoProcessor)

from consolidation.dinov2_fe import DinoV2FE

from accelerate import Accelerator

from consolidation.consolidate import consolidate
from consolidation.dataset import VideoDataset
from utils.logging import configure_logging

torch.set_grad_enabled(False)

# Data type for the history entries in the consolidation process.
# Each history entry refers to a specific instance of an object in a frame, along with its label and confidence score.
HistoryEntry = namedtuple("HistoryEntry", ["frame_idx", "obj_idx_in_frame", "name", "confidence_score"])

device = Accelerator().device

logger = logging.getLogger(__name__)




def read_job_file(job_path: str) -> Tuple[List[str], List[Optional[List[int]]]]:
    
    assert os.path.exists(job_path), f"Job path {job_path} does not exist"

    video_ids, video_ranges = [], []
    with open(job_path, "r") as f:
        for line in f:
            video_id, *frames = line.strip().split(",")
            video_ids.append(video_id)
            video_ranges.append(
                [(int(_range.split("-")[0]), int(_range.split("-")[1])) for _range in frames]
                if frames else None
            )

    return video_ids, video_ranges


def main(args):
    """Run consolidation for a set of videos.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments controlling dataset paths, selection mode,
        window configuration, and output versioning.

    This function loads the SAM2 tracking and grounding models once, then
    writes one JSON file per processed video.
    """
    logger.info(
        "Runtime device: %s (torch.cuda.is_available=%s, device_count=%d)",
        device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if str(args.device).startswith("cuda") and device.type != "cuda":
        raise RuntimeError(
            "CUDA was requested but is not visible inside the SG-Ego container. "
            "Run './benchmark/docker-run.sh smoke' and verify Docker GPU access."
        )
    
    videos_path = os.path.join(args.root, "videos")
    frame_graphs_path = os.path.join(args.root, "frame_graphs", args.frame_graphs_version)
    video_graphs_path = os.path.join(args.root, "video_graphs", args.video_graphs_version)

    os.makedirs(video_graphs_path, exist_ok=True)

    logger.info("")
    logger.info(f"Raw videos path: {videos_path}.")
    logger.info(f"Frame-level graphs path: {frame_graphs_path}.")
    logger.info(f"Video-level graphs path: {video_graphs_path}.")
    logger.info("")

    if args.input_file:
        video_filename = os.path.basename(args.input_file)
        video_id = os.path.splitext(video_filename)[0]
        video_ids, video_filenames, video_ranges = [video_id], [video_filename], [None]
    else:  # args.job_file
        video_filenames, video_ranges = read_job_file(args.job_file)
        video_ids = [os.path.splitext(os.path.basename(name))[0] for name in video_filenames]

    logger.info(f"About to process the following {len(video_ids)} videos.")

    # Build the tracking, grounding, and DINO models once
    logger.info("Building the SAM2 tracking model...")
    tracking_model = Sam2VideoModel.from_pretrained(f"facebook/{args.sam2_variant}").to(device, dtype=torch.bfloat16).eval()
    tracking_processor = Sam2VideoProcessor.from_pretrained(f"facebook/{args.sam2_variant}")
    logger.info("Building the SAM2 grounding model...")
    grounding_model = Sam2Model.from_pretrained(f"facebook/{args.sam2_variant}").to(device, dtype=torch.bfloat16).eval()
    grounding_processor = Sam2Processor.from_pretrained(f"facebook/{args.sam2_variant}")
    
    logger.info("Building the DINO model...")
    dino_model = DinoV2FE(variant=args.dino_variant).to(device).eval()
    logger.info(
        "Models ready on tracking=%s, grounding=%s, dino=%s",
        next(tracking_model.parameters()).device,
        next(grounding_model.parameters()).device,
        next(dino_model.parameters()).device,
    )

    pbar = tqdm(zip(video_ids, video_filenames, video_ranges), total=sum(len(ranges) if ranges is not None else 1 for ranges in video_ranges))
    for video_id, video_filename, video_range in pbar:

        try:
            consolidated_graphs = dict()

            fn = f"{video_id}_mapped.json" if args.use_mapped_graphs else f"{video_id}.json"
            if video_range is not None and os.path.exists(os.path.join(video_graphs_path, fn)):
                consolidated_graphs: dict = json_repair.load(open(os.path.join(video_graphs_path, fn), "r"))  # type: ignore
                video_range = [(s, e) for s, e in video_range if f"{s}-{e}" not in consolidated_graphs.keys()]  # type: ignore

                if len(video_range) == 0:
                    continue

            logger.info(f"Starting extraction for {video_id}...")

            dataset = VideoDataset(
                video_path=os.path.join(videos_path, video_filename),
                graphs_path=os.path.join(frame_graphs_path, os.path.basename(os.path.splitext(video_id)[0])),
                window_size=args.window_size,
                stride=args.stride,
                ranges=video_range,  # type: ignore
                use_mapped_graphs=args.use_mapped_graphs,
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=lambda x: x[0])  # type: ignore

            for frames, graphs, start, end in tqdm(dataloader, total=len(dataloader)):

                try:
                    frames = frames.to(device)

                    if start == end or all(len(graph["obj"]) == 0 for graph in graphs):
                        graph = {"obj": [], "pair": [], "rel": [], "bbox": [], "confidence": [], "frame_idx": [], "history": []}
                    else:
                        graph = consolidate(
                            frames,
                            graphs,
                            tracking_model,
                            tracking_processor,
                            grounding_model,
                            grounding_processor,
                            dino_model,
                            semantic_similarity_threshold=args.semantic_similarity_threshold,
                            enforce_label_consistency=args.enforce_label_consistency,
                        )

                except Exception as e:
                    graph = {"obj": [], "pair": [], "rel": [], "masks": [], "timespans": [], "frame_idx": [], "history": [], "bbox": [], "confidence": []}
                    logger.error(f"ERROR: processing video window {video_id} ({start}, {end}): {e}")

                consolidated_graphs[f"{start}-{end}"] = graph
                assert len([obj for obj in graph["obj"] if obj == "main actor"]) <= 1, "Duplicate main actor detected in consolidated graph, there should be at most one per window"

            with open(os.path.join(video_graphs_path, fn), "w", encoding="utf-8") as f:
                json.dump(consolidated_graphs, f, indent=2)

        except Exception as e:
            logger.error(f"ERROR: processing video {video_id}: {e}")

        pbar.update(len(video_range) if video_range is not None else 1)


if __name__ == "__main__":
    configure_logging("consolidation")

    arg_parser = ArgumentParser(description="SG-Ego Stage 3: Consolidation of frame-level graphs into video-level graphs")
    
    # Dataset root path
    arg_parser.add_argument("--root", type=str, help="Path to the video file", default="data/sg_ego")
    
    # Input can be either:
    # - a video file, in which case we process the entire video using a fixed sliding window, or
    # - a txt file containing the list of videos to process, each with the corresponding list of target windows
    input_selection_group = arg_parser.add_mutually_exclusive_group(required=True)
    input_selection_group.add_argument("--input-file", type=str, default=None, help="Path to the input mp4 file", required=False)
    input_selection_group.add_argument("--job-file", type=str, default=None, help="Path to the job file", required=False)

    # Versioning arguments for frame and video graphs
    arg_parser.add_argument("--frame-graphs-version", type=str, help="Frame graphs version", default="v1")
    arg_parser.add_argument("--video-graphs-version", type=str, help="Video graphs version", default="v1")

    # Consolidation-specific arguments
    # The following arguments are used only when an input file is provided, i.e., when we process a single video with a sliding window
    arg_parser.add_argument("--window-size", type=int, default=10, help="Size of the window for processing frames")
    arg_parser.add_argument("--stride", type=int, default=10, help="Stride for the sliding window")
    
    # Use mapped graphs for consolidation (if available)
    # In mapped graphs, the labels have been projected to a closed-set label space, which can improve the consolidation process,
    # when paired with the --enforce-label-consistency flag, which ensures that the labels are consistent across frames for the same object.
    arg_parser.add_argument("--use-mapped-graphs", action="store_true", help="Whether to use mapped graphs for consolidation")
    arg_parser.add_argument("--enforce-label-consistency", action="store_true", help="Whether to enforce label consistency during matching")
    
    # DINO-based matching fallback threshold
    # This threshold controls the cosine similarity for DINO-based matching fallback. 
    # If the similarity between two object features is above this threshold, they are considered to be the same object, regardless of SAM2 matching results. 
    arg_parser.add_argument("--semantic-similarity-threshold", type=float, default=0.8, help="Cosine similarity threshold for DINO-based matching fallback")
    
    # Optional arguments to change the SAM2 and DINO variants used for tracking
    arg_parser.add_argument("--sam2-variant", type=str, default="sam2.1-hiera-tiny", help="SAM2 variant to use for tracking and grounding")
    arg_parser.add_argument("--dino-variant", type=str, default="dinov2_vits14_reg", help="DINO variant to use for feature extraction")
    
    arg_parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on")

    args = arg_parser.parse_args()
    
    logger.info("###############################################################################")
    logger.info("# SG-Ego Stage 3: Consolidation of frame-level graphs into video-level graphs #")
    logger.info("###############################################################################")
    logger.info("")
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=2, sort_keys=True))

    main(args)
