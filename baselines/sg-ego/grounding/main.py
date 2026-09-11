"""Entry point for building frame-level scene graphs"""

import os
import os.path as osp

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import json
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from argparse import ArgumentParser

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import List, Tuple, Optional

from utils.logging import configure_logging

from grounding.dataset import CaptioningDataset, collate_fn
from grounding.build import build_video_scene_graphs
from grounding.detector import GroundingDINO
from grounding.utils.mapping import build_mapper

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

CAMERA_WEARER_LABEL = "##camera_wearer##"

def read_job_file(job_path: str) -> Tuple[List[str], List[Optional[List[int]]]]:
    
    assert os.path.exists(job_path), f"Job path {job_path} does not exist"

    video_ids, video_frames = [], []
    with open(job_path, "r") as f:
        for line in f:
            if "," in line:
                video_id, *frames = line.strip().split(",")
                video_ids.append(video_id)
                video_frames.append([int(f) for f in frames])
            else:
                video_ids.append(line.strip())
                video_frames.append(None)

    return video_ids, video_frames


def main(args):

    videos_path = os.path.join(args.root, "videos")
    captions_path = os.path.join(args.root, "captions", args.captions_version)
    output_path = os.path.join(args.root, "frame_graphs")
    os.makedirs(os.path.join(output_path, args.frame_graphs_version), exist_ok=True)

    logger.info("")
    logger.info(f"Videos path: {videos_path}.")
    logger.info(f"Captions path: {captions_path}.")
    logger.info(f"Output path: {output_path}.")
    logger.info("")

    if args.input_file:
        video_id = os.path.basename(args.input_file)
        video_ids, video_frames = [video_id], [None]
    else:  # args.job_file
        video_ids, video_frames = read_job_file(args.job_file)
    
    # Loading the GroundingDINO detector
    detector = GroundingDINO(resolution=args.det_resolution, box_threshold=args.det_threshold, autocast=True)
    st_mapper = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").eval()

    objects_mapper, num_objects = build_mapper(st_mapper, args.objects_path)
    relations_mapper, num_relations = build_mapper(st_mapper, args.relations_path)
    logger.info(f"Loaded our closed sets with {num_objects} objects and {num_relations} relations respectively.")

    for video_index, (video_id, frames) in tqdm(enumerate(zip(video_ids, video_frames)), desc="Processing videos", total=len(video_ids)):
        logger.info(f"Processing video {video_id} [{video_index}/{len(video_ids)}]...")

        try:

            video_captions_path = osp.join(captions_path, osp.splitext(video_id)[0] + ".json")
            assert os.path.exists(video_captions_path), f"Caption file {video_captions_path} does not exist for video {video_id}"

            video_output_path = osp.join(output_path, args.frame_graphs_version, osp.basename(osp.splitext(video_id)[0]))

            if osp.exists(f"{video_output_path}.json") and not args.overwrite:
                print(f"Skipping {video_output_path}")
                continue

            # Loading the frames dataset
            dataset = CaptioningDataset(osp.join(videos_path, video_id), video_captions_path, keep_frames=frames)
            video_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

            build_video_scene_graphs(
                video_dataloader,
                detector,
                objects_mapper=objects_mapper,
                relations_mapper=relations_mapper,
                output_path=video_output_path,
                det_batch_size=args.det_batch_size,
                overlap_iou_threshold=args.overlap_iou_threshold,
                use_spatial_heuristics=args.use_spatial_heuristics,
                double_detection_threshold=args.double_detection_threshold,
            )

        except Exception as e:
            logger.error(f"ERROR: processing video {video_id}: {e}")
            continue


if __name__ == "__main__":
    configure_logging("grounding")

    arg_parser = ArgumentParser(description="SG-Ego Stage 2: Frame-level relations grounding")
    
    # Dataset root path
    arg_parser.add_argument("--root", type=str, required=True, help="Root path for the dataset")

    # Input can be either:
    # - a video file, in which case we process the entire video, or
    # - a txt file containing the list of videos to process, each with the corresponding list of target frames to ground
    input_selection_group = arg_parser.add_mutually_exclusive_group(required=True)
    input_selection_group.add_argument("--input-file", type=str, default=None, help="Path to the input mp4 file", required=False)
    input_selection_group.add_argument("--job-file", type=str, default=None, help="Path to the job file", required=False)

    # Output parameters
    arg_parser.add_argument("--captions-version", type=str, required=True, help="Version of the captions to use")
    arg_parser.add_argument("--frame-graphs-version", type=str, required=True, help="Version of the graph to save")

    # Batching parameters
    arg_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing frames")
    arg_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for data loading")

    # Detectors parameters
    arg_parser.add_argument("--det-model", type=str, default="IDEA-Research/grounding-dino-base", help="Object detector to use (default: DINO)")
    arg_parser.add_argument("--det-resolution", type=int, default=320, help="Image resolution for Grounding-DINO inference (shortest edge)")
    arg_parser.add_argument("--det-threshold", type=float, default=0.1, help="Detection confidence threshold for Grounding-DINO")
    arg_parser.add_argument("--det-batch-size", type=int, default=8, help="Batch size for Grounding-DINO inference (set to 1 for no batching)")

    # Mapping for closed set projection
    arg_parser.add_argument("--objects-path", type=str, default="annotations/objects.json", help="Path to the objects mapping file")
    arg_parser.add_argument("--relations-path", type=str, default="annotations/relations.json", help="Path to the relations mapping file")

    # Build graph parameters
    arg_parser.add_argument("--overlap-iou-threshold", type=float, default=0.3, help="IoU threshold for matching detected boxes with the same label when building the graph")
    arg_parser.add_argument("--use-spatial-heuristics", action="store_true", default=False, help="Whether to use spatial heuristics for filtering candidate triplets")
    arg_parser.add_argument("--double-detection-threshold", type=float, default=0.8, help="IoU overlap threshold for double detections")

    arg_parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to run in debug mode (process only the first video and save intermediate results)")

    args = arg_parser.parse_args()

    logger.info("###################################################")
    logger.info("# SG-Ego Stage 2: Frame-level relations grounding #")
    logger.info("###################################################")
    logger.info("")
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, sort_keys=True))

    main(args)
