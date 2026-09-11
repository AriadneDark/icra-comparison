#!/usr/bin/env python3
"""Convert frame and video graph JSON directories into parquet shards and upload them to Hugging Face."""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

SHARD_SIZE = 100_000


def convert_frame_graphs_to_parquet(input_dir: str, output_dir: str):
    """Convert the frame graphs to parquet format and save them in the output directory.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing frame graph JSON files.
    output_dir : str
        Path to the output directory where parquet files will be saved.
    """

    def generate_data_rows():
        for video_file in Path(input_dir).glob("*_mapped.json"):
            video_id = video_file.stem

            with open(video_file, "r") as f:
                graphs = json.load(f)

            for frame_id, graph in graphs.items():
                yield {
                    "video_id": video_id.replace("_mapped", ""),
                    "frame_id": int(frame_id),
                    **graph,
                }

    shard_idx, batch = 0, []

    for row in tqdm(generate_data_rows(), desc="Processing frame graphs..."):
        batch.append(row)

        if len(batch) >= SHARD_SIZE:
            pa.Table.from_pylist(batch).to_pandas().to_parquet(f"{output_dir}/frame_graphs/{shard_idx:05d}.parquet")

            shard_idx += 1
            batch.clear()

    if batch:
        # Save the remaining rows in the last shard
        pa.Table.from_pylist(batch).to_pandas().to_parquet(f"{output_dir}/frame_graphs/{shard_idx:05d}.parquet")


def convert_video_graphs_to_parquet(input_dir: str, output_dir: str):
    """Convert the video graphs to parquet format and save them in the output directory.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing video graph JSON files.
    output_dir : str
        Path to the output directory where parquet files will be saved.
    """

    def generate_rows():
        for video_file in Path(input_dir).glob("*_mapped.json"):
            video_id = video_file.stem

            with open(video_file, "r") as f:
                video_graphs = json.load(f)

            for frame_id, graph in video_graphs.items():
                start, end = frame_id.split("-")
                start, end = (int(start), int(end))

                if "history" in graph:
                    # Update the schema for the history field to be a list of lists of dictionaries
                    graph["history"] = [[{"frame_idx": frame_idx, "obj_idx": obj_idx, "label": label} for (frame_idx, obj_idx, label) in track] for track in graph["history"]]
                else:
                    continue

                yield {
                    "video_id": video_id.replace("_mapped", ""),
                    "start_frame": start,
                    "end_frame": end,
                    **graph,
                }

    shard_idx, batch = 0, []

    for row in tqdm(generate_rows(), desc="Processing video graphs..."):
        batch.append(row)

        if len(batch) >= SHARD_SIZE:
            pa.Table.from_pylist(batch).to_pandas().to_parquet(f"{output_dir}/video_graphs/{shard_idx:05d}.parquet")

            shard_idx += 1
            batch.clear()

    if batch:
        # Save the remaining rows in the last shard
        pa.Table.from_pylist(batch).to_pandas().to_parquet(f"{output_dir}/video_graphs/{shard_idx:05d}.parquet")


def main():

    arg_parser = argparse.ArgumentParser(description="Convert frame and video graph JSON directories into a Hugging Face Dataset.")
    
    arg_parser.add_argument("--frame-graphs-dir", type=str, required=True, help="Path to the frame graphs directory.")
    arg_parser.add_argument("--video-graphs-dir", type=str, required=True, help="Path to the video graphs directory.")
    arg_parser.add_argument("--output-dir", type=str, default="out/", help="Path to the output directory for parquet shards.")
    
    arg_parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to Hugging Face Hub.")
    arg_parser.add_argument("--shards-count", type=int, default=100, help="Number of shards to create for each dataset.")
    arg_parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for authentication.")
    arg_parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face dataset repository.")

    args = arg_parser.parse_args()

    os.makedirs(f"tmp/frame_graphs", exist_ok=True)
    os.makedirs(f"tmp/video_graphs", exist_ok=True)
    os.makedirs("sg-ego", exist_ok=True)

    convert_frame_graphs_to_parquet(args.frame_graphs_dir, "tmp")
    convert_video_graphs_to_parquet(args.video_graphs_dir, "tmp")

    dset = DatasetDict(
        {
            "frame_graphs": load_dataset(f"tmp/frame_graphs")["train"],
            "video_graphs": load_dataset(f"tmp/video_graphs")["train"],
        }
    )

    dset.save_to_disk("sg-ego", num_shards={"frame_graphs": args.shards_count, "video_graphs": args.shards_count})
    
    if args.push_to_hub:
        for name, dataset in dset.items():
            dataset.push_to_hub(args.hf_repo, private=True, config_name=name, token=args.hf_token)  # type: ignore


if __name__ == "__main__":
    main()
