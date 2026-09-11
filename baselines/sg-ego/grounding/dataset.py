"""Dataset for frame-level triplet grounding."""

import json
import os
from typing import List, Optional, Tuple

import decord
from decord import VideoReader, cpu
from torch.utils.data import Dataset, get_worker_info

decord.bridge.set_bridge("torch")

import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

Frame = namedtuple("Frame", ["index", "captions", "triplets", "image", "tags"])

ROLE_NAMES = {"robot", "manipulated_object", "initial_support", "target"}


def parse_role_entity(value: str) -> tuple[str, int, Optional[str]]:
    """Parse ``role::visual_name_N`` without exposing the role to GroundingDINO."""
    value = value.strip()
    role = None
    if "::" in value:
        candidate, value = value.split("::", 1)
        candidate = candidate.strip().lower()
        if candidate not in ROLE_NAMES:
            raise ValueError(f"Unsupported task role: {candidate!r}")
        role = candidate
    parts = value.split("_")
    tag = int(parts[-1]) if parts and parts[-1].isnumeric() else -1
    label_parts = parts[:-1] if tag >= 0 else parts
    label = " ".join(label_parts).strip()
    if not label:
        raise ValueError(f"Missing visual label in entity {value!r}")
    return label, tag, role


def collate_fn(batch: List[Frame]) -> Frame:
    return Frame(
        index=[item.index for item in batch],
        captions=[item.captions for item in batch],
        triplets=[item.triplets for item in batch],
        image=[item.image for item in batch],
        tags=[item.tags for item in batch],
    )


class CaptioningDataset(Dataset):
    """
    A dataset class to load video frames and the corresponding
    frame-level annotated (subject, relation, object) triplets.
    """

    def __init__(self, video_path: str, captions_path: str, keep_frames: Optional[List[int]] = None):
        """Instantiate the CaptioningDataset class.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        captions_path : str
            Path to the captions file.
        keep_frames : Optional[List[int]], optional
            List of frame indices to include, by default None (include all frames)
        """
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        assert os.path.exists(captions_path), f"Captions file not found: {captions_path}"

        self.video_path = video_path

        # Read the input captions and triplets from the JSON file
        self.captions = json.load(open(captions_path, "r", encoding="utf-8"))
        if "parsed" in self.captions:
            self.captions = self.captions["parsed"]

        # Count the total number of frames in the video using decord VideoReader
        self.num_frames = len(VideoReader(self.video_path, ctx=cpu(0)))

        # Video reader (each worker should have its own instance to avoid conflicts)
        self.vr = None

        # If keep frames is provided, filter the frames to include only those specified
        if keep_frames is not None:
            logger.info(f"Filtering frames to include only: {keep_frames}")
            self.valid_idxs = sorted(set(range(self.num_frames)) & set(keep_frames))
        else:
            self.valid_idxs = sorted(range(self.num_frames))

        logger.info(f"Total frames in video: {self.num_frames}, total captions found: {len(self.captions)}. Processing {len(self.valid_idxs)} frames after filtering.")

    def __len__(self):
        return len(self.valid_idxs)

    def _get_captions(self, frame_idx: int) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[int, int]]]:
        """Get the captions and triplets for a given frame index.

        This function returns:
         - raw_triplets: The original triplets as read from the JSON file.
         - triplets: The parsed (subject, relation, object) triplets.
         - tags: The corresponding tags for the subject and object in the triplets.

        The raw_triplets are returned to track which captions are mapped to their corresponding objects in the scene.
        A shortcut is used to replace "camera_wearer" with "human" for grounding.

        Parameters
        ----------
        frame_idx : int
            Index of the frame in the video.

        Returns
        -------
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[int, int]]]
            The raw triplets (which may include traci), the parsed triplets, and the corresponding tags.
        """
        raw_triplets, triplets, tags = [], [], []

        # Iterate over all available captions for the given frame index
        for raw_triplet in self.captions[str(frame_idx)]['parsed']:
            
            if "parsed" in raw_triplet:
                raw_triplet = raw_triplet["parsed"]

            # Some basic checks that should always pass if the captions are well-formed
            assert len(raw_triplet) == 3, f"Each caption should be a triplet of (subject, relation, object), but got: {raw_triplet}"
            assert all(isinstance(x, str) for x in raw_triplet), f"Each element in the caption triplet should be a string, but got: {raw_triplet}"

            # Parse the subject, relation, and object by splitting on underscores and removing the last part (which is the tag)
            subj, rel, obj = raw_triplet

            # Separate the subject label from its tag (if present) and clean up the strings
            subj, subj_tag, _ = parse_role_entity(subj)

            rel = rel.replace("_", " ").strip() if "_" in rel else rel.strip()

            # Separate the object label from its tag (if present) and clean up the strings
            obj, obj_tag, _ = parse_role_entity(obj)

            # Replace camera wearer with "human" for grounding
            subj = "human" if raw_triplet[0].lower() == "camera_wearer" else subj
            obj = "human" if raw_triplet[2].lower() == "camera_wearer" else obj

            # There may some weird relations like 'on, on, floor_1' -> we manually filter them and take only the first relation (i.e., 'on')
            # Ideally, captions here should be cleaned up to avoid such cases, but we brace for the worst and handle possible mistakes here
            rel = rel.split(",")[0].strip() if "," in rel else rel

            # Extract the tracking tags for the subject and object (if present) to disambiguate between multiple instances of the same object in the scene
            raw_triplets.append(raw_triplet)
            triplets.append((subj, rel, obj))
            tags.append((subj_tag, obj_tag))

        return raw_triplets, triplets, tags

    def _get_frame(self, idx: int):
        if self.vr is None:
            worker = get_worker_info()
            self.vr = VideoReader(self.video_path, ctx=cpu(worker.id if worker else 0))

        return self.vr[idx].permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Frame:
        """Get the idx-th frame and its corresponding captions and triplets.

        Parameters
        ----------
        idx : int
            Index of the frame to retrieve.

        Returns
        -------
        Frame
            The frame and its associated captions and triplets.
        """

        # Convert the relative index to the actual frame index in the video
        idx = self.valid_idxs[idx]

        # Get the frame and the corresponding captions and triplets
        frame = self._get_frame(idx)
        raw_captions, triplets, tags = self._get_captions(idx)

        return Frame(index=int(idx), captions=raw_captions, triplets=triplets, image=frame, tags=tags)
