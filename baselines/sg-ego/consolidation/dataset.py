"""Dataset utilities for loading video windows and aligned frame graphs."""

import json
from typing import List, Optional, Tuple

import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset, get_worker_info

torch.set_grad_enabled(False)


class VideoDataset(Dataset):
    """Dataset returning temporal windows of video frames and frame graphs.

    Notes
    -----
    Each sample contains the frames for one window, the aligned per-frame graph
    dictionaries, and the absolute start and end frame indices.

    The ``ranges`` argument uses half-open intervals ``(start, end)``. When it
    is not provided, sliding windows are generated from ``window_size`` and
    ``stride``.
    """

    def __init__(
        self,
        video_path: str,
        graphs_path: str,
        window_size: int = 10,
        stride: int = 10,
        ranges: Optional[List[Tuple[int, int]]] = None,
        use_mapped_graphs: bool = False,
    ):
        """Initialize the dataset and precompute valid temporal ranges.
        
        The dataset accepts a list of window ranges to process. 
        When ranges is provided, the window_size and stride arguments are ignored.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        graphs_path : str
            Base path of the JSON file containing per-frame graph
            annotations, without extension.
        window_size : int, optional
            Number of frames included in each temporal window.
        stride : int, optional
            Step used to generate consecutive windows when ``ranges`` is not
            provided.
        ranges : list[tuple[int, int]] | None, optional
            Explicit window ranges to use instead of generating them automatically.
        use_mapped_graphs : bool, optional
            Whether to load the ``_mapped.json`` graph annotations.
        """
        super().__init__()

        self.vr = None

        self.window_size = window_size
        self.stride = stride

        self.video_path = video_path
        self.graphs_path = graphs_path

        self.num_frames = len(VideoReader(self.video_path, ctx=cpu(0)))

        if use_mapped_graphs:
            self.graphs = json.load(open(f"{graphs_path}_mapped.json", "r", encoding="utf-8"))
        else:
            self.graphs = json.load(open(f"{graphs_path}.json", "r", encoding="utf-8"))

        if ranges is None:
            self.ranges = [
                (i * self.stride, min(self.num_frames, i * self.stride + self.window_size))
                for i in range((self.num_frames - self.window_size) // self.stride + 1)
            ]
        else:
            # Keep only valid ranges
            self.ranges = ranges

    def __len__(self):
        """Return the number of temporal windows available in the dataset.

        Returns
        -------
        int
            Number of samples that can be retrieved.
        """
        return len(self.ranges)

    def _get_frames(self, idxs: list[int]):
        """Load a batch of frames for the requested indices.

        Parameters
        ----------
        idxs : list[int]
            Absolute frame indices to retrieve from the video.

        Returns
        -------
        torch.Tensor
            Tensor of frames with shape ``(T, C, H, W)``.

        Notes
        -----
        The underlying ``VideoReader`` is lazily initialized per worker to
        avoid sharing decoder state across dataloader processes.
        """
        if self.vr is None:
            worker = get_worker_info()
            self.vr = VideoReader(self.video_path, ctx=cpu(worker.id if worker else 0))

        frames = self.vr.get_batch(idxs)

        return frames.permute(0, 3, 1, 2)

    def __getitem__(self, idx: int):
        """Return one temporal window with its aligned graphs.

        Parameters
        ----------
        idx : int
            Sample index in the dataset.

        Returns
        -------
        tuple[torch.Tensor, list[dict], int, int]
            Frames in the selected window, graph annotations for each returned
            frame, and the absolute start and end indices of the requested
            interval.
        """
        start, end = self.ranges[idx]

        frames_idxs = list(
            set(range(start, end + 1)) & set(range(self.num_frames)) & set(map(int, self.graphs.keys()))
        )  # deduplicate and clamp to valid frame indices
        frames_idxs.sort()  # make sure the frames are returned in the correct order

        # window frames
        frames = self._get_frames(frames_idxs)

        # window graphs
        graphs = []
        for frame_idx in frames_idxs:
            graphs.append(self.graphs[str(frame_idx)])

        return frames, graphs, start, end
