"""Build frame-level scene graphs from the the output of the detector and the captions."""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

import json
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from typing import Callable, Tuple

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)


def build_mapper(text_encoder: SentenceTransformer, closed_set_path: str) -> Tuple[Callable[[list[str]], tuple[torch.Tensor, list[str]]], int]:
    """Create a mapper to map entities (objects or relations) to a closed set of entities.

    Parameters
    ----------
    text_encoder : SentenceTransformer
        Text encoder for comparing the input entities with the closed set entities (e.g., by cosine similarity)
    closed_set_path : str
        Path to the closed set of entities (e.g., objects or relations) to map to. The file should contain a json list of strings.

    Returns
    -------
    Tuple[Callable[[list[str]], tuple[torch.Tensor, list[str]]], int]
        The mapper function and the number of objects/relations in the closed set.
    """

    closed_set = json.load(open(closed_set_path, "r", encoding="utf-8"))
    closed_set = list(sorted(closed_set))
    closed_set_embeddings = text_encoder.encode(closed_set, convert_to_tensor=True, show_progress_bar=False)
    closed_set_embeddings = F.normalize(closed_set_embeddings, p=2, dim=-1)

    def mapper(objects: list[str]) -> tuple[torch.Tensor, list[str]]:
        if len(objects) == 0:
            return torch.empty(0, dtype=torch.long), []

        target_embeddings = text_encoder.encode(objects, convert_to_tensor=True, show_progress_bar=False)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

        # Find the best match in the closed set for each input.
        idxs = (target_embeddings @ closed_set_embeddings.T).argmax(-1)

        # Convert the matched indices to the corresponding closed set entities
        matches = [closed_set[i] for i in idxs.cpu().numpy()]

        logger.debug("Mapping results: %s", list(zip(objects, matches)))

        return idxs, matches

    return mapper, len(closed_set)