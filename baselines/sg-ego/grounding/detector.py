"""
GroundingDINO wrapper for object detection given a list of triplets (subj, rel, obj) as input.
"""

from typing import List, Tuple

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDINO:
    """Wrapper for GroundingDINO for object detection given a list of triplets (subj, rel, obj) as input."""

    def __init__(
        self,
        variant: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.1,
        resolution: int = 800,
        autocast: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Initialize the GroundingDINO detector.

        Parameters
        ----------
        variant : str, optional
            The variant of the GroundingDINO model to use, by default "IDEA-Research/grounding-dino-base"
        box_threshold : float, optional
            The threshold for bounding box scores, by default 0.1
        resolution : int, optional
            The resolution for image processing, by default 800
        autocast : bool, optional
            Whether to use automatic mixed precision, by default False
        device : torch.device, optional
            The device to run the model on, by default torch.device("cuda")
        """

        self.processor = AutoProcessor.from_pretrained(variant)
        self.processor.image_processor.size["shortest_edge"] = resolution

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(variant).to(device)

        self.box_threshold = box_threshold
        self.autocast = autocast

    def run(self, images: List[torch.Tensor], captions: List[Tuple[str, str, str]]) -> list[dict]:
        """Run the GroundingDINO model on a list of images and captions (triplets).

        Parameters
        ----------
        images : List[torch.Tensor]
            List of images as torch tensors.
        captions : List[Tuple[str, str, str]]
            List of caption triplets (subject, relation, object).

        Returns
        -------
        list[dict]
            List of detection results for each image and caption triplet.
        """

        # Flatten the triplets into captions like "the {subj} {rel} the {obj}." to feed into the model for grounding
        # We empirically observe better grounding performance when adding "the" before subjects and objects
        flat_captions = [f"the {subj} {rel} the {obj}." for subj, rel, obj in captions]

        # We run the tokenizer separately to get the token offsets for each caption
        text_encodings = self.processor.tokenizer(flat_captions, return_offsets_mapping=True, return_tensors="pt", padding="max_length", max_length=128).to(self.model.device)

        # Stack the images into a single tensor and prepare the inputs for the model
        stacked_images: torch.Tensor = torch.stack(images).to(self.model.device)
        inputs = self.processor(images=stacked_images, text=flat_captions, padding="max_length", max_length=128, return_tensors="pt").to(self.model.device)
        assert torch.allclose(inputs["input_ids"], text_encodings["input_ids"])

        # Run the model (with autocast if specified) and post-process the outputs to get the detection results
        with torch.autocast("cuda", enabled=self.autocast):
            outputs = self.model(**inputs)
            
        # Post-process the outputs to get the detection results for each image and caption triplet
        return self._post_process_detections(
            outputs,
            captions,
            flat_captions,
            text_encodings,
            box_threshold=self.box_threshold,
        )

    def _post_process_detections(
        self,
        outputs,
        parts,
        grounding_captions,
        text_encodings,
        box_threshold: float = 0.2,
    ) -> list[dict]:
        """Post-process the outputs of the GroundingDINO model to extract detection results.
        
        We follow the approach described here: https://github.com/IDEA-Research/GroundingDINO/blob/main/demo/inference_on_a_image.py
        to extract the bounding boxes and scores for the subject and object in each caption triplet using the
        predictions matching each part of the grounded sentence.

        Parameters
        ----------
        outputs
            Raw grounding DINO model outputs.
        parts
            Triplets to be grounded
        grounding_captions
            List of grounding captions corresponding to the triplets.
        text_encodings
            Tokenized text encodings of the grounding captions (required for computing the token offsets for the subject and object).
        box_threshold : float, optional
            Box detection threshold, by default 0.2

        Returns
        -------
        list[dict]
            List of detection results for each image and caption triplet.
        """

        # Convert the boxes to corners format and apply sigmoid to the logits to get the scores
        batch_pred_bboxes = self._convert_cxcy_to_xyxy(outputs.pred_boxes)
        batch_pred_probs = torch.sigmoid(outputs.logits)

        batch_pred_bboxes = torch.clamp(batch_pred_bboxes, 0, 1)  # Ensure the boxes are within [0, 1]

        results = []

        for batch_idx, ((subj, _, obj), grounding_caption, pred_scores, pred_bboxes) in enumerate(zip(parts, grounding_captions, batch_pred_probs, batch_pred_bboxes)):

            result = {"scores": [], "boxes": [], "text_labels": [], "labels": [], "type": []}

            try:

                # Find the token span corresponding to the subject in the grounding caption
                tok_start = text_encodings.char_to_token(batch_idx, grounding_caption.find("the " + subj))
                tok_end = text_encodings.char_to_token(batch_idx, grounding_caption.find("the " + subj) + len("the " + subj) - 1)
                subj_scores = pred_scores[:, tok_start : tok_end + 1].max(1).values  # (num_queries, )

                # Find the token span corresponding to the object in the grounding caption
                tok_start = text_encodings.char_to_token(batch_idx, grounding_caption.rfind("the " + obj))
                tok_end = text_encodings.char_to_token(batch_idx, grounding_caption.rfind("the " + obj) + len("the " + obj) - 1)
                obj_scores = pred_scores[:, tok_start : tok_end + 1].max(1).values  # (num_queries, )

                # Keep the predictions for the subject that have scores above the box threshold and are higher than the object scores
                keep = (subj_scores > box_threshold) & (subj_scores > obj_scores)
                result["scores"].extend(subj_scores[keep].cpu().tolist())
                result["boxes"].extend(pred_bboxes[keep].cpu().tolist())
                result["text_labels"].extend([subj] * int(keep.sum().item()))
                result["labels"].extend([subj] * int(keep.sum().item()))
                result["type"].extend(["subject"] * int(keep.sum().item()))

                # Keep the predictions for the object that have scores above the box threshold and are higher than the subject scores
                keep = (obj_scores > box_threshold) & (obj_scores > subj_scores)
                result["scores"].extend(obj_scores[keep].cpu().tolist())
                result["boxes"].extend(pred_bboxes[keep].cpu().tolist())
                result["text_labels"].extend([obj] * int(keep.sum().item()))
                result["labels"].extend([obj] * int(keep.sum().item()))
                result["type"].extend(["object"] * int(keep.sum().item()))

                results.append(result)
            except Exception:
                print("Error processing grounding caption:", grounding_caption)
                results.append({"scores": [], "boxes": [], "text_labels": [], "labels": [], "type": []})

        return results

    def _convert_cxcy_to_xyxy(self, bboxes_center: torch.Tensor) -> torch.Tensor:
        """Convert the bounding boxes from the center format (center_x, center_y, width, height) to the corner format (x_min, y_min, x_max, y_max).

        Parameters
        ----------
        bboxes_center : torch.Tensor
            Bounding boxes in center format with shape (num_boxes, 4).

        Returns
        -------
        torch.Tensor
            Bounding boxes in corner format with shape (num_boxes, 4).
        """
        center_x, center_y, width, height = bboxes_center.unbind(-1)
        x1, y1 = center_x - 0.5 * width, center_y - 0.5 * height
        x2, y2 = center_x + 0.5 * width, center_y + 0.5 * height
        return torch.stack([x1, y1, x2, y2], dim=-1)