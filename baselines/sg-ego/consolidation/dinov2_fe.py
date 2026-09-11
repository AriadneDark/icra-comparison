"""Visual feature extractors based on DINOv2 backbones.

This module provides a base feature extractor interface together with a
DINOv2-backed implementation that produces ROI-aligned node features from
image frames and bounding boxes.
"""

# Same code as sgrl/models/fe/dinov2_fe.py

from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from torchvision.ops import roi_align


class VisualFE(torch.nn.Module):
    """Base interface for visual feature extractors.

    Subclasses are expected to implement :meth:`extract_features` and return a
    spatial feature map that can be pooled over regions of interest.
    """

    def __init__(self):
        """Initialize the feature extractor module."""
        super(VisualFE, self).__init__()

    def extract_features(self, x):
        """Extract a spatial feature map from a batch of frames.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of frames with shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map tensor with shape ``(B, C', H', W')``.

        Raises
        ------
        NotImplementedError
            Raised when a subclass does not override this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, frames: torch.Tensor, boxes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node-level visual features from frame regions.

        Parameters
        ----------
        frames : torch.Tensor
            Input image batch with shape ``(B, C, H, W)``.
        boxes : torch.Tensor
            Bounding boxes with shape ``(num_nodes, 4)`` in pixel coordinates,
            ordered as ``(x1, y1, x2, y2)``.
        batch : torch.Tensor
            Batch indices mapping each box to the corresponding frame in
            ``frames``.

        Returns
        -------
        torch.Tensor
            ROI-pooled feature vectors with shape ``(num_nodes, C')``.

        Notes
        -----
        Bounding boxes are normalized against the input image resolution,
        scaled to the feature map resolution, and pooled with
        :func:`torchvision.ops.roi_align`.
        """
        assert frames.dim() == 4, "Input frames should be a 4D tensor (B, C, H, W)"
        assert len(boxes.shape) == 2, "Expected boxes to be a 2D tensor (num_nodes, 4)"
        assert boxes.shape[1] == 4, "Expected boxes to have shape (num_nodes, 4)"
        assert boxes.shape[0] == batch.shape[0], "Number of boxes should match the number of nodes in the batch"

        _, _, image_height, image_width = frames.shape

        # Normalize bounding boxes to [0, 1] relative to image dimensions
        boxes[:, 0] /= image_width  # x1
        boxes[:, 1] /= image_height  # y1
        boxes[:, 2] /= image_width  # x2
        boxes[:, 3] /= image_height  # y2

        feat_map = self.extract_features(frames)

        # Compute the relative scale between the feature map and the original image size
        _, _, feat_height, feat_width = feat_map.shape
        scale_h = feat_height / image_height
        scale_w = feat_width / image_width

        # Rescale the bounding boxes to the feature map size
        rois_scaled = boxes.clone().to(frames.device)

        assert rois_scaled.max() <= 1.0, "Node boxes should be normalized to [0,1]"
        rois_scaled[:, 0] *= image_width * scale_w
        rois_scaled[:, 1] *= image_height * scale_h
        rois_scaled[:, 2] *= image_width * scale_w
        rois_scaled[:, 3] *= image_height * scale_h

        assert rois_scaled[:, 0].max() <= feat_width and rois_scaled[:, 1].max() <= feat_height, "Scaled boxes exceed feature map dimensions"
        assert rois_scaled[:, 2].max() <= feat_width and rois_scaled[:, 3].max() <= feat_height, "Scaled boxes exceed feature map dimensions"

        # Convert bounding boxes to the format expected by roi_align: (batch_index, x1, y1, x2, y2)
        # Refer to https://docs.pytorch.org/vision/main/generated/torchvision.ops.roi_align.html#torchvision.ops.roi_align
        rois = torch.cat([batch.unsqueeze(1), rois_scaled], dim=1)  # (num_nodes, 5)
        roi_feat = roi_align(feat_map, rois, output_size=(7, 7))  # (num_nodes, C, 7,7)  # type: ignore

        # The output of roi_align is (num_nodes, C, 7, 7).
        # We can flatten the spatial dimensions to get a feature vector for each node.
        roi_feat = roi_feat.mean(dim=[2, 3])

        assert roi_feat.shape[0] == boxes.shape[0], "Number of ROI features should match the number of nodes"

        return roi_feat.view(roi_feat.size(0), -1)


class DinoV2FE(VisualFE):
    """Visual feature extractor backed by a pretrained DINOv2 model."""

    def __init__(self, variant="dinov2_vits14_reg"):
        """Load a pretrained DINOv2 backbone and input transforms.

        Parameters
        ----------
        variant : str, optional
            Torch Hub identifier of the DINOv2 backbone to load.
        """
        super(DinoV2FE, self).__init__()

        # Build the spatial feature extractor using a pretrained ResNet-50 model
        self.backbone: Any = torch.hub.load("facebookresearch/dinov2", variant)

        # Take the patch size from the backbone's patch embedding layer
        self.patch_size = int(
            self.backbone.patch_embed.patch_size if isinstance(self.backbone.patch_embed.patch_size, int) else self.backbone.patch_embed.patch_size[0]
        )

        self.transforms = TF.Compose(
            [
                TF.Lambda(lambda img: img / 255.0),  # Scale to [0,1]
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial feature maps with the DINOv2 backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input batch with shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Spatial feature maps with shape ``(B, C, H', W')``, where the
            spatial resolution depends on the patch size.
        """
        x = self.transforms(x)  # Normalize the input frames

        x = self.resize_and_pad(x, target=518, patch_size=self.patch_size)  # Resize and pad to ensure dimensions are multiples of patch size

        B, _, H, W = x.shape

        # Get patch tokens (no cls token)
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N, C)

        # Recover spatial grid
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        feat_map = patch_tokens.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)
        return feat_map  # (B, C, H', W')

    def resize_and_pad(self, frames: torch.Tensor, target: int = 518, patch_size: int = 14) -> torch.Tensor:
        """Resize frames while preserving aspect ratio, then pad to patch size.

        Parameters
        ----------
        frames : torch.Tensor
            Input batch of frames with shape ``(..., H, W)``.
        target : int, optional
            Target size for the shortest image side after resizing.
        patch_size : int, optional
            Patch size used by the backbone.

        Returns
        -------
        torch.Tensor
            Resized and padded frames whose spatial dimensions are multiples of
            ``patch_size``.
        """
        # resize longest side, keep aspect ratio
        *_, H, W = frames.shape

        scale = target / min(H, W)
        new_h, new_w = int(H * scale), int(W * scale)

        frames = torch.nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

        return self.pad_to_patch_size(frames, patch_size)

    def pad_to_patch_size(self, frames: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
        """Pad frames so height and width are divisible by the patch size.

        Parameters
        ----------
        frames : torch.Tensor
            Input batch of frames with shape ``(..., H, W)``.
        patch_size : int, optional
            Divisor required by the patch embedding layer.

        Returns
        -------
        torch.Tensor
            Zero-padded frames with spatial dimensions divisible by
            ``patch_size``.
        """
        # img: (C, H, W)
        *_, H, W = frames.shape

        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        frames = F.pad(frames, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        return frames
