import torch

def compress_ranges(seq):
    """Compress a sorted sequence of integers into contiguous ranges.

    Parameters
    ----------
    seq : Sequence[int]
        Sorted frame indices.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive ``(start, end)`` ranges covering the input sequence.
    """

    if not seq:
        return []

    ranges = []
    start = prev = seq[0]

    for x in seq[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x

    ranges.append((start, prev))
    return ranges

@torch.jit.script
def compute_iou_batch(
    grounding_masks: torch.Tensor,  # (G, H, W) bool
    tracking_masks: torch.Tensor,  # (T, H, W) bool
) -> torch.Tensor:
    """Compute the pairwise IoU matrix between grounding and tracking masks.

    Parameters
    ----------
    grounding_masks : torch.Tensor
        Boolean tensor with shape ``(G, H, W)`` on the target device.
    tracking_masks : torch.Tensor
        Boolean tensor with shape ``(T, H, W)`` on the target device.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(G, T)`` containing pairwise IoU values.

    Notes
    -----
    The implementation is fully vectorized and does not use Python loops over
    masks.
    """
    G = grounding_masks.shape[0]
    T = tracking_masks.shape[0]
    H, W = grounding_masks.shape[1:]

    # Compute areas
    g_areas = grounding_masks.reshape(G, -1).sum(dim=1, dtype=torch.float32)  # (G,)
    t_areas = tracking_masks.reshape(T, -1).sum(dim=1, dtype=torch.float32)  # (T,)

    # Reshape for broadcasting: (G, 1, H, W) and (1, T, H, W)
    g_masks_expanded = grounding_masks.unsqueeze(1)  # (G, 1, H, W)
    t_masks_expanded = tracking_masks.unsqueeze(0)  # (1, T, H, W)

    # Compute intersections: (G, T)
    intersections = (g_masks_expanded & t_masks_expanded).reshape(G, T, -1).sum(dim=2, dtype=torch.float32)

    # Compute union and IoU: (G, T)
    unions = g_areas.unsqueeze(1) + t_areas.unsqueeze(0) - intersections
    ious = intersections / (unions + 1e-6)

    return ious


def unbatch(x: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
    num_batches: int = batch.max().item() + 1  # type: ignore
    return [x[batch == i] for i in range(num_batches)]