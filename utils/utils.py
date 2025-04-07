import cv2
import numpy as np


def mask_specularities(img, mask=None, spec_thr=0.96):
    """
    mask sepcularities. Counted as specularities if one pixel is fully white.
    The mask is used to remove specularities from the image.

    Args:
        img (np.array, (H, W, C)): the image used to find the specularities.
        mask (np.array, (H, W), optional): an existing mask, may for surgical tools. The mask for specularities will be added to it.
        spec_thr (float, optional): Threshold. Defaults to 0.96.

    Returns:
        np.array, (H, W): The new mask.
    """
    # img in the shape of (H, W, C), mask in the shape of (H, W)
    spec_mask = img.sum(axis=-1) < (3 * 255 * spec_thr)
    mask = mask & spec_mask if mask is not None else spec_mask
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((11, 11)))
    return mask
