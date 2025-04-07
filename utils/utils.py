import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def tq2RT(pose_tq, depth_cutoff=1):
    """
    Converts a pose represented by translation and quaternion to a 4x4 transformation matrix.

    Args:
        pose_tq (np.array): An array of shape (7,) where the first 3 elements are the translation (t)
                            and the last 4 elements are the quaternion (q).
        depth_cuttoff (float, optional): used to normalize the translation vector. Defaults to 1.

    Returns:
        np.array: A 4x4 transformation matrix.
    """
    t = pose_tq[:3] / depth_cutoff  # Translation vector (3,)
    q = pose_tq[3:]  # Quaternion (4,)

    # Convert quaternion to rotation matrix
    r = R.from_quat(q)  # Create a Rotation object from the quaternion
    R_matrix = r.as_matrix()  # Get the 3x3 rotation matrix

    # Form the 4x4 transformation matrix
    RT = np.eye(4, dtype=np.float32)  # Initialize as identity matrix
    RT[:3, :3] = R_matrix  # Set rotation part
    RT[:3, 3] = t  # Set translation part

    return RT
