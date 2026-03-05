from PIL import Image
import numpy as np
import cv2

def is_valid_patch(patch, threshold: float=0.1):
    if isinstance(patch, Image.Image):
        patch = np.array(patch)
    total_pixels = patch.shape[0] * patch.shape[1]

    # compute tissue fraction
    tissue_mask = get_tissue_mask(patch)
    tissue_fraction = np.sum(tissue_mask > 0) / total_pixels

    # assess validitiy
    if tissue_fraction > threshold:
        if (tissue_fraction < 0.15) and (is_bkgnd_black(patch, threshold=0.5) or is_bkgnd_white(patch, threshold=0.8)):
            return False
        else:
            return True
    else:
        return False
    
def get_tissue_mask(patch: np.ndarray):
    # Convert to HSV (better separation of background)
    image_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Background: high value (v), low saturation (s)
    mask_s = image_hsv[:, :, 1] > 20   # saturation mask
    mask_v = image_hsv[:, :, 2] < 230  # value mask

    # Combine masks
    tissue_mask = np.logical_and(mask_s, mask_v).astype(np.uint8) * 255

    return tissue_mask

def is_bkgnd_black(patch, threshold: float=0.3):
    mask = np.all(patch < 50, axis=-1)
    fraction = np.sum(mask) / mask.size
    return fraction > threshold

def is_bkgnd_white(patch, threshold: float=0.8):
    mask = np.all(patch > 220, axis=-1)
    fraction = np.sum(mask) / mask.size
    return fraction > threshold