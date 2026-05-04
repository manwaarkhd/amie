from typing import Tuple, Union
import numpy as np
import matplotlib
import cv2

def get_best_level_for_mpp(wsi, slide_mpp: float, target_mpp: Union[int, float]) -> Tuple[int, float, float]:
    """ Override of the abstract `get_best_level_for_mpp` method for determining the best level for a target MPP."""     
    target_downsample = target_mpp / slide_mpp
    level = wsi.slide.get_best_level_for_downsample(target_downsample)
    level_downsample = float(wsi.level_downsamples[level])
    level_mpp = slide_mpp * level_downsample
    return level, level_downsample, level_mpp

def get_thumbnail(wsi, slide_mpp: float, target_mpp: float) -> np.ndarray:
    width, height = wsi.level_dimensions[0]

    # target dimensions at the requested mpp
    target_w = int(round(width  * slide_mpp / target_mpp))
    target_h = int(round(height * slide_mpp / target_mpp))

    # best pyramid level for the requested mpp
    level, _, _ = get_best_level_for_mpp(wsi, slide_mpp, target_mpp)
    level_w, level_h = wsi.level_dimensions[level]

    with wsi:
        image = wsi.read_region((0, 0), level, (level_w, level_h), return_type="numpy")

    # the selected level rarely matches mpp exactly — resize to the precise target
    if (image.shape[1], image.shape[0]) != (target_w, target_h):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return image  # RGB np.ndarray

def visualize_heatmap(
    wsi,                     # WholeSlideImage object
    slide_mpp: float,        # MPP of the whole-slide image
    coordinates: np.ndarray, # (N, 2)  top-left (x, y) in level-0 pixels
    scores: np.ndarray,      # (N,)    one score per patch
    patch_size: int,         # patch side length in level-0 pixels
    target_mpp: float=10.0,  # MPP at which to visualize
    alpha: float=0.5,        # heatmap opacity [0 = invisible, 1 = opaque]
    cmap: str="coolwarm",
) -> np.ndarray:             # RGB uint8, same dimensions as the thumbnail
    # 1. thumbnail at target mpp
    image = get_thumbnail(wsi, slide_mpp, target_mpp)   # RGB np.ndarray
    H, W  = image.shape[:2]

    # 2. scale coordinates and patch size from level-0 to vis-mpp space
    scale     = slide_mpp / target_mpp
    vis_xy    = np.floor(coordinates * scale).astype(np.int32)  # (N, 2)
    vis_patch = max(1, int(round(patch_size * scale)))

    # 3. accumulate scores onto a canvas (handles overlapping patches by averaging)
    score_sum = np.zeros((H, W), dtype=np.float32)
    count     = np.zeros((H, W), dtype=np.float32)

    s_min, s_max = scores.min(), scores.max()
    norm_scores  = (scores - s_min) / (s_max - s_min + 1e-8)

    for (x, y), s in zip(vis_xy, norm_scores):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + vis_patch, W), min(y + vis_patch, H)
        if x1 >= x2 or y1 >= y2:
            continue
        score_sum[y1:y2, x1:x2] += s
        count    [y1:y2, x1:x2] += 1

    # 4. compute per-pixel average score; mask marks where patches exist
    covered    = count > 0
    avg_scores = np.where(covered, score_sum / np.where(covered, count, 1), 0.0)

    # 5. apply matplotlib colormap → RGB
    colormap    = matplotlib.colormaps[cmap]
    heatmap_rgb = (colormap(avg_scores)[:, :, :3] * 255).astype(np.uint8)

    # 6. alpha blend only over covered pixels, leave background untouched
    mask   = covered[:, :, np.newaxis]
    result = np.where(
        mask,
        ((1 - alpha) * image + alpha * heatmap_rgb).astype(np.uint8),
        image
    )
    return result