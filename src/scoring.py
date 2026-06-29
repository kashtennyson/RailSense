import numpy as np
from . import config



def compute_error_maps(targets, reconstructions):
    """
    Per-pixel reconstruction error, averaged over the color channels.

    Accepts a single image (H, W, C) or a batch (N, H, W, C). Inputs may be
    TensorFlow tensors or NumPy arrays. Returns (H, W) or (N, H, W).
    """
    targets = np.asarray(targets, dtype=np.float32)
    reconstructions = np.asarray(reconstructions, dtype=np.float32)
    squared_error = np.square(targets - reconstructions)
    return np.mean(squared_error, axis=-1)



def score_from_error_maps(error_maps, method=None, topk_percent=None):
    """
    Reduces per-pixel error maps to a single anomaly score per image.

    method:
        "global_mse" - mean error over every pixel (the original behaviour).
        "topk"       - mean of only the highest-error pixels, concentrating
                       the score on the worst-reconstructed region so a small
                       localized defect is not averaged away by the background.
    """
    method = method or config.SCORE_METHOD
    topk_percent = topk_percent if topk_percent is not None else config.TOPK_PERCENT

    single = error_maps.ndim == 2
    if single:
        error_maps = error_maps[None, ...]

    n_images = error_maps.shape[0]
    flat = error_maps.reshape(n_images, -1)

    if method == "global_mse":
        scores = flat.mean(axis=1)

    elif method == "topk":
        n_pixels = flat.shape[1]
        k = max(1, int(round(n_pixels * topk_percent)))
        topk_idx = np.argpartition(flat, -k, axis=1)[:, -k:]
        topk_vals = np.take_along_axis(flat, topk_idx, axis=1)
        scores = topk_vals.mean(axis=1)

    else:
        raise ValueError(
            f"Unknown SCORE_METHOD '{method}'. Expected 'global_mse' or 'topk'."
        )

    return scores[0] if single else scores



def anomaly_scores(targets, reconstructions, method=None, topk_percent=None):
    """
    Convenience wrapper: error maps -> scalar score(s) in one call.
    """
    error_maps = compute_error_maps(targets, reconstructions)
    return score_from_error_maps(error_maps, method=method, topk_percent=topk_percent)
