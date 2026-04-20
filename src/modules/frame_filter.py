"""
modules/frame_filter.py

Scores every frame and removes bad ones before depth fusion and texture baking.

Two scores are computed per frame:
  1. Sharpness  — Laplacian variance of the greyscale RGB image.
                  Low = blurry or motion-blurred frame.
  2. Pose jump  — translation distance between consecutive poses.
                  Very large jump = ARKit glitch / tracking loss.

Frames are dropped if EITHER:
  - sharpness < sharpness_threshold  (absolute, normalised 0→1)
  - pose_jump  > pose_jump_threshold (metres, e.g. 0.5 m between frames)

Usage in pipeline.py:
    from src.modules.frame_filter import FrameFilter
    frames = FrameFilter(sharpness_threshold=0.15, pose_jump_threshold=0.4).filter(frames)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger(__name__)


def _laplacian_variance(rgb_path: Path) -> float:
    """Return Laplacian variance of a greyscale image. Higher = sharper."""
    try:
        import cv2
        img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        # Downsample to 640 wide for speed — result is scale-invariant enough
        h, w = img.shape
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)),
                             interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        pass
    # Fallback: use PIL + numpy if cv2 is missing
    try:
        from PIL import Image, ImageFilter
        img = Image.open(rgb_path).convert("L")
        if img.width > 640:
            img = img.resize((640, int(img.height * 640 / img.width)))
        arr = np.asarray(img, dtype=np.float32)
        # Simple finite-difference Laplacian
        lap = (arr[:-2, 1:-1] + arr[2:, 1:-1] +
               arr[1:-1, :-2] + arr[1:-1, 2:] -
               4 * arr[1:-1, 1:-1])
        return float(np.var(lap))
    except Exception:
        return 0.0


class FrameFilter:
    """
    Parameters
    ----------
    sharpness_threshold : float
        Normalised sharpness (0–1). Frames below this are dropped.
        0.10 = drop the blurriest 10 % of frames roughly.
        0.20 = more aggressive (safer for high frame-rate scans).
    pose_jump_threshold : float
        Maximum allowed translation (metres) between consecutive frames.
        0.4 m is a good default — larger jumps indicate ARKit tracking loss.
    workers : int
        Thread count for parallel image reads.
    """

    def __init__(
        self,
        sharpness_threshold: float = 0.12,
        pose_jump_threshold: float = 0.4,
        workers: int = 8,
    ):
        self.sharpness_threshold = sharpness_threshold
        self.pose_jump_threshold = pose_jump_threshold
        self.workers = workers

    def filter(self, frames) -> list:
        if not frames:
            return frames

        n = len(frames)
        log.info(f"FrameFilter: scoring {n} frames...")

        # ── 1. sharpness scores (parallel) ────────────────────────────────────
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            raw_vars = list(ex.map(
                lambda fr: _laplacian_variance(fr.rgb_path), frames
            ))
        raw_vars = np.array(raw_vars, dtype=np.float64)

        # Normalise to [0, 1] using the 99th-percentile as ceiling
        # so one extremely sharp frame doesn't crush everything else.
        p99 = np.percentile(raw_vars, 99) + 1e-9
        sharp_scores = np.clip(raw_vars / p99, 0.0, 1.0)

        # ── 2. pose jump scores ───────────────────────────────────────────────
        positions = np.array([fr.c2w[:3, 3] for fr in frames])  # (N, 3)
        jumps = np.zeros(n, dtype=np.float64)
        if n > 1:
            diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # (N-1,)
            # Frame i gets the maximum of the jump before and after it
            jumps[1:]   = diffs
            jumps[:-1]  = np.maximum(jumps[:-1], diffs)

        # ── 3. apply thresholds ───────────────────────────────────────────────
        sharp_ok = sharp_scores >= self.sharpness_threshold
        jump_ok  = jumps <= self.pose_jump_threshold

        keep_mask = sharp_ok & jump_ok
        n_keep = int(keep_mask.sum())

        # Safety: never drop more than 40 % of frames — if the threshold is too
        # aggressive we'd destroy the scan coverage.
        if n_keep < int(n * 0.60):
            log.warning(
                f"  FrameFilter: threshold would drop {n - n_keep}/{n} frames "
                f"(>{40}%). Relaxing sharpness threshold to keep ≥60% of frames."
            )
            # Relax: keep the top 60 % by sharpness at minimum
            cutoff = np.percentile(sharp_scores, 40.0)
            sharp_ok = sharp_scores >= cutoff
            keep_mask = sharp_ok & jump_ok
            n_keep = int(keep_mask.sum())

        filtered = [fr for fr, keep in zip(frames, keep_mask) if keep]

        n_blurry  = int((~sharp_ok).sum())
        n_jump    = int((~jump_ok).sum())
        log.info(
            f"  FrameFilter: kept {n_keep}/{n} frames  "
            f"(dropped {n_blurry} blurry, {n_jump} pose-jump)"
        )
        log.info(
            f"  Sharpness: min={sharp_scores.min():.3f}  "
            f"mean={sharp_scores.mean():.3f}  "
            f"threshold={self.sharpness_threshold:.3f}"
        )
        return filtered