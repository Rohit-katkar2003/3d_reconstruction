"""
modules/pose_smoother.py
Smooths ARKit/camera poses to remove jitter before depth fusion.

Strategy:
  - Translation: sliding-window average (robust, fast)
  - Rotation:    SLERP-based spherical average via quaternion mean

Averaging raw 4×4 matrices directly is mathematically wrong for rotations
(the averaged matrix may not be orthogonal). We decompose → average
translation and rotation separately → recompose.
"""

import logging
from typing import List

import numpy as np

log = logging.getLogger(__name__)


# ─── quaternion helpers ───────────────────────────────────────────────────────

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        w  = 0.25 / s
        x  = (R[2, 1] - R[1, 2]) * s
        y  = (R[0, 2] - R[2, 0]) * s
        z  = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w  = (R[2, 1] - R[1, 2]) / s
        x  = 0.25 * s
        y  = (R[0, 1] + R[1, 0]) / s
        z  = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w  = (R[0, 2] - R[2, 0]) / s
        x  = (R[0, 1] + R[1, 0]) / s
        y  = 0.25 * s
        z  = (R[1, 2] + R[2, 1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w  = (R[1, 0] - R[0, 1]) / s
        x  = (R[0, 2] + R[2, 0]) / s
        y  = (R[1, 2] + R[2, 1]) / s
        z  = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """
    Approximate mean of N unit quaternions via eigen-decomposition of Q^T Q.
    (Markley et al., 2007 — accurate for small rotational spread.)
    Ensures consistent hemisphere before averaging.
    """
    Q = np.array(quats, dtype=np.float64)
    # Flip quaternions that are in the opposite hemisphere to Q[0]
    ref = Q[0]
    for i in range(1, len(Q)):
        if np.dot(Q[i], ref) < 0:
            Q[i] = -Q[i]
    M   = Q.T @ Q                       # 4×4 accumulator
    _, vecs = np.linalg.eigh(M)         # eigenvalues ascending
    mean_q  = vecs[:, -1]               # largest eigenvector
    return mean_q / np.linalg.norm(mean_q)


# ─── PoseSmoother ─────────────────────────────────────────────────────────────

class PoseSmoother:
    """
    Smooth a list of Frame objects' c2w matrices in-place.

    Parameters
    ----------
    window : half-window size (each pose is averaged with ±window neighbours)
    """

    def __init__(self, window: int = 5):
        self.window = window

    def smooth(self, frames) -> list:
        """Return the same Frame list with smoothed c2w matrices."""
        n       = len(frames)
        c2ws    = [fr.c2w.copy() for fr in frames]
        quats   = [_rot_to_quat(c2w[:3, :3]) for c2w in c2ws]
        trans   = np.array([c2w[:3, 3] for c2w in c2ws])   # (N, 3)

        smoothed_c2w = []
        for i in range(n):
            lo = max(0, i - self.window)
            hi = min(n, i + self.window + 1)

            # Translation: simple mean
            t_mean = trans[lo:hi].mean(axis=0)

            # Rotation: quaternion mean
            q_mean = _mean_quaternion(quats[lo:hi])
            R_mean = _quat_to_rot(q_mean)

            new_c2w = np.eye(4, dtype=np.float64)
            new_c2w[:3, :3] = R_mean
            new_c2w[:3,  3] = t_mean
            smoothed_c2w.append(new_c2w)

        for fr, c2w in zip(frames, smoothed_c2w):
            fr.c2w = c2w

        log.info(
            f"PoseSmoother: smoothed {n} poses "
            f"(window=±{self.window}, ~{2*self.window+1} frames averaged)"
        )
        return frames