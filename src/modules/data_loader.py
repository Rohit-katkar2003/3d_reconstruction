"""
modules/data_loader.py
Loads RGB images, EXR depth maps, and pose files.
Returns a list of Frame dataclasses — the common currency of the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─── Frame dataclass ──────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int

    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx,    0, self.cx],
            [   0,    self.fy, self.cy],
            [   0,       0,      1  ],
        ], dtype=np.float64)


@dataclass
class Frame:
    idx:        int
    rgb_path:   Path
    depth_path: Path
    c2w:        np.ndarray          # 4×4 camera-to-world transform
    intrinsics: CameraIntrinsics
    depth_scale: float = 1.0       # metres per raw unit
    depth_res:  Optional[Tuple[int,int]] = None  # (W, H) of depth image


# ─── Parser helpers ───────────────────────────────────────────────────────────

def _parse_pose_file(pose_path: Path) -> List[dict]:
    """
    Flexible pose file parser.

    Supports two layouts:
    1. Your current format (one frame per block):
        # Frame <N>
        # extrinsics (camera-to-world)
        r00 r01 r02 tx
        r10 r11 r12 ty
        r20 r21 r22 tz
        0   0   0   1
        # intrinsics
        fx=... fy=... cx=... cy=... w=... h=...
        # depth_metadata (optional)
        depth_scale=... depth_resolution=WxH

    2. Plain matrix list: 16 numbers per line (row-major 4×4) with defaults.
    """
    text   = pose_path.read_text()
    blocks = re.split(r"#\s*Frame\s+\d+", text)
    if len(blocks) <= 1:
        return _parse_plain_matrices(text)

    frames_data = []
    for block in blocks[1:]:           # first element is text before "# Frame 0"
        fd = _parse_block(block)
        if fd:
            frames_data.append(fd)
    return frames_data


def _parse_block(block: str) -> Optional[dict]:
    lines = [l.strip() for l in block.strip().splitlines() if l.strip()]

    # Extract 4×4 matrix rows (lines that contain 4 numbers)
    matrix_rows = []
    intr         = {}
    depth_meta   = {}

    for line in lines:
        if line.startswith("#"):
            continue
        # key=value pairs (intrinsics / depth metadata)
        if "=" in line:
            for kv in line.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if "x" in v and "resolution" in k:   # e.g. 256x192
                        w, h = v.split("x")
                        depth_meta["depth_w"] = int(w)
                        depth_meta["depth_h"] = int(h)
                    else:
                        try:
                            (intr if k in ("fx","fy","cx","cy","w","h")
                             else depth_meta)[k] = float(v)
                        except ValueError:
                            pass
            continue
        # numeric rows
        nums = line.split()
        if len(nums) == 4:
            try:
                matrix_rows.append([float(x) for x in nums])
            except ValueError:
                pass

    if len(matrix_rows) < 4:
        return None

    c2w = np.array(matrix_rows[:4], dtype=np.float64)
    return dict(c2w=c2w, intr=intr, depth_meta=depth_meta)


def _parse_plain_matrices(text: str) -> List[dict]:
    """Fallback: each non-blank line = 16 space-separated floats (row-major 4×4)."""
    frames_data = []
    for line in text.splitlines():
        nums = line.strip().split()
        if len(nums) == 16:
            c2w = np.array(nums, dtype=np.float64).reshape(4, 4)
            frames_data.append(dict(c2w=c2w, intr={}, depth_meta={}))
    return frames_data


# ─── DataLoader ───────────────────────────────────────────────────────────────

class DataLoader:
    """
    Parameters
    ----------
    data_dir     : root directory
    rgb_subdir   : subfolder for RGB images  (relative to data_dir)
    depth_subdir : subfolder for EXR depths  (relative to data_dir)
    pose_file    : pose filename             (relative to data_dir)
    default_intrinsics : fallback if pose file has no intrinsics
    depth_scale  : metres per raw depth unit
    """

    RGB_EXTS   = {".jpg", ".jpeg", ".png"}
    DEPTH_EXTS = {".exr"}

    def __init__(
        self,
        data_dir:     Path,
        rgb_subdir:   str = "rgb",
        depth_subdir: str = "depth",
        pose_file:    str = "poses.txt",
        default_intrinsics: Optional[CameraIntrinsics] = None,
        depth_scale:  float = 1.0,
    ):
        self.data_dir    = Path(data_dir)
        self.rgb_dir     = self.data_dir / rgb_subdir
        self.depth_dir   = self.data_dir / depth_subdir
        self.pose_path   = self.data_dir / pose_file
        self.default_intr = default_intrinsics
        self.depth_scale  = depth_scale

    # ── public ────────────────────────────────────────────────────────────────

    def load_all(self) -> List[Frame]:
        self._validate_dirs()
        rgb_paths   = self._sorted_images(self.rgb_dir,   self.RGB_EXTS)
        depth_paths = self._sorted_images(self.depth_dir, self.DEPTH_EXTS)
        pose_data   = _parse_pose_file(self.pose_path)

        n = min(len(rgb_paths), len(depth_paths), len(pose_data))
        if n == 0:
            raise RuntimeError("No matching frames found. Check directory structure.")
        if len(rgb_paths) != len(depth_paths) or len(rgb_paths) != len(pose_data):
            log.warning(
                f"Count mismatch: RGB={len(rgb_paths)} depth={len(depth_paths)} "
                f"poses={len(pose_data)}. Using first {n} frames."
            )

        frames = []
        for i in range(n):
            pd   = pose_data[i]
            intr = self._resolve_intrinsics(pd["intr"], rgb_paths[i])
            dm   = pd.get("depth_meta", {})
            dep_res = None
            if "depth_w" in dm and "depth_h" in dm:
                dep_res = (int(dm["depth_w"]), int(dm["depth_h"]))

            frames.append(Frame(
                idx         = i,
                rgb_path    = rgb_paths[i],
                depth_path  = depth_paths[i],
                c2w         = pd["c2w"],
                intrinsics  = intr,
                depth_scale = float(dm.get("depth_scale", self.depth_scale)),
                depth_res   = dep_res,
            ))

        log.info(f"DataLoader: {n} frames ready")
        return frames

    # ── private ───────────────────────────────────────────────────────────────

    def _validate_dirs(self):
        for d in (self.rgb_dir, self.depth_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Directory not found: {d}")
        if not self.pose_path.is_file():
            raise FileNotFoundError(f"Pose file not found: {self.pose_path}")

    @staticmethod
    def _sorted_images(folder: Path, exts: set) -> List[Path]:
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in exts
        )
        return files

    def _resolve_intrinsics(self, intr_dict: dict, rgb_path: Path) -> CameraIntrinsics:
        """Merge pose-file intrinsics with defaults / image dimensions."""
        # Try to read actual image size without loading pixel data
        w = int(intr_dict.get("w", 0))
        h = int(intr_dict.get("h", 0))

        if w == 0 or h == 0:
            try:
                from PIL import Image
                with Image.open(rgb_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 1920, 1440   # safe default

        # Fall back to sensible defaults if fx/fy not in pose file
        fx = float(intr_dict.get("fx", w * 0.8))
        fy = float(intr_dict.get("fy", fx))
        cx = float(intr_dict.get("cx", w / 2.0))
        cy = float(intr_dict.get("cy", h / 2.0))

        if self.default_intr:
            fx = fx or self.default_intr.fx
            fy = fy or self.default_intr.fy
            cx = cx or self.default_intr.cx
            cy = cy or self.default_intr.cy

        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)