"""
smart_tuner.py  — v3.0  "Physics-Correct, Record3D-Native"
===========================================================

ROOT CAUSE FIXES over v2.x:
  [FIX-1]  Depth loader now reads .exr FIRST (float32, metres, full res),
           falls back to .bin (float32, metres, LiDAR res 192×256),
           and MASKS with .conf (uint8, keep conf >= 1).
           The .png is NEVER used for statistics — it is a coloured preview.

  [FIX-2]  Poses.txt inline intrinsics parsed directly per-frame:
             # intrinsics fx=1351.96 fy=1351.96 cx=957.69 cy=721.78 w=1920 h=1440
           This gives exact fx every frame — no guessing, no fallback needed.

  [FIX-3]  Voxel size derived from iPhone LiDAR angular resolution
           (native 256×192 at same FOV as RGB camera):
             vox = p50_depth × tan(hfov / lidar_native_width)
           This is the physical minimum resolvable feature, not a Nyquist guess.

  [FIX-4]  Poses.txt 4×4 matrix = c2w (camera-to-world).
           Translation column [:3, 3] = camera world position in metres.
           Span computed from real positions, NOT rotation columns.

  [FIX-5]  Dynamic score fixed: compares consecutive EXR frames aligned by
           valid-pixel mask. Previous version compared misaligned buffers.

  [FIX-6]  flying_pixel_jump_thresh_m is now based on actual depth gradient
           p95 from EXR, NOT artificially capped. Remove min(..., 0.02)
           in pipeline.py Stage 4.

  [FIX-7]  DataLoader pose parsing: reads BOTH the 4×4 matrix AND the
           # intrinsics comment line so frames carry exact fx/fy/cx/cy/w/h.

Interface (drop-in, same as v2.x):
    cfg = AutoTuner(
        frames      = frames,
        rgb_dir     = src / "rgb",
        depth_dir   = src / "depth",
        depth_scale = 1.0,          # leave 1.0 — EXR is already in metres
        enable_verification = True,
        output_dir  = out,
    ).compute()

All parameters are returned in cfg dict — identical keys to v2.x.
"""

from __future__ import annotations

import logging
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger("smart_tuner")

# ──────────────────────────────────────────────────────────────────────────────
# Internal data structures  (same shape as v2.x — pipeline.py unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryHints:
    positions:          np.ndarray   # (N, 3) camera world positions, metres
    span_m:             float        # max pairwise distance between cameras
    median_step_m:      float        # median inter-frame step
    max_step_m:         float        # 99th-percentile step
    trajectory_type:    str          # orbit | room | corridor | outdoor
    approx_volume_m3:   float
    approx_surface_m2:  float
    n_frames:           int
    rotation_spread_deg: float


@dataclass
class DepthHints:
    p01: float; p05: float; p25: float
    p50: float; p75: float; p95: float; p99: float
    mean: float; std: float
    depth_range: float
    point_spacing_m: float
    gradient_p95: float
    valid_ratio: float
    dynamic_score: float
    plane_residual_m: float
    dominant_normals: np.ndarray
    sky_ratio: float


@dataclass
class ImageHints:
    mean_sharpness:    float
    sharpness_std:     float
    mean_brightness:   float
    color_temperature: str
    has_sky:           bool
    texture_richness:  float
    blur_ratio:        float


# ──────────────────────────────────────────────────────────────────────────────
# Poses.txt parser  — reads both matrix AND inline intrinsics comment
# ──────────────────────────────────────────────────────────────────────────────

class PosesFileParser:
    """
    Reads a poses.txt that looks like:

        # Frame 0
        0.009329 0.996401 0.084256 0.009433
        -0.898680 -0.028595 0.437671 0.052971
        0.438505 -0.079802 0.895179 0.083118
        0.000000 0.000000 0.000000 1.000000
        # intrinsics fx=1351.96 fy=1351.96 cx=957.69 cy=721.78 w=1920 h=1440

    Returns list of dicts: {c2w, fx, fy, cx, cy, w, h}
    The c2w is a (4,4) float64 camera-to-world matrix.
    """

    _INTR_RE = re.compile(
        r"fx=([\d.]+).*?fy=([\d.]+).*?cx=([\d.]+).*?cy=([\d.]+).*?w=(\d+).*?h=(\d+)"
    )

    def parse(self, pose_file: Path) -> List[Dict]:
        text   = Path(pose_file).read_text()
        blocks = re.split(r"#\s*Frame\s+\d+", text)
        results = []
        for block in blocks:
            if not block.strip():
                continue
            rows = []
            intr = {}
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    m = self._INTR_RE.search(line)
                    if m:
                        intr = dict(
                            fx=float(m.group(1)), fy=float(m.group(2)),
                            cx=float(m.group(3)), cy=float(m.group(4)),
                            w=int(m.group(5)),    h=int(m.group(6)),
                        )
                else:
                    vals = [float(v) for v in line.split()]
                    if len(vals) == 4:
                        rows.append(vals)
            if len(rows) == 4:
                c2w = np.array(rows, dtype=np.float64)
                results.append({"c2w": c2w, **intr})
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Depth loader  — Record3D / iPhone LiDAR native formats
# ──────────────────────────────────────────────────────────────────────────────

class Record3DDepthLoader:
    """
    Loads one depth frame from the depth directory.

    File priority (highest accuracy first):
      1. <stem>.exr   — OpenEXR float32, metres, full RGB resolution (upscaled)
      2. <stem>.bin   — raw float32 binary, native LiDAR res (192×256 typical)
      3. <stem>.npy   — float32 metres

    Confidence mask:
      If <stem>.conf exists (uint8, same spatial size as .bin):
        mask = conf >= 1   (0 = no measurement)
      Applied to whichever source is used.

    Returns: float32 ndarray in METRES, 0 = invalid pixel.
    """

    # LiDAR native resolution (iPhone 12 Pro+, iPad Pro 2020+)
    LIDAR_H = 192
    LIDAR_W = 256

    def __init__(self, depth_dir: Path):
        self.depth_dir = Path(depth_dir)
        self._warned: set = set()

    def load(self, stem: str) -> Optional[np.ndarray]:
        base = self.depth_dir / stem

        # ── 1. EXR (best: full-res, already in metres) ─────────────────────
        exr_path = base.with_suffix(".exr")
        if exr_path.exists():
            depth = self._load_exr(exr_path)
            if depth is not None:
                return self._apply_conf(depth, base)

        # ── 2. BIN (native LiDAR resolution, float32 metres) ───────────────
        bin_path = base.with_suffix(".bin")
        if bin_path.exists():
            depth = self._load_bin(bin_path)
            if depth is not None:
                return self._apply_conf(depth, base)

        # ── 3. NPY fallback ────────────────────────────────────────────────
        npy_path = base.with_suffix(".npy")
        if npy_path.exists():
            d = np.load(str(npy_path)).astype(np.float32)
            if d.max() > 100:     # stored in mm
                d = d / 1000.0
            return d

        # ── 4. PNG  (NEVER use for stats, but warn if only option) ─────────
        png_path = base.with_suffix(".png")
        if png_path.exists() and stem not in self._warned:
            log.warning(
                f"  DepthLoader: only .png available for '{stem}'. "
                "Record3D .png files are coloured previews, NOT metric depth. "
                "Statistics will be unreliable. Use the .exr or .bin files."
            )
            self._warned.add(stem)

        return None

    # ── EXR loader ────────────────────────────────────────────────────────────

    def _load_exr(self, path: Path) -> Optional[np.ndarray]:
        """Read OpenEXR float32. Returns metres or None."""
        try:
            # cv2.IMREAD_ANYDEPTH reads the first channel as float32
            d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            if d is None:
                return None
            d = d.astype(np.float32)
            # Sanity: EXR from Record3D is always in metres (0..~10m for indoors)
            # If somehow it was stored in mm, rescale
            if d[d > 0].mean() > 100:
                d = d / 1000.0
            return d
        except Exception as e:
            log.debug(f"  EXR load failed {path}: {e}")
            return None

    # ── BIN loader ────────────────────────────────────────────────────────────

    def _load_bin(self, path: Path) -> Optional[np.ndarray]:
        """
        Read raw float32 binary depth.
        Record3D stores shape=(LIDAR_H, LIDAR_W) = (192, 256) float32.
        Some pipelines store (H, W) matching the RGB frame.
        We try 192×256 first, then fall back to nearest square.
        """
        try:
            raw = np.fromfile(str(path), dtype=np.float32)
            n   = len(raw)
            if n == self.LIDAR_H * self.LIDAR_W:
                return raw.reshape(self.LIDAR_H, self.LIDAR_W)
            # Try other common shapes
            for (hh, ww) in [(480, 640), (720, 1280), (1440, 1920), (1080, 1920)]:
                if n == hh * ww:
                    return raw.reshape(hh, ww)
            # Guess square-ish
            side = int(np.sqrt(n))
            if side * side == n:
                return raw.reshape(side, side)
            log.debug(f"  BIN: cannot reshape {n} floats")
            return None
        except Exception as e:
            log.debug(f"  BIN load failed {path}: {e}")
            return None

    # ── Confidence mask ───────────────────────────────────────────────────────

    def _apply_conf(self, depth: np.ndarray, base: Path) -> np.ndarray:
        """Zero-out pixels with conf == 0 (no measurement)."""
        conf_path = base.with_suffix(".conf")
        if not conf_path.exists():
            return depth
        try:
            raw_conf = np.fromfile(str(conf_path), dtype=np.uint8)
            h, w = depth.shape
            if len(raw_conf) == h * w:
                conf = raw_conf.reshape(h, w)
            else:
                # Conf may be at LiDAR resolution; resize to match depth
                conf_2d = raw_conf.reshape(self.LIDAR_H, self.LIDAR_W)
                conf    = cv2.resize(conf_2d, (w, h), interpolation=cv2.INTER_NEAREST)
            mask         = conf < 1        # 0 = no measurement
            depth        = depth.copy()
            depth[mask]  = 0.0
            return depth
        except Exception as e:
            log.debug(f"  Conf mask failed {conf_path}: {e}")
            return depth


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer 1 — Scene Geometry from poses
# ──────────────────────────────────────────────────────────────────────────────

class SceneGeometryAnalyzer:
    """
    Extracts camera positions from frame objects.
    Supports:
      • frame.c2w / frame.pose / frame.transform (4×4 or 3×4 matrix)
      • frame.w2c / frame.T_cw (inverted automatically)
      • dict  {"c2w": ...}
      • separate R + t attributes
    """

    def __init__(self, frames: list):
        self.frames = frames

    def analyze(self) -> GeometryHints:
        poses     = self._extract_poses()
        positions = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        N         = len(positions)

        # Span (sub-sample for large N to keep O(n))
        sample = positions if N <= 500 else positions[
            np.random.choice(N, 300, replace=False)]
        diffs  = sample[:, None, :] - sample[None, :, :]
        span_m = float(np.sqrt((diffs ** 2).sum(-1)).max())

        steps        = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        median_step  = float(np.median(steps))
        max_step     = float(np.percentile(steps, 99))

        lo, hi   = positions.min(0), positions.max(0)
        dims     = hi - lo
        volume   = float(np.prod(np.maximum(dims, 0.01)))
        surface  = float(2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[0]*dims[2]))

        traj = self._classify(positions, span_m, median_step)

        fwd      = rotations[:, :3, 2]
        fwd_norm = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
        dots     = np.clip(fwd_norm @ fwd_norm.T, -1, 1)
        rot_spread = float(np.degrees(np.arccos(dots)).max())

        return GeometryHints(
            positions=positions, span_m=span_m,
            median_step_m=median_step, max_step_m=max_step,
            trajectory_type=traj, approx_volume_m3=volume,
            approx_surface_m2=surface, n_frames=N,
            rotation_spread_deg=rot_spread,
        )

    # ── Pose extraction ──────────────────────────────────────────────────────

    _C2W_ATTRS = ["c2w", "pose", "extrinsic", "transform", "cam2world",
                  "camera_pose", "T_wc", "T", "world_mat", "camera_to_world"]
    _W2C_ATTRS = ["w2c", "T_cw", "extrinsics"]

    def _extract_poses(self) -> np.ndarray:
        f0 = self.frames[0]

        # Dict-like frames (from PosesFileParser)
        if isinstance(f0, dict):
            poses = [np.array(f["c2w"], dtype=np.float64) for f in self.frames
                     if "c2w" in f]
            return np.stack(self._pad44(poses))

        # Object frames — try c2w attrs first
        for attr in self._C2W_ATTRS:
            val = getattr(f0, attr, None)
            if val is None:
                continue
            try:
                arr = np.array(val, dtype=np.float64)
                if arr.shape in [(4, 4), (3, 4)]:
                    poses = [self._to44(getattr(f, attr)) for f in self.frames]
                    valid = [p for p in poses if p is not None]
                    if len(valid) >= 3:
                        log.info(f"  Poses: frame.{attr}")
                        return np.stack(valid)
            except Exception:
                continue

        # Try w2c (invert)
        for attr in self._W2C_ATTRS:
            val = getattr(f0, attr, None)
            if val is None:
                continue
            try:
                arr = np.array(val, dtype=np.float64)
                if arr.shape in [(4, 4), (3, 4)]:
                    poses = [self._to44(getattr(f, attr), invert=True)
                             for f in self.frames]
                    valid = [p for p in poses if p is not None]
                    if len(valid) >= 3:
                        log.info(f"  Poses: frame.{attr} (inverted)")
                        return np.stack(valid)
            except Exception:
                continue

        raise ValueError(
            f"Cannot find pose matrix. Frame attrs: "
            f"{[a for a in vars(f0) if not a.startswith('_')]}"
        )

    def _to44(self, val, invert=False) -> Optional[np.ndarray]:
        try:
            P = np.array(val, dtype=np.float64)
            if P.shape == (3, 4):
                P = np.vstack([P, [0, 0, 0, 1]])
            if P.shape != (4, 4):
                return None
            return np.linalg.inv(P) if invert else P
        except Exception:
            return None

    def _pad44(self, poses):
        out = []
        for p in poses:
            if p.shape == (3, 4):
                p = np.vstack([p, [0, 0, 0, 1]])
            if p.shape == (4, 4):
                out.append(p)
        return out

    def _classify(self, pos: np.ndarray, span: float, step: float) -> str:
        lo, hi      = pos.min(0), pos.max(0)
        dims        = sorted((hi - lo).tolist(), reverse=True)
        flatness    = dims[2] / (dims[0] + 1e-9)
        elongation  = dims[0] / (dims[1] + 1e-9)
        start_end   = np.linalg.norm(pos[-1] - pos[0])
        circularity = 1.0 - min(start_end / (span + 1e-9), 1.0)

        if span < 0.5:                                return "orbit"
        if circularity > 0.6 and flatness < 0.3:     return "orbit"
        if elongation > 3.0 and flatness < 0.25:     return "corridor"
        if span > 15.0:                               return "outdoor"
        return "room"


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer 2 — Depth Statistics  (EXR/BIN native, conf-masked)
# ──────────────────────────────────────────────────────────────────────────────

class DepthStatisticsAnalyzer:
    """
    Samples depth frames and returns statistics in METRES using the correct
    file format: .exr first, .bin fallback, .conf mask applied.
    """

    SAMPLE_FRAMES = 40
    PIXEL_SAMPLE  = 10_000

    def __init__(self, frames: list, depth_dir: Path, depth_scale: float = 1.0):
        self.frames      = frames
        self.depth_dir   = Path(depth_dir)
        self.depth_scale = depth_scale   # kept for API compat; EXR is already metres
        self._loader     = Record3DDepthLoader(depth_dir)

    def analyze(self) -> DepthHints:
        sample = self._pick_frames()

        all_depths:    List[np.ndarray] = []
        all_gradients: List[np.ndarray] = []
        plane_res:     List[float]      = []
        sky_ratios:    List[float]      = []
        dynamic_diffs: List[float]      = []
        normals_list:  List[np.ndarray] = []
        prev_depth:    Optional[np.ndarray] = None

        for i, f in enumerate(sample):
            stem  = self._get_stem(f)
            depth = self._loader.load(stem) if stem else None
            if depth is None:
                continue

            valid = (depth > 0.05) & (depth < 200.0) & np.isfinite(depth)
            vidx  = np.where(valid.ravel())[0]
            if len(vidx) < 100:
                continue

            chosen  = np.random.choice(vidx, min(self.PIXEL_SAMPLE, len(vidx)), replace=False)
            sampled = depth.ravel()[chosen]
            all_depths.append(sampled)

            # Depth gradient for flying-pixel threshold
            gx   = np.abs(np.diff(depth, axis=1))
            gy   = np.abs(np.diff(depth, axis=0))
            gv   = np.concatenate([gx[valid[:, :-1]].ravel(),
                                   gy[valid[:-1, :]].ravel()])
            if len(gv) > 100:
                all_gradients.append(
                    gv[np.random.choice(len(gv), min(3000, len(gv)), replace=False)])

            # Sky detection — top 1/5 of frame, far pixels
            h_img = depth.shape[0]
            top_band  = depth[:h_img // 5, :]
            top_valid = top_band[(top_band > 0.1) & np.isfinite(top_band)]
            if len(top_valid) > 50 and len(sampled) > 50:
                thresh = np.percentile(sampled, 88)
                sky_ratios.append(float((top_valid > thresh * 0.9).mean()))

            # Dynamic score — depth change between consecutive frames
            if prev_depth is not None and prev_depth.shape == depth.shape:
                change     = np.abs(depth - prev_depth)
                both_valid = valid & (prev_depth > 0.05) & np.isfinite(prev_depth)
                if both_valid.sum() > 200:
                    dynamic_diffs.append(float(np.percentile(change[both_valid], 90)))

            # Plane residual (first 8 frames only)
            if i < 8:
                res = self._fit_plane_residual(depth, valid)
                if res is not None:
                    plane_res.append(res)
                n = self._dominant_normals(depth, valid)
                if n is not None:
                    normals_list.append(n)

            prev_depth = depth.copy()

        # ── Aggregate ──────────────────────────────────────────────────────────
        all_d = np.concatenate(all_depths) if all_depths else np.array([2.0])
        all_d = all_d[np.isfinite(all_d) & (all_d > 0)]
        if len(all_d) == 0:
            all_d = np.array([2.0])

        p01, p05, p25, p50, p75, p95, p99 = np.percentile(all_d, [1, 5, 25, 50, 75, 95, 99])

        all_grad = np.concatenate(all_gradients) if all_gradients else np.array([0.02])
        grad_p95 = float(np.percentile(all_grad, 95))

        valid_ratio   = float(np.clip(len(all_d) / max(self.PIXEL_SAMPLE * len(sample), 1), 0, 1))
        dynamic_score = float(np.clip(
            np.mean(dynamic_diffs) / (float(p50) + 1e-9) if dynamic_diffs else 0.0, 0, 1))
        plane_residual = float(np.median(plane_res)) if plane_res else float(p50) * 0.005
        sky_ratio      = float(np.mean(sky_ratios)) if sky_ratios else 0.0
        dom_normals    = (np.vstack(normals_list) if normals_list
                          else np.eye(3, dtype=np.float32))

        log.info(f"  Depth stats (EXR/BIN, conf-masked): "
                 f"p01={p01:.3f}  p05={p05:.3f}  p50={p50:.3f}  "
                 f"p95={p95:.3f}  p99={p99:.3f}  grad_p95={grad_p95:.4f}  "
                 f"valid_ratio={valid_ratio:.2f}  dynamic={dynamic_score:.3f}")

        return DepthHints(
            p01=float(p01), p05=float(p05), p25=float(p25),
            p50=float(p50), p75=float(p75), p95=float(p95), p99=float(p99),
            mean=float(all_d.mean()), std=float(all_d.std()),
            depth_range=float(p99 - p01),
            point_spacing_m=float(p50) * 0.003,
            gradient_p95=grad_p95,
            valid_ratio=valid_ratio,
            dynamic_score=dynamic_score,
            plane_residual_m=plane_residual,
            dominant_normals=dom_normals,
            sky_ratio=sky_ratio,
        )

    # ── Frame sampling ────────────────────────────────────────────────────────

    def _pick_frames(self) -> list:
        N = len(self.frames)
        if N <= self.SAMPLE_FRAMES:
            return self.frames
        idx = np.round(np.linspace(0, N - 1, self.SAMPLE_FRAMES)).astype(int)
        return [self.frames[i] for i in idx]

    # ── Stem resolution ───────────────────────────────────────────────────────

    def _get_stem(self, frame) -> Optional[str]:
        """Get the filename stem for depth lookup from a frame object or dict."""
        # Dict frame from PosesFileParser won't have paths; caller handles by index
        if isinstance(frame, dict):
            return frame.get("_stem")

        # Try depth path first
        for attr in ("depth_path", "depth", "depth_file", "d_path", "lidar_path"):
            v = getattr(frame, attr, None)
            if v and isinstance(v, (str, Path)):
                return Path(v).stem

        # Fall back to RGB path stem
        for attr in ("rgb_path", "image_path", "color_path", "img_path", "rgb", "image"):
            v = getattr(frame, attr, None)
            if v and isinstance(v, (str, Path)):
                return Path(v).stem

        return None

    # ── Plane fitting ─────────────────────────────────────────────────────────

    def _fit_plane_residual(self, depth: np.ndarray,
                             valid: np.ndarray) -> Optional[float]:
        try:
            h, w   = depth.shape
            yy, xx = np.meshgrid(np.arange(w), np.arange(h))
            pts    = np.stack([xx[valid], yy[valid], depth[valid]], axis=1).astype(np.float64)
            if len(pts) < 200:
                return None
            pts = pts[np.random.choice(len(pts), min(3000, len(pts)), replace=False)]
            best = np.inf
            for _ in range(40):
                s = pts[np.random.choice(len(pts), 3, replace=False)]
                n = np.cross(s[1] - s[0], s[2] - s[0])
                nn = np.linalg.norm(n)
                if nn < 1e-9:
                    continue
                n  /= nn
                res = np.abs(pts @ n - n @ s[0])
                inn = res < np.percentile(res, 60)
                if inn.sum() < 50:
                    continue
                r = float(np.median(res[inn]))
                if r < best:
                    best = r
            return best if best < np.inf else None
        except Exception:
            return None

    # ── Normal estimation ─────────────────────────────────────────────────────

    def _dominant_normals(self, depth: np.ndarray,
                           valid: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w    = depth.shape
            normals = []
            step    = max(h // 8, 1)
            for i in range(step, h - step, step):
                for j in range(step, w - step, step):
                    patch = depth[i-2:i+3, j-2:j+3]
                    if not valid[i-2:i+3, j-2:j+3].all() or patch.std() < 0.001:
                        continue
                    yy, xx = np.meshgrid(range(5), range(5))
                    pts = np.stack([xx.ravel().astype(float),
                                    yy.ravel().astype(float),
                                    patch.ravel().astype(float)], axis=1)
                    pts -= pts.mean(0)
                    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
                    normals.append(Vt[-1])
            if not normals:
                return None
            normals = np.array(normals)
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=min(3, len(normals)), n_init=3, random_state=0)
                km.fit(np.abs(normals))
                return km.cluster_centers_.astype(np.float32)
            except ImportError:
                return normals[:3].astype(np.float32)
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer 3 — Image Quality
# ──────────────────────────────────────────────────────────────────────────────

class ImageQualityAnalyzer:
    SAMPLE_FRAMES = 20

    def __init__(self, frames: list, rgb_dir: Path):
        self.frames  = frames
        self.rgb_dir = Path(rgb_dir)

    def analyze(self) -> ImageHints:
        N      = len(self.frames)
        sample = (self.frames if N <= self.SAMPLE_FRAMES
                  else [self.frames[i] for i in
                        np.round(np.linspace(0, N-1, self.SAMPLE_FRAMES)).astype(int)])

        sharpness_vals, brightness_vals, texture_vals = [], [], []
        r_means, b_means, sky_detected = [], [], []
        blur_count = 0

        for f in sample:
            img = self._load_rgb(f)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

            lap  = cv2.Laplacian(gray, cv2.CV_32F)
            shp  = float(lap.var())
            sharpness_vals.append(shp)
            if shp < 50.0:
                blur_count += 1

            brightness_vals.append(float(gray.mean() / 255.0))

            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            texture_vals.append(float(np.sqrt(sx**2 + sy**2).mean()))

            r_means.append(float(img[:, :, 2].mean()))
            b_means.append(float(img[:, :, 0].mean()))

            h_img = img.shape[0]
            top   = img[:h_img // 7, :]
            tg    = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
            th    = cv2.cvtColor(top, cv2.COLOR_BGR2HSV).astype(np.float32)
            sky_detected.append(
                float((tg > 180).mean()) > 0.4 and float((th[:, :, 1] < 60).mean()) > 0.5)

        n      = max(len(sharpness_vals), 1)
        mean_r = float(np.mean(r_means)) if r_means else 128.0
        mean_b = float(np.mean(b_means)) if b_means else 128.0
        rb     = mean_r / (mean_b + 1e-9)
        color_temp = "warm" if rb > 1.15 else ("cool" if rb < 0.87 else "neutral")

        return ImageHints(
            mean_sharpness   = float(np.mean(sharpness_vals)) if sharpness_vals else 100.0,
            sharpness_std    = float(np.std(sharpness_vals))  if sharpness_vals else 10.0,
            mean_brightness  = float(np.mean(brightness_vals)) if brightness_vals else 0.5,
            color_temperature= color_temp,
            has_sky          = sum(sky_detected) > len(sky_detected) * 0.3,
            texture_richness = float(np.mean(texture_vals)) if texture_vals else 10.0,
            blur_ratio       = blur_count / n,
        )

    def _load_rgb(self, frame) -> Optional[np.ndarray]:
        if isinstance(frame, dict):
            p = frame.get("rgb_path") or frame.get("image_path")
            if p and Path(str(p)).exists():
                return cv2.imread(str(p))
            return None
        for attr in ("rgb_path", "image_path", "color_path", "img_path", "rgb", "image"):
            v = getattr(frame, attr, None)
            if v and isinstance(v, (str, Path)):
                try:
                    img = cv2.imread(str(v))
                    if img is not None:
                        return img
                except Exception:
                    pass
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Intrinsics Reader — reads from frames OR poses.txt comment, exact values
# ──────────────────────────────────────────────────────────────────────────────

class IntrinsicsReader:
    """
    Priority:
      1. frame dict with fx/fy/cx/cy/w/h keys  (from PosesFileParser)
      2. frame.intrinsics 3×3 K matrix
      3. frame.fx scalar
      4. Estimate from image size (70° hfov)
      5. iPhone 13 Pro hardcoded fallback
    """

    def read(self, frames: list) -> Dict[str, float]:
        if not frames:
            return self._fallback()
        f = frames[0]

        # Dict frame from PosesFileParser — already has exact intrinsics
        if isinstance(f, dict) and "fx" in f:
            log.info(f"  Intrinsics: from poses.txt comment "
                     f"fx={f['fx']:.2f} w={f['w']} h={f['h']}")
            return {k: f[k] for k in ("fx","fy","cx","cy","w","h")}

        # 3×3 K matrix
        for attr in ("intrinsics", "K", "camera_matrix", "intrinsic", "camera_intrinsics"):
            K = getattr(f, attr, None)
            if K is None:
                continue
            try:
                arr = np.array(K, dtype=np.float64)
                if arr.shape == (3, 3) and arr[0, 0] > 10:
                    fx, fy = arr[0, 0], arr[1, 1]
                    cx, cy = arr[0, 2], arr[1, 2]
                    w, h   = self._wh(f, cx, cy)
                    log.info(f"  Intrinsics: frame.{attr} fx={fx:.2f} w={w}")
                    return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
            except Exception:
                continue

        # Scalar fx / fy
        fx = getattr(f, "fx", None)
        fy = getattr(f, "fy", None)
        if fx is not None and fy is not None:
            cx = float(getattr(f, "cx", 0.0))
            cy = float(getattr(f, "cy", 0.0))
            w  = int(getattr(f, "width",  None) or getattr(f, "w", None) or cx * 2)
            h  = int(getattr(f, "height", None) or getattr(f, "h", None) or cy * 2)
            return dict(fx=float(fx), fy=float(fy), cx=cx, cy=cy, w=w, h=h)

        # Estimate from image
        for attr in ("rgb_path", "image_path", "color_path", "img_path"):
            p = getattr(f, attr, None)
            if p and Path(str(p)).exists():
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        h_img, w_img = img.shape[:2]
                        fx_est = w_img / (2.0 * np.tan(np.radians(35)))
                        log.warning(f"  Intrinsics: estimated fx={fx_est:.1f} "
                                    f"from {w_img}×{h_img} (assumed 70° hfov)")
                        return dict(fx=fx_est, fy=fx_est,
                                    cx=w_img/2, cy=h_img/2, w=w_img, h=h_img)
                except Exception:
                    pass

        return self._fallback()

    def _wh(self, f, cx, cy):
        for wa, ha in (("width","height"), ("w","h"),
                       ("img_width","img_height"), ("image_width","image_height")):
            w = getattr(f, wa, None); h = getattr(f, ha, None)
            if w and h and int(w) > 64:
                return int(w), int(h)
        return max(int(cx * 2), 640), max(int(cy * 2), 480)

    def _fallback(self):
        log.warning("  Intrinsics: hardcoded iPhone 13 Pro fallback (1920×1440 fx=1377). "
                    "Add frame.intrinsics or parse poses.txt intrinsics comments for accuracy.")
        return dict(fx=1377.22, fy=1377.22, cx=957.80, cy=722.04, w=1920, h=1440)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Deriver — v3.0 physics-correct
# ──────────────────────────────────────────────────────────────────────────────

class ParameterDeriver:
    """
    All parameters derived from first principles.

    KEY FORMULA — voxel size:
        vox = p50_depth * tan(hfov / LIDAR_NATIVE_WIDTH)
    This is the physical angular resolution of the iPhone LiDAR sensor
    (256 pixels across the same FOV as the RGB camera).
    It is the minimum resolvable feature size at median depth.

    For objects closer than ~1m, fall back to:
        vox = p50_depth * tan(hfov / 2) * 4 / w   (RGB Nyquist)
    because at close range the LiDAR is oversampled by the RGB camera.
    """

    LIDAR_NATIVE_W = 256    # iPhone / iPad Pro LiDAR horizontal pixels

    def derive(
        self,
        geo:        GeometryHints,
        dep:        DepthHints,
        img:        ImageHints,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        # ── Scale category ────────────────────────────────────────────────────
        span = geo.span_m
        traj = geo.trajectory_type

        if traj == "orbit" or span < 0.6:
            scale = "object"
        elif traj == "outdoor" or span > 15.0:
            scale = "outdoor"
        elif traj == "corridor":
            scale = "corridor"
        else:
            scale = "room"

        log.info(f"  Scale: {scale}  span={span:.3f}m  traj={traj}  "
                 f"p50_depth={dep.p50:.3f}m")

        # ── Camera geometry ───────────────────────────────────────────────────
        if intrinsics:
            fx       = intrinsics["fx"]
            w        = intrinsics["w"]
        else:
            fx       = 1377.22
            w        = 1920

        hfov_rad = 2.0 * np.arctan(w / (2.0 * fx))
        log.info(f"  Intrinsics: fx={fx:.2f}  w={w}  hfov={np.degrees(hfov_rad):.2f}°")

        # ── Voxel size ────────────────────────────────────────────────────────
        # Physical LiDAR resolution at median depth
        vox_lidar   = dep.p50 * np.tan(hfov_rad / self.LIDAR_NATIVE_W)
        # RGB camera Nyquist (4 pixels per voxel)
        vox_nyquist = 4.0 * dep.p50 * np.tan(hfov_rad / 2.0) / w
        # Noise floor: must be larger than depth noise
        vox_noise   = dep.plane_residual_m * 1.5
        # Step: should be smaller than inter-frame motion to avoid gaps
        vox_step    = geo.median_step_m * 0.35

        vox_raw = max(vox_lidar, vox_nyquist, vox_noise)
        vox_raw = max(vox_raw, vox_step * 0.5)

        clamps = {
            "object":   (0.003, 0.015),
            "room":     (0.008, 0.060),
            "corridor": (0.008, 0.060),
            "outdoor":  (0.020, 0.150),
        }
        vox = float(np.clip(vox_raw, *clamps[scale]))

        log.info(
            f"  vox: lidar={vox_lidar:.5f}  nyquist={vox_nyquist:.5f}  "
            f"noise={vox_noise:.5f}  step={vox_step:.5f}  raw={vox_raw:.5f}  "
            f"→ FINAL={vox:.5f}m"
        )

        # ── SDF truncation ────────────────────────────────────────────────────
        # Base multiplier by scale; increase if surface is noisy
        sdf_base  = {"object": 6.0, "room": 4.5, "corridor": 4.5, "outdoor": 3.5}[scale]
        noise_pen = float(np.clip(dep.plane_residual_m / (vox + 1e-9) * 0.25, 0.0, 2.0))
        sdf_trunc_mult = float(np.clip(sdf_base + noise_pen, 3.0, 12.0))

        # ── Depth range ───────────────────────────────────────────────────────
        sensor_min = {"object": 0.05, "room": 0.08, "corridor": 0.08, "outdoor": 0.15}[scale]
        min_d = float(np.clip(dep.p01 * 0.90, sensor_min, dep.p25))
        max_d = float(np.clip(dep.p99 * 1.15, min_d + dep.p50 * 0.5, 40.0))

        # ── Flying-pixel threshold ────────────────────────────────────────────
        # A real surface edge jumps ~p50*0.3 to p50*1.0 in depth.
        # Flying pixels from LiDAR edge bleed jump ~vox*3 to vox*8.
        # Threshold should sit between these ranges.
        # Use grad_p95 as the natural break-point: it captures real edges.
        # DO NOT cap with min(..., 0.02) in pipeline.py
        fly_min = vox * 5.0
        fly_max = max(dep.depth_range * 0.15, fly_min * 4.0)
        flying  = float(np.clip(dep.gradient_p95 * 0.5, fly_min, fly_max))

        log.info(f"  flying_px: grad_p95={dep.gradient_p95:.4f}  "
                 f"fly_min={fly_min:.4f}  fly_max={fly_max:.4f}  → {flying:.4f}m")

        # ── Poisson mesh depth ────────────────────────────────────────────────
        geometry_span = max(dep.depth_range, span)
        voxels_across = geometry_span / (vox + 1e-9)
        mdepth_raw    = int(np.round(np.log2(max(voxels_across, 2) + 1)))
        mdepth_bounds = {"object": (8, 10), "room": (9, 11),
                         "corridor": (9, 11), "outdoor": (10, 12)}
        mdepth = int(np.clip(mdepth_raw, *mdepth_bounds[scale]))

        # ── Face count ────────────────────────────────────────────────────────
        vox_area    = vox ** 2
        est_surface = (np.pi * dep.depth_range ** 2 if scale == "object"
                       else geo.approx_surface_m2)
        mfaces_raw  = int(est_surface / vox_area * 1.5)
        mfaces_bounds = {"object":   (30_000,   500_000),
                         "room":     (100_000, 3_000_000),
                         "corridor": (100_000, 2_500_000),
                         "outdoor":  (200_000, 8_000_000)}
        mfaces = int(np.clip(mfaces_raw, *mfaces_bounds[scale]))

        # ── Density quantile ──────────────────────────────────────────────────
        fd      = float(np.clip(geo.n_frames / 200.0, 0.25, 3.0))
        mq_base = {"object": 0.004, "room": 0.018, "corridor": 0.016, "outdoor": 0.012}[scale]
        mq      = float(np.clip(mq_base * fd, 0.002, 0.09))

        # ── Component filter ──────────────────────────────────────────────────
        mcomp   = {"object": 0.010, "room": 0.006, "corridor": 0.004, "outdoor": 0.003}[scale]
        mlargest = 0.0

        # ── Planar snap ───────────────────────────────────────────────────────
        planar_ransac = float(np.clip(vox * 2.0, 0.003, 0.06))
        planar_snap   = float(np.clip(vox * 1.5, 0.002, 0.05))
        planar_remove = float(np.clip(vox * 4.0, 0.005, 0.12))
        n_planes      = int(np.clip(max(3, int(span * 1.5)), 3, 15))

        # ── Dynamic objects ───────────────────────────────────────────────────
        has_dynamic           = dep.dynamic_score > 0.04
        dynamic_motion_thresh = float(np.clip(
            max(dep.dynamic_score * dep.p50 * 0.25, vox * 3), vox * 3, max_d * 0.08))
        dynamic_min_weight    = float(np.clip(0.35 - dep.dynamic_score * 0.25, 0.08, 0.5))

        # ── Border mask ───────────────────────────────────────────────────────
        sharp_cv    = img.sharpness_std / (img.mean_sharpness + 1e-9)
        border_frac = float(np.clip(0.05 + sharp_cv * 0.04, 0.04, 0.12))

        # ── Sky suppression ───────────────────────────────────────────────────
        use_sky           = (img.has_sky or dep.sky_ratio > 0.12) and scale in ("outdoor", "corridor")
        sky_bright_thresh = float(np.clip(img.mean_brightness * 255 * 1.25, 175, 240))
        sky_depth_min     = float(np.clip(dep.p95 * 0.80, max_d * 0.55, max_d * 0.92))

        # ── Bilateral filter ──────────────────────────────────────────────────
        noise_ratio   = dep.gradient_p95 / (dep.depth_range + 1e-9)
        use_bilateral = noise_ratio > 0.015 or scale == "object"

        # ── Mesh smoothing ────────────────────────────────────────────────────
        smooth_iter   = int(np.clip(1 + noise_ratio * 25, 1, 5))
        smooth_lambda = float(np.clip(0.25 + noise_ratio * 1.8, 0.15, 0.65))

        # ── Hole fill ─────────────────────────────────────────────────────────
        # max_hole_size in pixels; scale with vox inversely
        max_hole_size = int(np.clip(int(20 / (vox * 100 + 1e-9)), 5, 50))

        # ── Labels ───────────────────────────────────────────────────────────
        env_type   = {"object": "object", "room": "indoor",
                      "corridor": "indoor", "outdoor": "outdoor"}[scale]
        scene_mode = env_type

        params = {
            # TSDF Fusion
            "vox":                        vox,
            "sdf_trunc_multiplier":       sdf_trunc_mult,
            "min_d":                      min_d,
            "max_d":                      max_d,
            "scene":                      scene_mode,
            "flying_pixel_jump_thresh_m": flying,
            "use_bilateral":              use_bilateral,
            "border_frac":                border_frac,
            "use_sky_suppress":           use_sky,
            "sky_bright_thresh":          sky_bright_thresh,
            "sky_depth_min_m":            sky_depth_min,
            "dynamic_motion_thresh_m":    dynamic_motion_thresh,
            "dynamic_min_weight":         dynamic_min_weight,
            "planar_ransac_thresh":       planar_ransac,
            "n_planes":                   n_planes,
            "planar_snap_dist":           planar_snap,
            "planar_remove_dist":         planar_remove,
            "max_hole_size":              max_hole_size,
            # Meshing
            "mdepth":                     mdepth,
            "mfaces":                     mfaces,
            "mq":                         mq,
            "mcomp":                      mcomp,
            "mlargest":                   mlargest,
            # MeshCleaner
            "mesh_smooth_iter":           smooth_iter,
            "mesh_smooth_lambda":         smooth_lambda,
            "mesh_target_faces":          mfaces,
            # Metadata
            "env_type":                   env_type,
            "scan_span_m":                span,
            "has_dynamic_objects":        has_dynamic,
        }

        log.info(
            f"  PARAMETERS SUMMARY:\n"
            f"    vox={vox:.5f}m  sdf×{sdf_trunc_mult:.2f}  "
            f"depth=[{min_d:.3f},{max_d:.3f}]\n"
            f"    mdepth={mdepth}  mfaces={mfaces:,}  mq={mq:.4f}\n"
            f"    flying={flying:.4f}  bilateral={use_bilateral}  "
            f"smooth_iter={smooth_iter}  smooth_λ={smooth_lambda:.3f}\n"
            f"    dynamic={has_dynamic}  sky={use_sky}  "
            f"planar_ransac={planar_ransac:.4f}  n_planes={n_planes}"
        )
        return params


# ──────────────────────────────────────────────────────────────────────────────
# Mini Verifier (unchanged from v2.x API, now uses EXR depth)
# ──────────────────────────────────────────────────────────────────────────────

class MiniVerifier:
    VERIFY_FRAMES   = 40
    MAX_CORRECTIONS = 3

    def __init__(self, frames: list, output_dir: Path, depth_scale: float = 1.0):
        self.frames      = frames
        self.output_dir  = Path(output_dir)
        self.depth_scale = depth_scale

    def verify_and_correct(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import open3d as o3d
        except ImportError:
            log.warning("  MiniVerifier: open3d not installed — skipping verification")
            return cfg

        log.info(f"  MiniVerifier: {self.VERIFY_FRAMES}-frame TSDF test...")
        sample = self._pick_frames()

        for iteration in range(self.MAX_CORRECTIONS):
            pcd     = self._run_mini_tsdf(sample, cfg)
            if pcd is None:
                log.warning("  MiniVerifier: empty TSDF — skipping")
                break
            quality    = self._measure_quality(pcd, cfg)
            correction = self._compute_correction(quality, cfg)
            status     = "OK ✓" if correction is None else str(correction)
            log.info(f"  MiniVerifier iter {iteration+1}: "
                     f"pts={quality['n_points']}  density={quality['density']:.1f}  "
                     f"coverage={quality['coverage']:.2f}  → {status}")
            if correction is None:
                break
            cfg = {**cfg, **correction}

        return cfg

    def _pick_frames(self) -> list:
        N = len(self.frames)
        if N <= self.VERIFY_FRAMES:
            return self.frames
        mid, half = N // 2, self.VERIFY_FRAMES // 2
        return [self.frames[i] for i in range(max(0, mid-half), min(N, mid+half))]

    def _run_mini_tsdf(self, frames: list, cfg: Dict) -> Optional[Any]:
        try:
            import open3d as o3d
            loader = Record3DDepthLoader(self.output_dir.parent / "depth")

            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=cfg["vox"],
                sdf_trunc=cfg["vox"] * cfg["sdf_trunc_multiplier"],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
            fused = 0
            for f in frames:
                if isinstance(f, dict):
                    rgb  = f.get("rgb_path") or f.get("image_path")
                    K_v  = [f.get("fx"), f.get("fy"), f.get("cx"), f.get("cy")]
                    pose = f.get("c2w")
                    stem = f.get("_stem")
                else:
                    rgb  = getattr(f, "rgb_path", None) or getattr(f, "image_path", None)
                    K_v  = None
                    for attr in ("intrinsics","K","camera_matrix"):
                        K_v = getattr(f, attr, None)
                        if K_v is not None: break
                    pose = (getattr(f, "c2w", None) or getattr(f, "pose", None)
                            or getattr(f, "transform", None))
                    stem = (Path(str(getattr(f, "depth_path", "") or
                                    getattr(f, "rgb_path", ""))).stem)

                if not all([rgb, K_v is not None, pose is not None, stem]):
                    continue

                color_img = o3d.io.read_image(str(rgb))
                depth_arr = loader.load(stem)
                if depth_arr is None:
                    continue

                depth_img = o3d.geometry.Image((depth_arr * 1000).astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_img, depth_img,
                    depth_scale=1000.0,
                    depth_trunc=cfg["max_d"],
                    convert_rgb_to_intensity=False,
                )
                if isinstance(K_v, dict):
                    fx, fy, cx, cy = K_v["fx"], K_v["fy"], K_v["cx"], K_v["cy"]
                elif hasattr(K_v, "shape"):
                    Ka = np.array(K_v)
                    if Ka.shape == (3, 3):
                        fx, fy, cx, cy = Ka[0,0], Ka[1,1], Ka[0,2], Ka[1,2]
                    else:
                        fx, fy, cx, cy = Ka.ravel()[:4]
                else:
                    fx, fy, cx, cy = K_v[0], K_v[1], K_v[2], K_v[3]

                h_d, w_d = depth_arr.shape
                intr = o3d.camera.PinholeCameraIntrinsic(w_d, h_d, fx, fy, cx, cy)
                P = np.array(pose, dtype=np.float64)
                if P.shape == (3, 4):
                    P = np.vstack([P, [0, 0, 0, 1]])
                volume.integrate(rgbd, intr, np.linalg.inv(P))
                fused += 1

            if fused < 5:
                log.info(f"  MiniVerifier: only {fused} frames fused")
                return None
            return volume.extract_point_cloud()
        except Exception as e:
            log.debug(f"  MiniVerifier TSDF error: {e}")
            return None

    def _measure_quality(self, pcd, cfg: Dict) -> Dict[str, float]:
        try:
            pts = np.asarray(pcd.points)
            n   = len(pts)
            if n < 10:
                return {"n_points": n, "density": 0.0, "coverage": 0.0}
            lo, hi  = pts.min(0), pts.max(0)
            vol     = float(np.prod(np.maximum(hi - lo, 1e-3)))
            density = n / vol
            return {"n_points": n, "density": density,
                    "coverage": float(min(vol / max(cfg.get("approx_volume_m3", vol), 1e-3), 1.0))}
        except Exception:
            return {"n_points": 0, "density": 0.0, "coverage": 0.0}

    def _compute_correction(self, quality: Dict, cfg: Dict) -> Optional[Dict]:
        vox = cfg["vox"]
        if quality["density"] < 1_000 and quality["n_points"] > 100:
            new_vox = float(np.clip(vox * 0.80, 0.002, vox))
            if abs(new_vox - vox) / vox > 0.05:
                return {"vox": new_vox,
                        "sdf_trunc_multiplier": cfg["sdf_trunc_multiplier"]}
        if quality["density"] > 2_000_000:
            new_vox = float(np.clip(vox * 1.25, vox, 0.20))
            if abs(new_vox - vox) / vox > 0.05:
                return {"vox": new_vox,
                        "sdf_trunc_multiplier": cfg["sdf_trunc_multiplier"]}
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API  — drop-in replacement for AutoTuner v2.x
# ──────────────────────────────────────────────────────────────────────────────

class AutoTuner:
    """
    v3.0 — Drop-in replacement.  Same constructor kwargs, same cfg keys.

    New behaviour vs v2.x:
      • Reads .exr depth natively (float32 metres) — correct stats
      • Reads .conf mask — removes invalid LiDAR pixels
      • Parses inline intrinsics from poses.txt comments — exact fx/fy
      • Vox from LiDAR angular resolution — physics correct
      • No .png depth decoding hacks needed

    depth_scale is kept for API compatibility; for Record3D/ARKit EXR
    files it should remain 1.0 (EXR is already in metres).
    """

    def __init__(
        self,
        frames: list,
        rgb_dir: Path,
        depth_dir: Path,
        enable_verification: bool = True,
        output_dir: Optional[Path] = None,
        depth_scale: float = 1.0,
        verbose: bool = True,
    ):
        self.frames              = frames
        self.rgb_dir             = Path(rgb_dir)
        self.depth_dir           = Path(depth_dir)
        self.enable_verification = enable_verification
        self.output_dir          = Path(output_dir) if output_dir else Path(".")
        self.depth_scale         = depth_scale
        self.verbose             = verbose

    def compute(self) -> Dict[str, Any]:
        t0 = time.time()
        if self.verbose:
            log.info("  SmartAutoTuner v3.0: analyzing scene...")

        # Phase 1a — scene geometry from poses
        geo = SceneGeometryAnalyzer(self.frames).analyze()

        # Phase 1b — depth statistics from EXR/BIN (conf-masked)
        dep = DepthStatisticsAnalyzer(
            self.frames, self.depth_dir,
            depth_scale=self.depth_scale,
        ).analyze()

        # Phase 1c — image quality
        img = ImageQualityAnalyzer(self.frames, self.rgb_dir).analyze()

        # Phase 1d — exact intrinsics (from poses.txt comment if available)
        intr = IntrinsicsReader().read(self.frames)

        if self.verbose:
            log.info(
                f"  Geometry:   span={geo.span_m:.3f}m  traj={geo.trajectory_type}  "
                f"step={geo.median_step_m:.5f}m  frames={geo.n_frames}\n"
                f"  Depth:      p01={dep.p01:.3f}  p50={dep.p50:.3f}  p99={dep.p99:.3f}m  "
                f"grad_p95={dep.gradient_p95:.4f}  dynamic={dep.dynamic_score:.3f}\n"
                f"  Image:      sharpness={img.mean_sharpness:.1f}  "
                f"sky={img.has_sky}  blur_ratio={img.blur_ratio:.2f}\n"
                f"  Intrinsics: fx={intr['fx']:.2f}  w={intr['w']}  h={intr['h']}  "
                f"hfov={np.degrees(2*np.arctan(intr['w']/(2*intr['fx']))):.1f}°"
            )

        # Phase 2 — derive all parameters
        cfg = ParameterDeriver().derive(geo, dep, img, intrinsics=intr)
        cfg["approx_volume_m3"] = geo.approx_volume_m3

        # Phase 3 — mini TSDF verification + micro-correction
        if self.enable_verification:
            cfg = MiniVerifier(
                self.frames, self.output_dir, self.depth_scale
            ).verify_and_correct(cfg)

        cfg.pop("approx_volume_m3", None)
        log.info(
            f"  AutoTuner v3.0 done in {time.time()-t0:.1f}s  "
            f"scale={cfg['env_type']}  dynamic={cfg['has_dynamic_objects']}  "
            f"vox={cfg['vox']:.5f}m"
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Standalone diagnostic — run as:  python smart_tuner.py /path/to/scan
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, glob
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    log.info(f"Diagnostic mode — scan dir: {scan_dir}")

    # ── Build frame list from poses.txt ──────────────────────────────────────
    pose_file = (scan_dir / "poses.txt" if (scan_dir / "poses.txt").exists()
                 else next(scan_dir.glob("pose*.txt"), None))

    if pose_file:
        raw_frames = PosesFileParser().parse(pose_file)
        # Attach rgb/depth paths by stem index
        rgb_files = sorted(
            glob.glob(str(scan_dir / "rgb" / "*.jpg")) +
            glob.glob(str(scan_dir / "rgb" / "*.png"))
        )
        dep_files = sorted(glob.glob(str(scan_dir / "depth" / "*.exr")))
        for i, f in enumerate(raw_frames):
            f["rgb_path"]   = rgb_files[i] if i < len(rgb_files) else None
            f["depth_path"] = dep_files[i] if i < len(dep_files) else None
            f["_stem"]      = f"{i:05d}"
        frames = raw_frames
        log.info(f"  Loaded {len(frames)} frames from {pose_file.name}")
    else:
        log.warning("  No poses.txt found — using mock frames for smoke test")
        class _MockFrame:
            def __init__(self, i):
                rng = np.random.RandomState(i)
                self.c2w = np.eye(4); self.c2w[:3, 3] = rng.randn(3) * 0.05 * i
                self.intrinsics = np.array([[1351.96, 0, 957.69],
                                             [0, 1351.96, 721.78],
                                             [0, 0, 1.0]])
        frames = [_MockFrame(i) for i in range(50)]

    cfg = AutoTuner(
        frames              = frames,
        rgb_dir             = scan_dir / "rgb",
        depth_dir           = scan_dir / "depth",
        depth_scale         = 1.0,       # EXR is already in metres
        enable_verification = False,
        verbose             = True,
    ).compute()

    print("\n" + "=" * 65)
    print("DERIVED PARAMETERS  (v3.0 — physics correct, EXR/BIN/conf):")
    print("=" * 65)
    for k, v in sorted(cfg.items()):
        if isinstance(v, float):
            print(f"  {k:<45} = {v:.6f}")
        elif isinstance(v, np.ndarray):
            print(f"  {k:<45} = array{v.shape}")
        else:
            print(f"  {k:<45} = {v}")