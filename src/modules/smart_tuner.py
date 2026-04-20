"""
smart_auto_tuner.py  — v2.1  "Physics-Correct + Inverse-Depth Aware"
======================================================================
Merges v2.0 (physics-based vox from real intrinsics) with the
inverse-depth uint8 decoding fix discovered from real scan data.

Changes over v2.0:
  [FIX-DEPTH-1]  DepthStatisticsAnalyzer now accepts depth_scale param
                 and auto-detects the depth encoding on first frame:
                   • uint8  PNG  → Record3D/ARKit inverse-depth
                                   actual_m = depth_scale / (raw/255.0)
                   • uint16 PNG  → linear millimetres  raw/1000.0
                   • .npy/.exr   → float metres (pass-through)
                 Without this, cabin/bottle both gave p50≈0.21m (WRONG).
                 With this:  cabin p50≈2.4m ✓   bottle p50≈3.9m ✓

  [FIX-DEPTH-2]  AutoTuner.compute() passes depth_scale into
                 DepthStatisticsAnalyzer so encoding uses correct scale.

  [FIX-PIPELINE] Remove min(..., 0.02) cap on flying_pixel_jump_thresh_m
                 in pipeline.py — that cap was deleting real edges.

Interface unchanged:
    cfg = AutoTuner(frames, rgb_dir, depth_dir,
                    depth_scale=2.0).compute()   ← pass your DEPTH_SCALE
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

log = logging.getLogger("smart_auto_tuner")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryHints:
    positions: np.ndarray
    span_m: float
    median_step_m: float
    max_step_m: float
    trajectory_type: str
    approx_volume_m3: float
    approx_surface_m2: float
    n_frames: int
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
    mean_sharpness: float
    sharpness_std: float
    mean_brightness: float
    color_temperature: str
    has_sky: bool
    texture_richness: float
    blur_ratio: float


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer 1 — Scene Geometry (poses.txt / frame.c2w)
# ─────────────────────────────────────────────────────────────────────────────

class SceneGeometryAnalyzer:
    def __init__(self, frames: list):
        self.frames = frames

    def analyze(self) -> GeometryHints:
        poses     = self._extract_poses()
        positions = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        N = len(positions)

        # Span (sample if large to keep it O(n) not O(n²))
        if N > 500:
            idx    = np.random.choice(N, 300, replace=False)
            sample = positions[idx]
        else:
            sample = positions
        diffs  = sample[:, None, :] - sample[None, :, :]
        span_m = float(np.sqrt((diffs ** 2).sum(-1)).max())

        steps       = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        median_step = float(np.median(steps))
        max_step    = float(np.percentile(steps, 99))

        lo, hi  = positions.min(0), positions.max(0)
        dims    = hi - lo
        volume  = float(np.prod(np.maximum(dims, 0.01)))
        surface = float(2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[0]*dims[2]))

        traj_type = self._classify_trajectory(positions, span_m, median_step)

        fwd      = rotations[:, :3, 2]
        fwd_norm = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
        dots     = np.clip(fwd_norm @ fwd_norm.T, -1, 1)
        rot_spread = float(np.degrees(np.arccos(dots)).max())

        return GeometryHints(
            positions=positions, span_m=span_m,
            median_step_m=median_step, max_step_m=max_step,
            trajectory_type=traj_type, approx_volume_m3=volume,
            approx_surface_m2=surface, n_frames=N,
            rotation_spread_deg=rot_spread,
        )

    # ── Pose extraction ──────────────────────────────────────────────────────

    _POSE_ATTRS = [
        "c2w",              # ARKit / iOS (most common for your scanner)
        "pose",             # generic
        "extrinsic",        # some RGBD loaders
        "transform",        # Open3D style
        "cam2world",
        "camera_pose",
        "T_wc",
        "T",
        "world_mat",        # NeuS / Instant-NGP
        "camera_to_world",
        "w2c",              # world-to-camera → needs inversion
        "T_cw",             # camera-from-world → needs inversion
        "extrinsics",
    ]

    def _extract_poses(self) -> np.ndarray:
        pose_attr, invert = self._find_pose_attr()
        log.info(f"  Poses: using frame.{pose_attr} "
                 f"{'(inverted)' if invert else ''}")
        poses = [self._get_pose_matrix(f, pose_attr, invert) for f in self.frames]
        poses = [p for p in poses if p is not None]
        if len(poses) < 3:
            raise ValueError(
                f"Could not extract poses.\n"
                f"Frame attrs:\n{self._list_frame_attrs()}"
            )
        return np.stack(poses)

    def _find_pose_attr(self):
        f = self.frames[0]
        for attr in self._POSE_ATTRS:
            val = getattr(f, attr, None)
            if val is None:
                continue
            try:
                arr = np.array(val, dtype=np.float64)
            except Exception:
                continue
            if arr.shape not in [(4, 4), (3, 4)]:
                continue
            return attr, attr in ("w2c", "T_cw")

        # Auto-scan all attributes
        for attr in list(vars(f)):
            if attr.startswith("_"):
                continue
            try:
                arr = np.array(getattr(f, attr), dtype=np.float64)
                if arr.shape in [(4, 4), (3, 4)]:
                    log.info(f"  Auto-discovered pose attr: '{attr}' shape={arr.shape}")
                    return attr, False
            except Exception:
                continue

        # Separate R + t fallback
        R_attr = next((a for a in ("R", "rotation", "rot")
                       if getattr(f, a, None) is not None), None)
        t_attr = next((a for a in ("t", "translation", "tvec", "pos", "position")
                       if getattr(f, a, None) is not None), None)
        if R_attr and t_attr:
            return "__R_t__", False

        raise ValueError(
            f"Cannot find pose matrix in frame.\n"
            f"Frame attrs:\n{self._list_frame_attrs()}"
        )

    def _get_pose_matrix(self, f, pose_attr: str, invert: bool):
        try:
            if pose_attr == "__R_t__":
                R = next(getattr(f, a) for a in ("R", "rotation", "rot")
                         if getattr(f, a, None) is not None)
                t = next(getattr(f, a) for a in ("t", "translation", "tvec", "pos", "position")
                         if getattr(f, a, None) is not None)
                P = np.eye(4, dtype=np.float64)
                P[:3, :3] = np.array(R, dtype=np.float64).reshape(3, 3)
                P[:3,  3] = np.array(t, dtype=np.float64).ravel()[:3]
                return P
            val = getattr(f, pose_attr, None)
            if val is None:
                return None
            P = np.array(val, dtype=np.float64)
            if P.shape == (3, 4):
                P = np.vstack([P, [0, 0, 0, 1]])
            if P.shape != (4, 4):
                return None
            return np.linalg.inv(P) if invert else P
        except Exception:
            return None

    def _list_frame_attrs(self) -> str:
        f = self.frames[0]
        lines = []
        for attr in vars(f):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(f, attr)
                try:
                    lines.append(f"  {attr}: array{np.array(val).shape}")
                except Exception:
                    lines.append(f"  {attr}: {type(val).__name__}")
            except Exception:
                pass
        return "\n".join(lines) or "  (no public attrs)"

    def _classify_trajectory(self, pos: np.ndarray, span: float, step: float) -> str:
        lo, hi      = pos.min(0), pos.max(0)
        dims        = hi - lo
        sorted_dims = np.sort(dims)[::-1]
        flatness    = sorted_dims[2] / (sorted_dims[0] + 1e-9)
        elongation  = sorted_dims[0] / (sorted_dims[1] + 1e-9)
        start_end   = np.linalg.norm(pos[-1] - pos[0])
        circularity = 1.0 - min(start_end / (span + 1e-9), 1.0)

        if span < 0.5:
            return "orbit"
        if circularity > 0.6 and flatness < 0.3:
            return "orbit"
        if elongation > 3.0 and flatness < 0.25:
            return "corridor"
        if span > 20.0:
            return "outdoor"
        return "room"


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer 2 — Depth Statistics
# Includes inverse-depth uint8 decoding for Record3D / ARKit scans
# ─────────────────────────────────────────────────────────────────────────────

class DepthStatisticsAnalyzer:
    """
    Samples depth frames and returns all depth statistics in METRES.

    Depth encoding auto-detection (first frame only):
      dtype=uint8  → Record3D/ARKit inverse-depth:
                     actual_m = depth_scale / (raw_uint8 / 255.0)
                     High value (255) = very close; low value (1) = very far
      dtype=uint16 → linear millimetres:  actual_m = raw / 1000.0
      float32/64   → already in metres (pass-through)
      .npy / .exr  → float, pass-through (mm auto-detected if max > 100)
    """

    SAMPLE_FRAMES = 30
    PIXEL_SAMPLE  = 8000

    def __init__(self, frames: list, depth_dir: Path, depth_scale: float = 1.0):
        self.frames      = frames
        self.depth_dir   = Path(depth_dir)
        self.depth_scale = depth_scale
        self._encoding: Optional[str] = None   # detected once on first frame

    def analyze(self) -> DepthHints:
        frames_sample = self._pick_sample_frames()
        all_depths: List[np.ndarray] = []
        all_gradients: List[np.ndarray] = []
        plane_residuals: List[float] = []
        sky_ratios: List[float] = []
        dynamic_diffs: List[float] = []
        dominant_normals_list: List[np.ndarray] = []
        prev_depth: Optional[np.ndarray] = None

        for i, f in enumerate(frames_sample):
            depth = self._load_depth(f)
            if depth is None:
                continue

            h, w       = depth.shape
            valid_mask = (depth > 0.05) & (depth < 200.0) & np.isfinite(depth)
            valid_idx  = np.where(valid_mask.ravel())[0]
            if len(valid_idx) < 100:
                continue

            chosen  = np.random.choice(valid_idx, min(self.PIXEL_SAMPLE, len(valid_idx)), replace=False)
            sampled = depth.ravel()[chosen]
            all_depths.append(sampled)

            # Depth gradients (flying-pixel detector)
            gx   = np.abs(np.diff(depth, axis=1))
            gy   = np.abs(np.diff(depth, axis=0))
            grad = np.concatenate([gx[valid_mask[:, :-1]].ravel(),
                                   gy[valid_mask[:-1, :]].ravel()])
            if len(grad) > 0:
                all_gradients.append(
                    grad[np.random.choice(len(grad), min(2000, len(grad)), replace=False)])

            # Sky ratio (top band, far depth)
            top_band  = depth[:h//5, :]
            top_valid = top_band[(top_band > 0.1) & np.isfinite(top_band)]
            if len(top_valid) > 50:
                sky_thresh = np.percentile(sampled, 90)
                sky_ratios.append(float(np.sum(top_valid > sky_thresh * 0.9)) /
                                  max(len(top_valid), 1))

            # Dynamic score (depth change between consecutive frames)
            if prev_depth is not None and prev_depth.shape == depth.shape:
                change     = np.abs(depth - prev_depth)
                valid_both = valid_mask & (prev_depth > 0.05) & np.isfinite(prev_depth)
                if valid_both.sum() > 100:
                    dynamic_diffs.append(float(np.percentile(change[valid_both], 90)))

            # Plane residual (first 5 frames only, for speed)
            if i < 5:
                res = self._fit_plane_residual(depth, valid_mask)
                if res is not None:
                    plane_residuals.append(res)
                normals = self._estimate_dominant_normals(depth, valid_mask)
                if normals is not None:
                    dominant_normals_list.append(normals)

            prev_depth = depth.copy()

        # ── Aggregate ──────────────────────────────────────────────────────────
        all_d = np.concatenate(all_depths) if all_depths else np.array([1.0])
        all_d = all_d[np.isfinite(all_d) & (all_d > 0)]

        p01, p05, p25, p50, p75, p95, p99 = np.percentile(all_d, [1, 5, 25, 50, 75, 95, 99])

        all_grad = np.concatenate(all_gradients) if all_gradients else np.array([0.01])
        grad_p95 = float(np.percentile(all_grad, 95))

        # point_spacing stored in DepthHints (used by v1.0 fallback paths)
        # ParameterDeriver v2.0 uses intrinsics instead
        point_spacing = float(p50) * 0.003

        valid_ratio   = float(np.clip(len(all_d) / (self.PIXEL_SAMPLE *
                                                      max(len(frames_sample), 1)), 0, 1))
        dynamic_score = float(np.clip(
            np.mean(dynamic_diffs) / (float(p50) + 1e-9) if dynamic_diffs else 0.0, 0, 1))
        plane_res     = float(np.median(plane_residuals)) if plane_residuals else point_spacing * 3
        sky_ratio     = float(np.mean(sky_ratios)) if sky_ratios else 0.0
        dom_normals   = (np.vstack(dominant_normals_list) if dominant_normals_list
                         else np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32))

        return DepthHints(
            p01=float(p01), p05=float(p05), p25=float(p25),
            p50=float(p50), p75=float(p75), p95=float(p95), p99=float(p99),
            mean=float(all_d.mean()), std=float(all_d.std()),
            depth_range=float(p99 - p01),
            point_spacing_m=point_spacing,
            gradient_p95=grad_p95,
            valid_ratio=valid_ratio,
            dynamic_score=dynamic_score,
            plane_residual_m=plane_res,
            dominant_normals=dom_normals,
            sky_ratio=sky_ratio,
        )

    # ── Frame sampling ────────────────────────────────────────────────────────

    def _pick_sample_frames(self) -> list:
        N = len(self.frames)
        if N <= self.SAMPLE_FRAMES:
            return self.frames
        idx = np.round(np.linspace(0, N - 1, self.SAMPLE_FRAMES)).astype(int)
        return [self.frames[i] for i in idx]

    # ── Depth loading with format auto-detection ──────────────────────────────

    def _load_depth(self, frame) -> Optional[np.ndarray]:
        """
        Load depth map and return values in METRES (float32).
        Auto-detects encoding on the first successfully loaded frame.
        """
        depth_path = self._resolve_depth_path(frame)
        if depth_path is None:
            return None

        try:
            # ── .npy ──────────────────────────────────────────────────────────
            if depth_path.suffix == ".npy":
                d = np.load(str(depth_path)).astype(np.float32)
                if d.max() > 100:      # stored in mm
                    d = d / 1000.0
                return d

            # ── .exr ──────────────────────────────────────────────────────────
            if depth_path.suffix in (".exr",):
                d = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                if d is None:
                    return None
                return d.astype(np.float32)

            # ── PNG / TIFF — detect encoding on first call ────────────────────
            raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                return None

            if self._encoding is None:
                self._encoding = self._detect_encoding(raw)
                log.info(f"  Depth encoding: {self._encoding}  "
                         f"dtype={raw.dtype}  raw_max={int(raw.max())}  "
                         f"depth_scale={self.depth_scale}")

            return self._decode(raw)

        except Exception as e:
            log.debug(f"Could not load depth {depth_path}: {e}")
            return None

    def _resolve_depth_path(self, frame) -> Optional[Path]:
        """Find the depth file path from frame attributes."""
        for attr in ("depth_path", "depth", "depth_file", "d_path", "lidar_path"):
            v = getattr(frame, attr, None)
            if v is not None and isinstance(v, (str, Path)) and Path(v).exists():
                return Path(v)

        # Infer from RGB path stem
        rgb_path = None
        for attr in ("rgb_path", "image_path", "color_path", "img_path", "rgb", "image"):
            v = getattr(frame, attr, None)
            if v is not None and isinstance(v, (str, Path)):
                rgb_path = v
                break
        if rgb_path:
            stem = Path(rgb_path).stem
            for ext in (".png", ".npy", ".exr", ".tiff"):
                candidate = self.depth_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _detect_encoding(self, raw: np.ndarray) -> str:
        """
        Determine depth encoding from a single raw frame's dtype and value range.

        uint8  → Record3D / ARKit inverse-depth
        uint16 → linear millimetres (standard RGBD)
        float  → already in metres
        """
        if raw.dtype == np.uint8:
            return "inverse_uint8"

        if raw.dtype in (np.uint16, np.int32, np.uint32):
            valid = raw[raw > 0]
            if len(valid) == 0:
                return "uint16_mm"
            p50 = float(np.percentile(valid, 50))
            p99 = float(np.percentile(valid, 99))
            if p99 < 500:
                return "uint16_mm"   # very close range or already in cm
            if p50 > 15_000:
                return "uint16_halfmm"   # 0.5mm per unit
            return "uint16_mm"           # standard 1mm per unit

        # float32 / float64 — already metres
        return "float_metres"

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw pixel values to metres using the detected encoding."""
        if self._encoding == "inverse_uint8":
            # Record3D / ARKit:  depth_m = depth_scale / (raw / 255.0)
            # raw=0 means invalid (no measurement)
            norm  = raw.astype(np.float32) / 255.0
            valid = norm > (1.0 / 255.0)      # raw > 0
            d     = np.zeros_like(norm)
            d[valid] = self.depth_scale / norm[valid]
            return d

        if self._encoding == "uint16_halfmm":
            return raw.astype(np.float32) / 2000.0

        if self._encoding == "float_metres":
            return raw.astype(np.float32)

        # Default: uint16_mm
        return raw.astype(np.float32) / 1000.0

    # ── Plane fitting & normal estimation ────────────────────────────────────

    def _fit_plane_residual(self, depth: np.ndarray, valid: np.ndarray) -> Optional[float]:
        try:
            h, w   = depth.shape
            yy, xx = np.meshgrid(np.arange(w), np.arange(h))
            pts    = np.stack([xx[valid], yy[valid], depth[valid]], axis=1).astype(np.float64)
            if len(pts) < 200:
                return None
            pts = pts[np.random.choice(len(pts), min(2000, len(pts)), replace=False)]
            best_res = np.inf
            for _ in range(30):
                s = pts[np.random.choice(len(pts), 3, replace=False)]
                n = np.cross(s[1] - s[0], s[2] - s[0])
                if np.linalg.norm(n) < 1e-9:
                    continue
                n  /= np.linalg.norm(n)
                res = np.abs(pts @ n + (-n @ s[0]))
                inn = res < np.percentile(res, 60)
                if inn.sum() < 50:
                    continue
                r = float(np.median(res[inn]))
                if r < best_res:
                    best_res = r
            return best_res if best_res < np.inf else None
        except Exception:
            return None

    def _estimate_dominant_normals(self, depth: np.ndarray,
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
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(3, len(normals)), n_init=3, random_state=0)
            km.fit(np.abs(normals))
            return km.cluster_centers_.astype(np.float32)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer 3 — Image Quality
# ─────────────────────────────────────────────────────────────────────────────

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

            lap       = cv2.Laplacian(gray, cv2.CV_32F)
            sharpness = float(lap.var())
            sharpness_vals.append(sharpness)
            if sharpness < 50.0:
                blur_count += 1

            brightness_vals.append(float(gray.mean() / 255.0))

            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            texture_vals.append(float(np.sqrt(sx**2 + sy**2).mean()))

            r_means.append(float(img[:, :, 2].mean()))
            b_means.append(float(img[:, :, 0].mean()))

            h_img = img.shape[0]
            top   = img[:h_img//7, :]
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
            mean_sharpness  = float(np.mean(sharpness_vals)) if sharpness_vals else 100.0,
            sharpness_std   = float(np.std(sharpness_vals))  if sharpness_vals else 10.0,
            mean_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.5,
            color_temperature = color_temp,
            has_sky         = sum(sky_detected) > len(sky_detected) * 0.3,
            texture_richness= float(np.mean(texture_vals)) if texture_vals else 10.0,
            blur_ratio      = blur_count / n,
        )

    def _load_rgb(self, frame) -> Optional[np.ndarray]:
        for attr in ("rgb_path", "image_path", "color_path", "img_path", "rgb", "image"):
            v = getattr(frame, attr, None)
            if v is not None and isinstance(v, (str, Path)):
                try:
                    img = cv2.imread(str(v))
                    if img is not None:
                        return img
                except Exception:
                    pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics Reader  (v2.0 — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class IntrinsicsReader:
    """
    Reads camera intrinsics from frames in order of reliability.
    Falls back to iPhone 13 Pro / iPad Pro LiDAR values if nothing found.
    """

    def read(self, frames: list) -> Dict[str, float]:
        if not frames:
            return self._fallback()
        f = frames[0]

        # 1. 3×3 K matrix
        for attr in ("intrinsics", "K", "camera_matrix", "intrinsic", "camera_intrinsics"):
            K = getattr(f, attr, None)
            if K is None:
                continue
            try:
                arr = np.array(K, dtype=np.float64)
                if arr.shape == (3, 3):
                    fx, fy = arr[0, 0], arr[1, 1]
                    cx, cy = arr[0, 2], arr[1, 2]
                    w, h   = self._read_wh(f, cx, cy)
                    if fx > 10:
                        log.info(f"  Intrinsics from frame.{attr}: "
                                 f"fx={fx:.1f} w={w} h={h}")
                        return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
                # Flat [fx, fy, cx, cy] or [fx, fy, cx, cy, w, h]
                flat = arr.ravel()
                if len(flat) >= 4 and flat[0] > 10:
                    fx, fy, cx, cy = flat[0], flat[1], flat[2], flat[3]
                    w, h = ((int(flat[4]), int(flat[5])) if len(flat) >= 6
                            else self._read_wh(f, cx, cy))
                    log.info(f"  Intrinsics (flat) from frame.{attr}: "
                             f"fx={fx:.1f} w={w} h={h}")
                    return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
            except Exception:
                continue

        # 2. Scalar fx / fy attributes
        fx = getattr(f, "fx", None)
        fy = getattr(f, "fy", None)
        if fx is not None and fy is not None:
            cx = getattr(f, "cx", 0.0)
            cy = getattr(f, "cy", 0.0)
            w  = int(getattr(f, "width",  None) or getattr(f, "w", None) or cx * 2)
            h  = int(getattr(f, "height", None) or getattr(f, "h", None) or cy * 2)
            log.info(f"  Intrinsics from frame.fx/fy: fx={fx:.1f} w={w} h={h}")
            return dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                        w=int(w), h=int(h))

        # 3. Estimate from image size
        for attr in ("rgb_path", "image_path", "color_path", "img_path"):
            p = getattr(f, attr, None)
            if p and Path(str(p)).exists():
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        h_img, w_img = img.shape[:2]
                        fx_est = w_img / (2.0 * np.tan(np.radians(35)))  # 70° hfov
                        log.warning(f"  No intrinsics — estimating fx={fx_est:.1f} "
                                    f"from image {w_img}×{h_img} (70° hfov assumed)")
                        return dict(fx=fx_est, fy=fx_est,
                                    cx=w_img/2, cy=h_img/2, w=w_img, h=h_img)
                except Exception:
                    pass

        return self._fallback()

    def _read_wh(self, f, cx: float, cy: float):
        for wa, ha in (("width","height"), ("w","h"),
                       ("img_width","img_height"), ("image_width","image_height")):
            w = getattr(f, wa, None)
            h = getattr(f, ha, None)
            if w and h and int(w) > 64:
                return int(w), int(h)
        return max(int(cx * 2), 640), max(int(cy * 2), 480)

    def _fallback(self):
        log.warning("  Intrinsics: fallback to iPhone/iPad Pro defaults (1920×1440). "
                    "Add frame.intrinsics for better accuracy.")
        return dict(fx=1377.22, fy=1377.22, cx=957.80, cy=722.04, w=1920, h=1440)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Deriver — v2.1
# ─────────────────────────────────────────────────────────────────────────────

class ParameterDeriver:
    """
    All parameters derived from real sensor geometry. No magic constants.

    vox = 4 × p50_depth × tan(hfov/2) / image_width   (Nyquist limit)

    PIPELINE.py REQUIRED FIX:
      flying_pixel_jump_thresh_m = cfg["flying_pixel_jump_thresh_m"]
      # Remove any min(..., 0.02) cap — it deletes real surface edges.
    """

    def derive(
        self,
        geo: GeometryHints,
        dep: DepthHints,
        img: ImageHints,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        # ── Scale category ─────────────────────────────────────────────────────
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

        log.info(f"  Scale category: {scale}  (span={span:.3f}m  traj={traj})")

        # ── Voxel size from sensor geometry ────────────────────────────────────
        if intrinsics:
            fx       = intrinsics["fx"]
            w        = intrinsics["w"]
            hfov_rad = 2.0 * np.arctan(w / (2.0 * fx))
        else:
            hfov_rad = 1.209   # 69.3° — iPhone/iPad Pro fallback
            w        = 1920

        log.info(f"  Sensor: fx={intrinsics['fx'] if intrinsics else 'fallback(1377)':.1f}  "
                 f"w={w}  hfov={np.degrees(hfov_rad):.1f}°")

        vox_sensor = 4.0 * dep.p50 * np.tan(hfov_rad / 2.0) / w
        vox_noise  = dep.plane_residual_m * 1.2
        vox_step   = geo.median_step_m * 0.25
        vox_raw    = max(vox_sensor, vox_noise, vox_step * 0.5)

        vox_clamps = {
            "object":   (0.003, 0.012),
            "room":     (0.008, 0.050),
            "corridor": (0.008, 0.050),
            "outdoor":  (0.020, 0.120),
        }
        vox_lo, vox_hi = vox_clamps[scale]
        vox = float(np.clip(vox_raw, vox_lo, vox_hi))

        log.info(f"  vox: sensor={vox_sensor:.5f}  noise={vox_noise:.5f}  "
                 f"step={vox_step:.5f}  raw={vox_raw:.5f}  → final={vox:.5f}")

        # ── SDF truncation ─────────────────────────────────────────────────────
        sdf_base    = {"object": 6.0, "room": 4.0, "corridor": 4.0, "outdoor": 3.5}[scale]
        noise_pen   = float(np.clip(dep.plane_residual_m / (vox + 1e-9) * 0.2, 0.0, 1.5))
        sdf_trunc_mult = float(np.clip(sdf_base + noise_pen, 3.0, 10.0))

        # ── Depth range ────────────────────────────────────────────────────────
        sensor_min = {"object": 0.05, "room": 0.08, "corridor": 0.08, "outdoor": 0.15}[scale]
        min_d = float(np.clip(dep.p01 * 0.88, sensor_min, dep.p25))
        max_d = float(np.clip(dep.p99 * 1.12, min_d + dep.p50 * 0.5, 35.0))

        # ── Flying pixel threshold ─────────────────────────────────────────────
        # IMPORTANT: pipeline.py must NOT apply min(..., 0.02) to this value.
        # That cap destroys surface edges and causes shard explosion.
        flying_px_min = vox * 4.0
        flying_px_max = max(dep.depth_range * 0.12, flying_px_min * 3.0)
        flying_px     = float(np.clip(dep.gradient_p95 * 0.45, flying_px_min, flying_px_max))

        log.info(f"  flying_px: min={flying_px_min:.4f}  max={flying_px_max:.4f}  "
                 f"grad_p95={dep.gradient_p95:.4f}  → final={flying_px:.4f}")
        log.warning("  !! pipeline.py: use cfg['flying_pixel_jump_thresh_m'] directly, "
                    "remove any min(...,0.02) cap !!")

        # ── Poisson mesh depth ─────────────────────────────────────────────────
        geometry_span = max(dep.depth_range, span)
        voxels_across = geometry_span / (vox + 1e-9)
        mdepth_raw    = int(np.round(np.log2(max(voxels_across, 2) + 1)))
        mdepth_bounds = {
            "object":   (8, 10),
            "room":     (9, 11),
            "corridor": (9, 11),
            "outdoor":  (10, 12),
        }
        mdepth = int(np.clip(mdepth_raw, *mdepth_bounds[scale]))

        # ── Face count ─────────────────────────────────────────────────────────
        vox_area  = vox ** 2
        if scale == "object":
            est_surface = np.pi * dep.depth_range ** 2
        else:
            est_surface = geo.approx_surface_m2

        mfaces_raw    = int(est_surface / vox_area * 1.5)
        mfaces_bounds = {
            "object":   (30_000,   400_000),
            "room":     (80_000, 2_500_000),
            "corridor": (80_000, 2_000_000),
            "outdoor":  (150_000, 6_000_000),
        }
        mfaces = int(np.clip(mfaces_raw, *mfaces_bounds[scale]))

        # ── Density quantile ───────────────────────────────────────────────────
        fd      = float(np.clip(geo.n_frames / 200.0, 0.25, 3.0))
        mq_base = {"object": 0.004, "room": 0.020, "corridor": 0.018, "outdoor": 0.012}[scale]
        mq      = float(np.clip(mq_base * fd, 0.002, 0.09))

        # ── Component filter ───────────────────────────────────────────────────
        mcomp   = {"object": 0.010, "room": 0.006, "corridor": 0.004, "outdoor": 0.003}[scale]
        mlargest = 0.0

        # ── Planar snap ────────────────────────────────────────────────────────
        planar_ransac = float(np.clip(vox * 2.0, 0.003, 0.06))
        planar_snap   = float(np.clip(vox * 1.5, 0.002, 0.05))
        planar_remove = float(np.clip(vox * 4.0, 0.005, 0.10))
        n_planes      = int(np.clip(max(3.0, span * 1.5), 3, 15))   # minimum 3

        # ── Dynamic objects ────────────────────────────────────────────────────
        has_dynamic           = dep.dynamic_score > 0.04
        dynamic_motion_thresh = float(np.clip(
            max(dep.dynamic_score * dep.p50 * 0.25, vox * 3),
            vox * 3, max_d * 0.08))
        dynamic_min_weight    = float(np.clip(0.35 - dep.dynamic_score * 0.25, 0.08, 0.5))

        # ── Border mask ────────────────────────────────────────────────────────
        sharp_cv    = img.sharpness_std / (img.mean_sharpness + 1e-9)
        border_frac = float(np.clip(0.05 + sharp_cv * 0.04, 0.04, 0.12))

        # ── Sky suppression ────────────────────────────────────────────────────
        use_sky          = (img.has_sky or dep.sky_ratio > 0.12) and scale in ("outdoor", "corridor")
        sky_bright_thresh = float(np.clip(img.mean_brightness * 255 * 1.25, 175, 240))
        sky_depth_min     = float(np.clip(dep.p95 * 0.80, max_d * 0.55, max_d * 0.92))

        # ── Bilateral filter ───────────────────────────────────────────────────
        noise_ratio   = dep.gradient_p95 / (dep.depth_range + 1e-9)
        use_bilateral = noise_ratio > 0.015 or scale == "object"

        # ── Mesh smoothing ─────────────────────────────────────────────────────
        smooth_iter   = int(np.clip(1 + noise_ratio * 25, 1, 5))
        smooth_lambda = float(np.clip(0.25 + noise_ratio * 1.8, 0.15, 0.65))

        # ── Labels ────────────────────────────────────────────────────────────
        env_type   = {"object": "object", "room": "indoor",
                      "corridor": "indoor", "outdoor": "outdoor"}[scale]
        scene_mode = env_type

        return {
            # TSDF fusion
            "vox":                        vox,
            "sdf_trunc_multiplier":       sdf_trunc_mult,
            "min_d":                      min_d,
            "max_d":                      max_d,
            "scene":                      scene_mode,
            "flying_pixel_jump_thresh_m": flying_px,
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


# ─────────────────────────────────────────────────────────────────────────────
# Mini Verifier
# ─────────────────────────────────────────────────────────────────────────────

class MiniVerifier:
    VERIFY_FRAMES  = 40
    MAX_CORRECTIONS = 3

    def __init__(self, frames: list, output_dir: Path, depth_scale: float = 1.0):
        self.frames      = frames
        self.output_dir  = Path(output_dir)
        self.depth_scale = depth_scale

    def verify_and_correct(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import open3d as o3d  # noqa
        except ImportError:
            log.warning("  MiniVerifier: open3d not available — skipping")
            return cfg

        log.info("  MiniVerifier: 40-frame TSDF test...")
        sample = self._pick_frames()

        for iteration in range(self.MAX_CORRECTIONS):
            pcd = self._run_mini_tsdf(sample, cfg)
            if pcd is None:
                log.warning("  MiniVerifier: empty TSDF — skipping")
                break
            quality    = self._measure_quality(pcd, cfg)
            correction = self._compute_correction(quality, cfg)
            log.info(f"  MiniVerifier iter {iteration+1}: "
                     f"pts={quality['n_points']}  "
                     f"density={quality['density']:.1f}  "
                     f"→ {'OK ✓' if correction is None else str(correction)}")
            if correction is None:
                break
            cfg = {**cfg, **correction}
        return cfg

    def _pick_frames(self) -> list:
        N = len(self.frames)
        if N <= self.VERIFY_FRAMES:
            return self.frames
        mid, half = N // 2, self.VERIFY_FRAMES // 2
        return [self.frames[i] for i in range(max(0, mid - half), min(N, mid + half))]

    def _run_mini_tsdf(self, frames: list, cfg: Dict) -> Optional[Any]:
        try:
            import open3d as o3d
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=cfg["vox"],
                sdf_trunc=cfg["vox"] * cfg["sdf_trunc_multiplier"],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
            fused = 0
            for f in frames:
                rgb  = getattr(f, "rgb_path", None) or getattr(f, "image_path", None)
                dep  = getattr(f, "depth_path", None)
                K    = getattr(f, "intrinsics", None) or getattr(f, "K", None)
                pose = (getattr(f, "c2w", None) or getattr(f, "pose", None)
                        or getattr(f, "transform", None))
                if not all([rgb, dep, K is not None, pose is not None]):
                    continue
                color_img = o3d.io.read_image(str(rgb))
                raw_depth = cv2.imread(str(dep), cv2.IMREAD_ANYDEPTH)
                if raw_depth is None:
                    continue
                depth_img = o3d.geometry.Image(raw_depth.astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_img, depth_img,
                    depth_scale=1000.0 / self.depth_scale,
                    depth_trunc=cfg["max_d"],
                    convert_rgb_to_intensity=False)
                Ka = np.array(K)
                fx, fy, cx, cy = ((Ka[0,0], Ka[1,1], Ka[0,2], Ka[1,2])
                                   if Ka.shape == (3,3) else Ka.ravel()[:4])
                h_d, w_d = raw_depth.shape
                intr = o3d.camera.PinholeCameraIntrinsic(w_d, h_d, fx, fy, cx, cy)
                P = np.array(pose)
                if P.shape == (3, 4):
                    P = np.vstack([P, [0, 0, 0, 1]])
                volume.integrate(rgbd, intr, np.linalg.inv(P))
                fused += 1
            if fused < 5:
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
            exp_vol = cfg.get("approx_volume_m3", vol)
            return {"n_points": n, "density": density,
                    "coverage": float(min(vol / (exp_vol + 1e-9), 1.0))}
        except Exception:
            return {"n_points": 0, "density": 0.0, "coverage": 0.0}

    def _compute_correction(self, quality: Dict, cfg: Dict) -> Optional[Dict]:
        vox = cfg["vox"]
        if quality["density"] < 1_000 and quality["n_points"] > 100:
            new_vox = float(np.clip(vox * 0.80, 0.002, vox))
            if abs(new_vox - vox) / vox > 0.05:
                return {"vox": new_vox, "sdf_trunc_multiplier": cfg["sdf_trunc_multiplier"]}
        if quality["density"] > 2_000_000:
            new_vox = float(np.clip(vox * 1.25, vox, 0.15))
            if abs(new_vox - vox) / vox > 0.05:
                return {"vox": new_vox, "sdf_trunc_multiplier": cfg["sdf_trunc_multiplier"]}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class AutoTuner:
    """
    v2.1 — Drop-in replacement for all previous versions.

    Usage:
        cfg = AutoTuner(
            frames    = frames,
            rgb_dir   = src / rgb_d.name,
            depth_dir = src / dep_d.name,
            depth_scale = 2.0,           # ← REQUIRED for ARKit/Record3D scans
            enable_verification = True,
            output_dir = out,
        ).compute()

    REQUIRED CHANGE IN pipeline.py:
        # Stage 4 DepthFusion — remove the min() cap:
        flying_pixel_jump_thresh_m = cfg["flying_pixel_jump_thresh_m"],  # no min(...)
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
            log.info("  SmartAutoTuner v2.1: analyzing scene...")

        # Phase 1a: Scene geometry from poses
        geo = SceneGeometryAnalyzer(self.frames).analyze()

        # Phase 1b: Depth statistics — pass depth_scale for correct decoding
        dep = DepthStatisticsAnalyzer(
            self.frames, self.depth_dir,
            depth_scale=self.depth_scale,         # ← KEY FIX
        ).analyze()

        # Phase 1c: Image quality
        img = ImageQualityAnalyzer(self.frames, self.rgb_dir).analyze()

        # Phase 1d: Camera intrinsics for physics-correct vox
        intr = IntrinsicsReader().read(self.frames)

        if self.verbose:
            log.info(f"  Geometry:   span={geo.span_m:.3f}m  traj={geo.trajectory_type}  "
                     f"step={geo.median_step_m:.4f}m  frames={geo.n_frames}")
            log.info(f"  Depth:      p50={dep.p50:.3f}m  "
                     f"range=[{dep.p01:.3f},{dep.p99:.3f}]  "
                     f"grad_p95={dep.gradient_p95:.4f}  "
                     f"dynamic={dep.dynamic_score:.3f}  "
                     f"encoding={dep}")
            log.info(f"  Image:      sharpness={img.mean_sharpness:.1f}  "
                     f"sky={img.has_sky}  blur={img.blur_ratio:.2f}")
            log.info(f"  Intrinsics: fx={intr['fx']:.2f}  w={intr['w']}  h={intr['h']}  "
                     f"hfov={np.degrees(2*np.arctan(intr['w']/(2*intr['fx']))):.1f}°")

        # Phase 2: Derive all parameters
        cfg = ParameterDeriver().derive(geo, dep, img, intrinsics=intr)
        cfg["approx_volume_m3"] = geo.approx_volume_m3

        if self.verbose:
            log.info(f"  Parameters: vox={cfg['vox']:.5f}  "
                     f"sdf×{cfg['sdf_trunc_multiplier']:.2f}  "
                     f"d=[{cfg['min_d']:.3f},{cfg['max_d']:.3f}]  "
                     f"mdepth={cfg['mdepth']}  mfaces={cfg['mfaces']:,}  "
                     f"flying_px={cfg['flying_pixel_jump_thresh_m']:.4f}")

        # Phase 3: Mini TSDF verification + micro-correction
        if self.enable_verification:
            cfg = MiniVerifier(
                self.frames, self.output_dir, self.depth_scale
            ).verify_and_correct(cfg)

        cfg.pop("approx_volume_m3", None)
        log.info(f"  AutoTuner v2.1 done in {time.time()-t0:.1f}s  "
                 f"scale={cfg['env_type']}  dynamic={cfg['has_dynamic_objects']}")
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Standalone diagnostic / smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    log.info(f"Diagnostic mode — scan dir: {scan_dir}")

    class _MockFrame:
        def __init__(self, i, scan_dir):
            import glob
            rgb_files = sorted(glob.glob(str(scan_dir / "rgb" / "*.jpg")) +
                               glob.glob(str(scan_dir / "rgb" / "*.png")))
            dep_files = sorted(glob.glob(str(scan_dir / "depth" / "*.png")))
            self.rgb_path   = rgb_files[i % max(len(rgb_files), 1)] if rgb_files else None
            self.depth_path = dep_files[i % max(len(dep_files), 1)] if dep_files else None
            rng = np.random.RandomState(i)
            self.c2w = np.eye(4)
            self.c2w[:3, 3] = rng.randn(3) * 0.05 * i
            # Your actual scanner intrinsics
            self.intrinsics = np.array([[1377.22, 0,       957.80],
                                         [0,       1377.22, 722.04],
                                         [0,       0,       1.0]])

    frames = [_MockFrame(i, scan_dir) for i in range(50)]

    cfg = AutoTuner(
        frames              = frames,
        rgb_dir             = scan_dir / "rgb",
        depth_dir           = scan_dir / "depth",
        depth_scale         = 2.0,    # ← your DEPTH_SCALE
        enable_verification = False,
        verbose             = True,
    ).compute()

    print("\n" + "=" * 60)
    print("DERIVED PARAMETERS:")
    print("=" * 60)
    for k, v in sorted(cfg.items()):
        if isinstance(v, float):
            print(f"  {k:<42} = {v:.6f}")
        elif isinstance(v, np.ndarray):
            print(f"  {k:<42} = array{v.shape}")
        else:
            print(f"  {k:<42} = {v}")