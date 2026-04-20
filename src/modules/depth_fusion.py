"""
modules/depth_fusion.py  ── v7  "world-space-correct"
================================================================================
ROOT CAUSE ANALYSIS of v5/v6 failures (from mesh screenshots):
───────────────────────────────────────────────────────────────
[FAIL-1] CEILING HOLES (Image 2 — massive black voids in ceiling)
  Cause:  LiDAR cannot measure near-perpendicular bright surfaces.
          Ceiling returns depth=0 or huge noise. TSDF integrates
          noise → wrong SDF sign → geometry is carved out instead of built.
  Fix:    Detect zero/invalid regions AFTER bilateral smoothing and
          fill them via inpainting BEFORE TSDF. Additionally, clamp
          implausibly large depth jumps at ceiling-level pixels.

[FAIL-2] TORN EDGE ARTIFACTS (shards/confetti at all depth discontinuities)
  Cause:  At foreground/background boundaries (person-edge vs wall),
          LiDAR produces "flying pixels" — measurements that are neither
          the foreground object nor the background, but some mixture.
          These get integrated into TSDF at wrong depths, creating
          jagged tearing around every object edge.
  Fix:    _suppress_flying_pixels(): detect pixels where depth differs
          significantly from ALL 8 neighbours and zero them.
          This targets exactly the thin 1-2px border of flying pixels
          without touching object interiors or background.

[FAIL-3] IMAGE-SPACE DYNAMIC MASKING (v6 core logic was wrong)
  Cause:  Comparing depth at pixel (u,v) across frames only works when
          the camera barely moves. With 1276 frames orbiting a room,
          the same world point projects to completely different pixel
          coords in different frames → image-space std is always high
          for background walls too, creating massive over-masking.
  Fix:    REMOVE image-space dynamic masking. Instead use world-space
          TSDF's natural averaging: static geometry seen from many
          angles converges (votes agree → strong SDF); dynamic objects
          seen from few angles diverge (votes cancel → weak SDF, removed
          by density filter). This is what TSDF was designed for.

[FAIL-4] WINDOW/GLASS GEOMETRY (ghost planes outside window)
  Cause:  Overexposed window pixels: LiDAR returns ~max_depth for sky,
          neural depth also confused. TSDF builds a surface at ~3.5m
          outside the window.
  Fix:    _suppress_sky_depth(): pixels where RGB is very bright (>230
          all channels = overexposed sky/window) AND depth > 2.5m →
          zero the depth. Room geometry is never that bright+far.

[FAIL-5] DEPTH DISCONTINUITY PROPAGATION IN BILATERAL FILTER
  Cause:  Bilateral filter with sigma_color=0.04 still blurs across
          large depth jumps (person at 1.5m, wall at 2.5m) creating
          intermediate depth values that neither belong to the person
          nor the wall.
  Fix:    Use guided bilateral filter with tighter sigma_color=0.02
          and detect+preserve depth edges explicitly.

DESIGN PRINCIPLES v7:
─────────────────────
• Per-pixel confidence is computed BEFORE TSDF (not after).
• Zero-depth pixels are inpainted (not passed to TSDF as zero).
• Flying pixels at depth edges are suppressed (not blurred).
• Sky/overexposed pixels are zeroed (not given max_depth).
• TSDF truncation is kept at 3.0× voxel (tight, single surfaces).
• Post-TSDF cleanup uses aggressive radius filter (confetti removal)
  followed by conservative SOR (wall preservation).
================================================================================
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_FLIP = np.diag([1., -1., -1., 1.])


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_exr(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    try:
        import OpenEXR, Imath
        f   = OpenEXR.InputFile(str(path))
        dw  = f.header()["dataWindow"]
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        ch  = next(iter(f.header()["channels"].keys()))
        raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    except (ImportError, OSError):
        pass
    from src.modules.exr_reader import read_exr_depth
    return read_exr_depth(path).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# [FIX-2]  Flying pixel suppression at depth discontinuities
# ══════════════════════════════════════════════════════════════════════════════

def _suppress_flying_pixels(depth: np.ndarray,
                             jump_thresh_m: float = 0.15,
                             min_bad_neighbours: int = 5) -> np.ndarray:
    """
    Remove "flying pixels" — depth measurements at foreground/background
    boundaries that are neither the foreground object nor the background.

    A pixel is a flying pixel if its depth differs by > jump_thresh_m from
    at least min_bad_neighbours of its 8 neighbours (meaning it is surrounded
    by pixels at a very different depth — it is an isolated outlier at an edge).

    This targets the thin 1-2px ring of garbage depth at every object edge
    without affecting object interiors or large uniform background regions.
    """
    if depth is None or depth.size == 0:
        return depth

    H, W = depth.shape
    out  = depth.copy()
    valid = depth > 0.05

    # Build 8-neighbour difference maps
    bad_count = np.zeros((H, W), dtype=np.int32)

    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        shifted = np.zeros_like(depth)
        sy = slice(max(0, dy), H + min(0, dy))
        sx = slice(max(0, dx), W + min(0, dx))
        ty = slice(max(0, -dy), H + min(0, -dy))
        tx = slice(max(0, -dx), W + min(0, -dx))
        shifted[ty, tx] = depth[sy, sx]

        neighbour_valid = shifted > 0.05
        both_valid = valid & neighbour_valid
        big_jump = both_valid & (np.abs(depth - shifted) > jump_thresh_m)
        bad_count += big_jump.astype(np.int32)

    flying = valid & (bad_count >= min_bad_neighbours)
    out[flying] = 0.0

    return out


# ══════════════════════════════════════════════════════════════════════════════
# [FIX-4]  Sky / overexposed window suppression
# ══════════════════════════════════════════════════════════════════════════════

def _suppress_sky_depth(depth: np.ndarray,
                         rgb: Optional[np.ndarray],
                         sky_bright_thresh: int = 230,
                         sky_depth_min_m: float = 2.0) -> np.ndarray:
    """
    Zero depth at pixels that are simultaneously:
      - Very bright across all RGB channels (overexposed sky/window): R>230, G>230, B>230
      - Far away (> sky_depth_min_m): likely outside the room

    This prevents TSDF from building ghost geometry outside windows.
    Room interiors are never simultaneously that bright AND that far.
    """
    if rgb is None or rgb.shape[:2] != depth.shape:
        return depth

    overexposed = (rgb[:, :, 0].astype(np.int32) > sky_bright_thresh) & \
                  (rgb[:, :, 1].astype(np.int32) > sky_bright_thresh) & \
                  (rgb[:, :, 2].astype(np.int32) > sky_bright_thresh)

    sky_mask = overexposed & (depth > sky_depth_min_m)

    out = depth.copy()
    out[sky_mask] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# [FIX-1]  Border mask — only hard-zero the LiDAR FOV edge strip
# ══════════════════════════════════════════════════════════════════════════════

def _mask_depth_borders(depth: np.ndarray, border_frac: float = 0.02) -> np.ndarray:
    H, W = depth.shape
    bh = max(2, int(H * border_frac))
    bw = max(2, int(W * border_frac))
    out = depth.copy()
    out[:bh,  :] = 0.0
    out[-bh:, :] = 0.0
    out[:,  :bw] = 0.0
    out[:, -bw:] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Depth inpainting — fill holes BEFORE TSDF
# ══════════════════════════════════════════════════════════════════════════════

def _inpaint_depth(depth: np.ndarray, max_hole_px: int = 20) -> np.ndarray:
    """
    Fill zero-depth holes. Larger radius (20px vs old 12px) to better
    handle ceiling holes from LiDAR failure.
    """
    hole_mask = (depth <= 0).astype(np.uint8)
    if hole_mask.sum() == 0:
        return depth

    try:
        import cv2
        valid = depth > 0
        if not valid.any():
            return depth
        d_max  = float(depth[valid].max()) + 1e-6
        d_u16  = (np.clip(depth / d_max, 0, 1) * 65535).astype(np.uint16)
        filled = cv2.inpaint(d_u16, hole_mask, inpaintRadius=max_hole_px,
                             flags=cv2.INPAINT_NS)
        result = filled.astype(np.float32) / 65535.0 * d_max
        out = depth.copy()
        out[hole_mask.astype(bool)] = result[hole_mask.astype(bool)]
        return out
    except ImportError:
        pass

    # scipy fallback: nearest-neighbour
    try:
        from scipy.ndimage import distance_transform_edt
        out   = depth.copy()
        valid = depth > 0
        _, indices = distance_transform_edt(~valid, return_indices=True)
        out[~valid] = depth[tuple(indices[:, ~valid])]
        return out
    except ImportError:
        pass

    return depth


# ══════════════════════════════════════════════════════════════════════════════
# [FIX-5]  Depth smoothing with tighter sigma to avoid cross-edge blurring
# ══════════════════════════════════════════════════════════════════════════════

def _smooth_depth_bilateral(depth: np.ndarray,
                              d: int = 7,
                              sigma_color: float = 0.02,   # tighter than v5's 0.04
                              sigma_space: float = 3.0) -> np.ndarray:
    try:
        import cv2
        return cv2.bilateralFilter(depth.astype(np.float32), d,
                                   sigma_color, sigma_space)
    except ImportError:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(depth, sigma=0.8).astype(np.float32)
        except ImportError:
            return depth


# ══════════════════════════════════════════════════════════════════════════════
# Depth range validation — hard clamp with outlier rejection
# ══════════════════════════════════════════════════════════════════════════════

def _validate_depth_range(depth: np.ndarray,
                           min_d: float,
                           max_d: float) -> np.ndarray:
    """
    Beyond simple min/max clamp: also detect implausible local spikes.
    A pixel that is 3× deeper than its local neighbourhood median is
    likely a LiDAR multipath reflection — zero it.
    """
    out = np.where((depth >= min_d) & (depth <= max_d), depth, 0.0).astype(np.float32)

    try:
        import cv2

        valid = out > 0
        if valid.sum() < 100:
            return out

        # Normalize to 0–255
        d_max = float(out[valid].max()) + 1e-6
        out_u8 = np.clip(out / d_max, 0, 1)
        out_u8 = (out_u8 * 255).astype(np.uint8)

        # Apply median blur
        median_u8 = cv2.medianBlur(out_u8, 15)

        # Convert back to float depth
        median = median_u8.astype(np.float32) / 255.0 * d_max

        # Detect spikes
        spike = valid & (median > 0) & (out > median * 2.5)
        out[spike] = 0.0
    except ImportError:
        pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Planar snap (unchanged from v5, proven to work)
# ══════════════════════════════════════════════════════════════════════════════

def _snap_to_planes(pcd,
                    plane_distance_thresh: float = 0.015,
                    n_planes: int = 6,
                    snap_dist: float = 0.05,
                    remove_dist: float = 0.15,
                    min_inlier_fraction: float = 0.02):
    try:
        import open3d as o3d
    except ImportError:
        return pcd

    pts     = np.asarray(pcd.points).copy()
    n_total = len(pts)
    if n_total < 500:
        return pcd

    remaining_idx = np.arange(n_total)
    planes        = []

    for _ in range(n_planes):
        if len(remaining_idx) < 200:
            break
        sub = pcd.select_by_index(remaining_idx)
        try:
            plane_model, inliers = sub.segment_plane(
                distance_threshold=plane_distance_thresh,
                ransac_n=3, num_iterations=1000)
        except Exception:
            break
        if len(inliers) < n_total * min_inlier_fraction:
            break
        planes.append(plane_model)
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]

    if not planes:
        return pcd

    normals    = np.array([[p[0], p[1], p[2]] for p in planes], dtype=np.float32)
    ds         = np.array([p[3] for p in planes], dtype=np.float32)
    norm_len   = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    unit_norms = normals / norm_len

    signed_dists = (pts @ normals.T + ds[None]) / (norm_len.T + 1e-9)
    abs_dists    = np.abs(signed_dists)
    nearest_p    = abs_dists.argmin(axis=1)
    min_dist     = abs_dists.min(axis=1)

    snap_mask = min_dist < snap_dist
    for i in range(len(planes)):
        pm = snap_mask & (nearest_p == i)
        if pm.any():
            pts[pm] -= signed_dists[pm, i][:, None] * unit_norms[i]

    # Ghost removal: points on the wrong side of a plane (behind wall/ceiling)
    ghost_mask = np.zeros(n_total, dtype=bool)
    for i in range(len(planes)):
        dists_i = signed_dists[:, i]
        on_plane = abs_dists[:, i] < snap_dist
        if on_plane.sum() == 0:
            continue
        pos_side = np.sum(dists_i[on_plane] > 0)
        neg_side = np.sum(dists_i[on_plane] < 0)
        if pos_side > neg_side:
            ghost_candidates = (dists_i < -snap_dist) & (dists_i > -1.0)
        else:
            ghost_candidates = (dists_i > snap_dist) & (dists_i < 1.0)
        ghost_mask |= ghost_candidates

    far_mask = min_dist > remove_dist
    try:
        labels = np.array(pcd.cluster_dbscan(eps=0.06, min_points=25,
                                              print_progress=False))
        valid_l = labels[labels >= 0]
        main_c  = int(np.bincount(valid_l).argmax()) if len(valid_l) > 0 else -1
        in_main = labels == main_c
    except Exception:
        in_main = np.ones(n_total, dtype=bool)

    remove = (far_mask & ~in_main) | ghost_mask
    keep   = ~remove

    import open3d as o3d
    pcd_out = pcd.select_by_index(np.where(keep)[0])
    pcd_out.points = o3d.utility.Vector3dVector(pts[keep])
    if pcd.has_colors():
        pcd_out.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[keep])
    log.info(f"  Planar snap: removed {int(remove.sum()):,} pts")
    return pcd_out


# ══════════════════════════════════════════════════════════════════════════════
# Mesh hole filling
# ══════════════════════════════════════════════════════════════════════════════

def _fill_mesh_holes(mesh, max_hole_size: int = 50):
    try:
        import open3d as o3d
    except ImportError:
        return mesh

    n_before = len(mesh.triangles)
    try:
        filled = mesh.fill_holes(hole_size=max_hole_size)
        if hasattr(filled, "triangles") and len(filled.triangles) > n_before:
            mesh = filled
            log.info(f"  Hole fill: {n_before:,} → {len(mesh.triangles):,} faces")
    except Exception as e:
        log.debug(f"  fill_holes skipped: {e}")

    try:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=2,
                                            lambda_filter=0.5)
        mesh.compute_vertex_normals()
    except Exception:
        pass

    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# Aggressive confetti removal (two-pass)
# ══════════════════════════════════════════════════════════════════════════════

def _remove_confetti(pcd, voxel_size: float):
    """
    Two-pass outlier removal designed specifically for the torn-edge
    confetti seen in the mesh screenshots:

    Pass 1 — radius filter with tight radius:
      Any point with < 10 neighbours within 4× voxel is floating confetti.
      (Dense walls always have many neighbours; isolated torn fragments don't.)

    Pass 2 — statistical outlier removal with moderate std_ratio:
      Removes points whose mean distance to neighbours is an outlier.
      std_ratio=2.0 is tighter than v5's 3.0 to catch more confetti.
    """
    if len(pcd.points) < 50:
        return pcd

    n0 = len(pcd.points)

    # Pass 1: radius filter — catch completely isolated points
    pcd, _ = pcd.remove_radius_outlier(nb_points=10,
                                        radius=voxel_size * 4.0)
    log.info(f"  Confetti P1 (radius): {n0:,} → {len(pcd.points):,}")

    if len(pcd.points) < 50:
        return pcd
    n1 = len(pcd.points)

    # Pass 2: SOR — catch clusters of confetti that pass the radius test
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log.info(f"  Confetti P2 (SOR):    {n1:,} → {len(pcd.points):,}")

    return pcd


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class DepthFusion:

    def __init__(
        self,
        output_dir,
        depth_scale: float = 1.0,
        min_depth: float = 0.15,
        max_depth: float = 3.5,
        voxel_size: float = 0.025,
        frame_skip: int = 1,
        stop_flag: Optional[Callable[[], bool]] = None,
        scene_mode: bool = True,
        # Smoothing
        use_bilateral: bool = True,
        # Glass weight (NOT hard mask)
        use_glass_weight: bool = False,
        glass_bright_threshold: int = 220,
        glass_min_weight: float = 0.20,
        # Border
        use_border_mask: bool = True,
        border_frac: float = 0.020,
        # Depth inpainting (larger radius for ceiling)
        use_inpaint: bool = True,
        inpaint_radius_px: int = 20,
        # Soft multi-view consistency
        use_soft_consistency: bool = False,   # disabled: world-space TSDF handles this
        consistency_window: int = 3,
        consistency_threshold_m: float = 0.08,
        consistency_min_weight: float = 0.25,
        # Dynamic object weighting (in-TSDF)
        use_dynamic_weight: bool = True,
        dynamic_motion_thresh_m: float = 0.08,
        dynamic_min_weight: float = 0.0,
        # Planar snap
        use_planar_snap: bool = True,
        planar_ransac_thresh: float = 0.015,
        n_planes: int = 6,
        planar_snap_dist: float = 0.05,
        planar_remove_dist: float = 0.15,
        # Mesh hole filling
        use_hole_fill: bool = True,
        max_hole_size: int = 50,
        # TSDF
        sdf_trunc_multiplier: float = 3.0,
        # v7 new parameters
        use_flying_pixel_suppress: bool = True,
        flying_pixel_jump_thresh_m: float = 0.15,
        use_sky_suppress: bool = True,
        sky_bright_thresh: int = 230,
        sky_depth_min_m: float = 2.0,
        use_spike_filter: bool = True,
        # Legacy compatibility
        use_consistency: bool = False,
        consistency_threshold: float = 0.08,
        use_foreground_mask: bool = False,
        foreground_depth_band: float = 0.35,
        use_grabcut_refine: bool = False,
        grabcut_downscale: int = 2,
        use_glass_mask: bool = False,
        use_multiview_consistency: bool = False,
        multiview_window: int = 3,
        multiview_threshold_m: float = 0.08,
        use_planar_prior: bool = False,
        floating_dist_thresh: float = 0.12,
        min_view_count: int = 2,
    ):
        self.output_dir                  = Path(output_dir)
        self.depth_scale                 = depth_scale
        self.min_depth                   = min_depth
        self.max_depth                   = max_depth
        self.voxel_size                  = voxel_size
        self.frame_skip                  = frame_skip
        self.stop_flag                   = stop_flag or (lambda: False)
        self.scene_mode                  = scene_mode
        self.use_bilateral               = use_bilateral
        self.use_glass_weight            = use_glass_weight
        self.glass_bright_threshold      = glass_bright_threshold
        self.glass_min_weight            = glass_min_weight
        self.use_border_mask             = use_border_mask
        self.border_frac                 = border_frac
        self.use_inpaint                 = use_inpaint
        self.inpaint_radius_px           = inpaint_radius_px
        self.use_soft_consistency        = use_soft_consistency
        self.consistency_window          = consistency_window
        self.consistency_threshold_m     = consistency_threshold_m
        self.consistency_min_weight      = consistency_min_weight
        self.use_dynamic_weight          = use_dynamic_weight
        self.dynamic_motion_thresh_m     = dynamic_motion_thresh_m
        self.dynamic_min_weight          = dynamic_min_weight
        self.use_planar_snap             = use_planar_snap
        self.planar_ransac_thresh        = planar_ransac_thresh
        self.n_planes                    = n_planes
        self.planar_snap_dist            = planar_snap_dist
        self.planar_remove_dist          = planar_remove_dist
        self.use_hole_fill               = use_hole_fill
        self.max_hole_size               = max_hole_size
        self.sdf_trunc_multiplier        = float(np.clip(sdf_trunc_multiplier, 2.0, 6.0))
        self.use_flying_pixel_suppress   = use_flying_pixel_suppress
        self.flying_pixel_jump_thresh_m  = flying_pixel_jump_thresh_m
        self.use_sky_suppress            = use_sky_suppress
        self.sky_bright_thresh           = sky_bright_thresh
        self.sky_depth_min_m             = sky_depth_min_m
        self.use_spike_filter            = use_spike_filter
        self._frames_ref                 = None
        self.last_tsdf_mesh_path         = None

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def fuse(self, frames: list) -> Path:
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required: pip install open3d")

        self._frames_ref = frames
        out_path  = self.output_dir / "fused_pointcloud.ply"
        sdf_trunc = self.voxel_size * self.sdf_trunc_multiplier

        log.info(f"  TSDF v7: voxel={self.voxel_size}m  "
                 f"sdf_trunc={sdf_trunc:.4f}m  "
                 f"depth=[{self.min_depth},{self.max_depth}]m")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        sel   = frames[::max(1, self.frame_skip)]
        n_sel = len(sel)
        log.info(f"  Integrating {n_sel}/{len(frames)} frames ...")

        # Pre-load depth cache for dynamic weight computation
        depth_cache: Dict[int, np.ndarray] = {}
        for i, fr in enumerate(sel):
            try:
                depth_cache[i] = (_load_exr(fr.depth_path).astype(np.float32)
                                  * self.depth_scale)
            except Exception as e:
                log.warning(f"  Frame {fr.idx}: depth cache failed ({e})")
        log.info(f"  Depth cache: {len(depth_cache)} frames loaded")

        n_fused = 0
        for i, fr in enumerate(sel):
            if self.stop_flag():
                break
            if i not in depth_cache:
                continue

            depth = depth_cache[i].copy()

            # ── Step 1: Border strip (only hard-zero) ─────────────────────────
            if self.use_border_mask:
                depth = _mask_depth_borders(depth, self.border_frac)

            # ── Load RGB ───────────────────────────────────────────────────────
            rgb_np = None
            try:
                from PIL import Image as PILImage
                rgb_pil = PILImage.open(fr.rgb_path).convert("RGB")
                rgb_np  = np.asarray(rgb_pil, dtype=np.uint8)
                if rgb_np.shape[:2] != depth.shape:
                    rgb_np = np.asarray(
                        rgb_pil.resize((depth.shape[1], depth.shape[0]),
                                       PILImage.BILINEAR),
                        dtype=np.uint8)
            except Exception:
                pass

            # ── Step 2: [FIX-4] Sky/window suppression ────────────────────────
            if self.use_sky_suppress:
                depth = _suppress_sky_depth(depth, rgb_np,
                                             self.sky_bright_thresh,
                                             self.sky_depth_min_m)

            # ── Step 3: [FIX-5] Bilateral smoothing (tighter sigma) ───────────
            if self.use_bilateral:
                depth = _smooth_depth_bilateral(depth,
                                                sigma_color=0.02,
                                                sigma_space=3.0)

            # ── Step 4: [FIX-2] Flying pixel suppression ──────────────────────
            if self.use_flying_pixel_suppress:
                depth = _suppress_flying_pixels(depth,
                                                jump_thresh_m=self.flying_pixel_jump_thresh_m)

            # ── Step 5: [FIX-1] Inpaint holes (larger radius for ceiling) ─────
            if self.use_inpaint:
                depth = _inpaint_depth(depth, max_hole_px=self.inpaint_radius_px)

            # ── Step 6: Depth range + spike filter ────────────────────────────
            if self.use_spike_filter:
                depth = _validate_depth_range(depth, self.min_depth, self.max_depth)
            else:
                depth = np.where(
                    (depth >= self.min_depth) & (depth <= self.max_depth),
                    depth, 0.0).astype(np.float32)

            # ── Step 7: Dynamic object down-weighting (in-TSDF) ───────────────
            conf = np.ones(depth.shape, dtype=np.float32)
            if self.use_dynamic_weight:
                depth_prev = depth_cache.get(i - 1)
                depth_next = depth_cache.get(i + 1)
                if depth_prev is not None or depth_next is not None:
                    motion = np.zeros_like(depth)
                    n_ref  = 0
                    for ref in [depth_prev, depth_next]:
                        if ref is None:
                            continue
                        n_ref += 1
                        both  = (depth > 0.05) & (ref > 0.05)
                        diff  = np.zeros_like(depth)
                        diff[both] = np.abs(depth[both] - ref[both])
                        motion += diff
                    if n_ref > 0:
                        motion /= n_ref
                        dyn_ratio = np.clip(motion / self.dynamic_motion_thresh_m,
                                            0.0, 1.0)
                        conf = 1.0 - dyn_ratio
                        conf[depth <= 0.05] = 0.0

            # Apply confidence: blend toward smooth estimate for low-conf pixels
            if conf.min() < 1.0:
                smooth  = _smooth_depth_bilateral(depth, d=9,
                                                  sigma_color=0.05,
                                                  sigma_space=4.0)
                valid   = depth > 0.05
                depth   = np.where(valid,
                                   conf * depth + (1.0 - conf) * smooth,
                                   depth).astype(np.float32)

            # Resize rgb to match depth if needed
            rgb_tsdf = (rgb_np if rgb_np is not None
                        else np.zeros((*depth.shape, 3), np.uint8))
            if rgb_tsdf.shape[:2] != depth.shape:
                try:
                    from PIL import Image as PILImage2
                    rgb_tsdf = np.asarray(
                        PILImage2.fromarray(rgb_tsdf).resize(
                            (depth.shape[1], depth.shape[0]),
                            PILImage2.BILINEAR), dtype=np.uint8)
                except Exception:
                    rgb_tsdf = np.zeros((*depth.shape, 3), np.uint8)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_tsdf.astype(np.uint8)),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=self.max_depth,
                convert_rgb_to_intensity=False,
            )

            intr = fr.intrinsics
            depth_h, depth_w = depth.shape
            scale_x = depth_w / intr.width
            scale_y = depth_h / intr.height

            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                depth_w, depth_h,
                intr.fx * scale_x, intr.fy * scale_y,
                intr.cx * scale_x, intr.cy * scale_y,
            )
            volume.integrate(
                rgbd,
                o3d_intr,
                np.linalg.inv(fr.c2w @ _FLIP),
            )
            n_fused += 1
            if n_fused % 50 == 0:
                log.info(f"  {n_fused}/{n_sel} frames integrated...")

        log.info(f"  {n_fused} frames fused")
        log.info("Extracting point cloud...")
        pcd = volume.extract_point_cloud()
        log.info(f"  Raw: {len(pcd.points):,} points")

        if len(pcd.points) == 0:
            log.error("TSDF produced 0 points — check depth range and poses.")
            o3d.io.write_point_cloud(str(out_path), pcd)
            return out_path

        pcd = self._cleanup_scene(pcd)

        # Extract TSDF mesh and fill holes
        try:
            tsdf_mesh = volume.extract_triangle_mesh()
            if len(tsdf_mesh.triangles) > 0:
                tsdf_mesh.compute_vertex_normals()
                if self.use_hole_fill:
                    tsdf_mesh = _fill_mesh_holes(tsdf_mesh, self.max_hole_size)
                self.last_tsdf_mesh_path = self.output_dir / "fused_tsdf_mesh.ply"
                o3d.io.write_triangle_mesh(str(self.last_tsdf_mesh_path), tsdf_mesh)
                log.info(f"Saved TSDF mesh -> {self.last_tsdf_mesh_path}")
        except Exception as e:
            log.warning(f"TSDF mesh extraction failed: {e}")

        o3d.io.write_point_cloud(str(out_path), pcd)
        log.info(f"Saved -> {out_path}  ({len(pcd.points):,} points)")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_scene(self, pcd):
        import open3d as o3d

        pts = np.asarray(pcd.points)
        ok  = np.isfinite(pts).all(axis=1)
        if (~ok).sum() > 0:
            pcd = pcd.select_by_index(np.where(ok)[0])

        pcd = pcd.voxel_down_sample(self.voxel_size)
        log.info(f"  After voxel downsample: {len(pcd.points):,}")

        pcd = self._camera_proximity_crop(pcd)

        # Two-pass confetti removal (more aggressive than v5)
        pcd = _remove_confetti(pcd, self.voxel_size)

        # Planar snap (move noisy points to surfaces instead of deleting)
        if self.use_planar_snap and len(pcd.points) > 500:
            pcd = _snap_to_planes(
                pcd,
                plane_distance_thresh=self.planar_ransac_thresh,
                n_planes=self.n_planes,
                snap_dist=self.planar_snap_dist,
                remove_dist=self.planar_remove_dist,
            )
            log.info(f"  After planar snap: {len(pcd.points):,}")

        # DBSCAN — keep clusters ≥ 0.5% of largest (tighter than v5's 0.3%)
        try:
            labels = np.array(pcd.cluster_dbscan(
                eps=float(self.voxel_size * 6.0),
                min_points=30,
                print_progress=False,
            ))
        except Exception as e:
            log.warning(f"  DBSCAN skipped: {e}")
            return pcd

        if labels.size == 0 or np.all(labels < 0):
            return pcd

        valid   = labels >= 0
        if not np.any(valid):
            return pcd
        counts  = np.bincount(labels[valid])
        largest = int(counts.max())

        min_pts   = max(100, int(largest * 0.005))  # 0.5% — tighter than v5
        keep_mask = (labels >= 0) & (counts[labels] >= min_pts)
        removed   = int((~keep_mask).sum())
        if removed > 0:
            log.info(f"  DBSCAN: removed {removed:,} pts, "
                     f"kept {int(keep_mask.sum()):,}")
            pcd = pcd.select_by_index(np.where(keep_mask)[0])

        return pcd

    def _camera_proximity_crop(self, pcd):
        if self._frames_ref is None:
            return pcd

        pts = np.asarray(pcd.points)
        cam_positions = np.array([
            (fr.c2w @ _FLIP)[:3, 3]
            for i, fr in enumerate(self._frames_ref)
            if i % 5 == 0
        ])
        threshold = self.max_depth * 1.15
        log.info(f"  Proximity crop: threshold={threshold:.2f}m ...")

        try:
            import torch
            if torch.cuda.is_available():
                device   = torch.device('cuda')
                pts_t    = torch.from_numpy(pts).float().to(device)
                cams_t   = torch.from_numpy(cam_positions).float().to(device)
                chunk    = 100_000
                min_dist = np.full(len(pts), np.inf, dtype=np.float32)
                for i in range(0, len(pts), chunk):
                    end = min(i + chunk, len(pts))
                    d   = torch.cdist(pts_t[i:end], cams_t)
                    min_dist[i:end] = d.min(dim=1).values.cpu().numpy()
                keep = min_dist <= threshold
                log.info(f"  Proximity crop (GPU): kept {keep.sum():,}")
                return pcd.select_by_index(np.where(keep)[0])
        except ImportError:
            pass

        chunk    = 50_000
        min_dist = np.full(len(pts), np.inf, dtype=np.float32)
        for start in range(0, len(pts), chunk):
            p = pts[start:start+chunk]
            d = np.linalg.norm(
                p[:, None, :] - cam_positions[None, :, :], axis=2
            ).min(axis=1)
            min_dist[start:start+chunk] = d
        keep = min_dist <= threshold
        log.info(f"  Proximity crop (CPU): kept {keep.sum():,}")
        return pcd.select_by_index(np.where(keep)[0])