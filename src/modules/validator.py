"""
modules/validator.py
Sanity-checks the reconstruction and prints a measurement report:
  - Point cloud bounding box (real-world metres)
  - Scene scale consistency
  - Pose trajectory length
  - Depth statistics per frame (sample)
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger(__name__)


class Validator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    # ── public ────────────────────────────────────────────────────────────────

    def report(self, frames, pcd_path: Path, mesh_path: Path):
        log.info("\n" + "=" * 60)
        log.info("  RECONSTRUCTION VALIDATION REPORT")
        log.info("=" * 60)

        self._check_poses(frames)
        self._check_pcd(pcd_path)
        self._check_mesh(mesh_path)
        self._check_depth_sample(frames)

        log.info("=" * 60 + "\n")

    # ── checks ────────────────────────────────────────────────────────────────

    def _check_poses(self, frames):
        log.info(f"\n[Poses]  {len(frames)} frames")
        positions = np.stack([fr.c2w[:3, 3] for fr in frames])
        traj_len  = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        bbox_min  = positions.min(axis=0)
        bbox_max  = positions.max(axis=0)
        log.info(f"  Camera trajectory length : {traj_len:.3f} m")
        log.info(f"  Camera bbox min (m)      : {bbox_min}")
        log.info(f"  Camera bbox max (m)      : {bbox_max}")
        log.info(f"  Spatial extent (m)       : {bbox_max - bbox_min}")

        # Check for degenerate poses
        det_vals = [np.linalg.det(fr.c2w[:3, :3]) for fr in frames]
        bad      = [i for i, d in enumerate(det_vals) if abs(d - 1.0) > 0.01]
        if bad:
            log.warning(f"  ⚠ Non-unit-det rotation matrices in frames: {bad}")
        else:
            log.info("  ✓ All rotation matrices have det≈1")

    def _check_pcd(self, pcd_path: Path):
        log.info(f"\n[Point Cloud]  {pcd_path.name}")
        if not pcd_path.exists():
            log.warning("  Not found – skipping")
            return

        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            pts = np.asarray(pcd.points)
        except ImportError:
            pts = self._read_ply_xyz(pcd_path)

        if len(pts) == 0:
            log.warning("  Empty point cloud!")
            return

        log.info(f"  Points           : {len(pts):,}")
        log.info(f"  Bounding box min : {pts.min(axis=0)}")
        log.info(f"  Bounding box max : {pts.max(axis=0)}")
        extent = pts.max(axis=0) - pts.min(axis=0)
        log.info(f"  Extent (m)       : {extent}  →  volume ≈ {np.prod(extent):.4f} m³")

        # Outlier detection
        center = pts.mean(axis=0)
        dists  = np.linalg.norm(pts - center, axis=1)
        p95    = np.percentile(dists, 95)
        far    = (dists > p95 * 5).sum()
        if far > 0:
            log.warning(f"  ⚠ {far:,} likely outlier points (>5× p95 distance from centroid)")
        else:
            log.info("  ✓ No gross outliers detected")

    def _check_mesh(self, mesh_path: Path):
        log.info(f"\n[Mesh]  {mesh_path.name}")
        if not mesh_path.exists():
            log.warning("  Not found – skipping")
            return

        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            log.info(f"  Vertices   : {len(mesh.vertices):,}")
            log.info(f"  Triangles  : {len(mesh.triangles):,}")
            log.info(f"  Watertight : {mesh.is_watertight()}")
            log.info(f"  Self-intersecting : {mesh.is_self_intersecting()}")
            vol = mesh.get_volume() if mesh.is_watertight() else None
            if vol is not None:
                log.info(f"  Volume     : {vol:.6f} m³  ({vol*1e6:.2f} cm³)")
        except ImportError:
            log.info("  (Open3D not available – skipping mesh validation)")

    def _check_depth_sample(self, frames, n_sample: int = 5):
        log.info(f"\n[Depth sample – first {n_sample} frames]")
        from src.modules.depth_fusion import _load_exr

        for fr in frames[:n_sample]:
            try:
                d = _load_exr(fr.depth_path).astype(np.float32) * fr.depth_scale
                valid = d[(d > 0.05) & (d < 50.0)]
                if len(valid) == 0:
                    log.warning(f"  Frame {fr.idx}: no valid depth values!")
                else:
                    log.info(
                        f"  Frame {fr.idx}: "
                        f"min={valid.min():.3f}m  "
                        f"max={valid.max():.3f}m  "
                        f"mean={valid.mean():.3f}m  "
                        f"valid={100*len(valid)/d.size:.1f}%"
                    )
            except Exception as e:
                log.warning(f"  Frame {fr.idx}: depth load error – {e}")

    # ── PLY reader (no Open3D) ────────────────────────────────────────────────

    @staticmethod
    def _read_ply_xyz(path: Path) -> np.ndarray:
        pts = []
        in_data = False
        with open(path) as f:
            for line in f:
                if line.strip() == "end_header":
                    in_data = True
                    continue
                if in_data:
                    vals = line.strip().split()
                    if len(vals) >= 3:
                        try:
                            pts.append([float(v) for v in vals[:3]])
                        except ValueError:
                            pass
        return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))