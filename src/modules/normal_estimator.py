"""
modules/normal_estimator.py

Estimates per-point surface normals from RGB images using a neural network
(DSINE or StableNormal), then injects them into the point cloud before
Poisson reconstruction.

Why this matters:
  Open3D's built-in normal estimation uses local PCA on the point cloud.
  On a noisy LiDAR scan, those normals are unreliable at surfaces with low
  point density (far walls, ceilings). Poisson meshing depends entirely on
  normal quality — bad normals = wavy, hallucinated geometry.
  Neural normals are computed per-pixel from the sharp RGB image,
  so they respect texture edges and fine surface detail.

Requirements:
    pip install torch torchvision
    pip install huggingface_hub        # for auto-download of DSINE weights

Usage in pipeline.py  (add after DepthFusion.fuse(), before Meshing):
    from src.modules.normal_estimator import NormalEstimator
    pcd_path = NormalEstimator(output_dir=out).estimate(pcd_path, frames)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

_FLIP = np.diag([1., -1., -1., 1.])


def _load_exr_depth(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    try:
        import OpenEXR, Imath
        f  = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        W  = dw.max.x - dw.min.x + 1
        H  = dw.max.y - dw.min.y + 1
        ch = next(iter(f.header()["channels"].keys()))
        raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    except (ImportError, OSError):
        pass
    from src.modules.exr_reader import read_exr_depth
    return read_exr_depth(path).astype(np.float32)


class _DSINEBackend:
    """
    DSINE: Rethinking Monocular Surface Normal Estimation.
    Auto-downloads weights (~200 MB) on first use.
    Paper: https://baegwangbin.github.io/DSINE/
    """
    REPO = "baegwangbin/DSINE"

    def __init__(self, device: str):
        self.device = device
        self._model = None

    def load(self):
        if self._model is not None:
            return
        log.info("  Loading DSINE normal estimator (~200 MB download on first run)...")
        import torch
        from huggingface_hub import hf_hub_download
        try:
            # Try loading from HuggingFace hub
            weights_path = hf_hub_download(repo_id=self.REPO,
                                           filename="dsine.pt")
            # DSINE expects its own repo structure — use torch.hub as primary
            raise ImportError("use torch.hub path")
        except Exception:
            pass
        try:
            self._model = torch.hub.load(
                "baegwangbin/DSINE", "DSINE", pretrained=True
            ).to(self.device).eval()
            log.info("  DSINE loaded via torch.hub.")
        except Exception as e:
            log.warning(f"  DSINE load failed: {e}. Will use geometric normals.")
            self._model = None

    def predict(self, rgb_path: Path,
                intrinsics_K: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns surface normal map (H×W×3, float32, world-space normal vectors).
        Returns None if model is not available.
        """
        if self._model is None:
            return None
        try:
            import torch
            from PIL import Image
            img = Image.open(rgb_path).convert("RGB")
            W_img, H_img = img.size

            # DSINE expects a normalised tensor and intrinsics
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img).unsqueeze(0).to(self.device)  # (1,3,H,W)

            # Build intrinsics tensor expected by DSINE: (1, 3, 3)
            K_t = torch.from_numpy(intrinsics_K.astype(np.float32)) \
                       .unsqueeze(0).to(self.device)

            with torch.no_grad():
                normals = self._model(img_t, K_t)  # (1,3,H,W), range [-1,1]

            normals_np = normals[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
            return normals_np.astype(np.float32)
        except Exception as e:
            log.warning(f"  DSINE predict failed: {e}")
            return None


class NormalEstimator:
    """
    Replaces Open3D's PCA normals with neural per-pixel normals.

    How it works:
      1. For each point in the cloud, find which frame observed it (nearest camera).
      2. Project the 3D point back to that frame's image.
      3. Look up the neural normal at that pixel.
      4. Transform the normal from camera space to world space.
      5. Assign to the point cloud.

    Parameters
    ----------
    output_dir : Path
        Where to save the normal-enhanced point cloud.
    model : "dsine" | "geometric"
        "dsine"      → neural normals (best quality, needs GPU).
        "geometric"  → Open3D PCA normals (faster, lower quality, no GPU needed).
    frame_skip : int
        Process every Nth frame for normal estimation (0 = all frames).
        Higher values = faster but sparser normal coverage.
    """

    def __init__(
        self,
        output_dir: Path,
        model: str = "dsine",
        frame_skip: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model
        self.frame_skip = frame_skip
        self._backend   = None

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def estimate(self, pcd_path: Path, frames) -> Path:
        """
        Add neural normals to the point cloud at pcd_path.
        Returns path to the enhanced point cloud (overwrites in-place).
        """
        try:
            import open3d as o3d
        except ImportError:
            log.warning("open3d not available — skipping normal estimation.")
            return pcd_path

        log.info(f"NormalEstimator ({self.model_name}): {pcd_path.name}")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points)

        if len(pts) == 0:
            return pcd_path

        if self.model_name == "geometric":
            return self._geometric_normals(pcd, pcd_path, frames)

        # ── neural path ───────────────────────────────────────────────────────
        device = self._get_device()
        if self._backend is None:
            self._backend = _DSINEBackend(device)
            self._backend.load()
            if self._backend._model is None:
                log.warning("  DSINE unavailable — falling back to geometric normals.")
                return self._geometric_normals(pcd, pcd_path, frames)

        # Select frames to use
        sel_frames = [fr for i, fr in enumerate(frames)
                      if i % (self.frame_skip + 1) == 0]
        log.info(f"  Using {len(sel_frames)} frames for normal estimation...")

        # Build per-point normal accumulator
        normal_sum   = np.zeros((len(pts), 3), dtype=np.float64)
        normal_count = np.zeros(len(pts), dtype=np.int32)

        try:
            import torch
            from scipy.spatial import cKDTree
        except ImportError:
            log.warning("  scipy not available — falling back to geometric normals.")
            return self._geometric_normals(pcd, pcd_path, frames)

        pts_t = torch.from_numpy(pts.astype(np.float32))

        for fr in sel_frames:
            try:
                intr = fr.intrinsics
                K    = intr.K().astype(np.float32)
                c2w  = (fr.c2w @ _FLIP).astype(np.float32)
                w2c  = np.linalg.inv(c2w)

                # Predict normals for this frame
                normals_map = self._backend.predict(fr.rgb_path, K)
                if normals_map is None:
                    continue

                H_n, W_n = normals_map.shape[:2]

                # Project all points into this camera
                pts_h  = np.hstack([pts, np.ones((len(pts), 1),
                                    dtype=np.float32)])  # (N,4)
                cam    = (w2c @ pts_h.T).T               # (N,4)
                Zc     = cam[:, 2]
                u_px   = (K[0,0] * cam[:,0] / (Zc + 1e-9) + K[0,2])
                v_px   = (K[1,1] * cam[:,1] / (Zc + 1e-9) + K[1,2])

                # Keep only points visible in this frame
                visible = (
                    (Zc > 0.1) &
                    (u_px >= 0) & (u_px < W_n) &
                    (v_px >= 0) & (v_px < H_n)
                )
                if not np.any(visible):
                    continue

                vis_idx = np.where(visible)[0]
                ui = u_px[vis_idx].astype(np.int32).clip(0, W_n - 1)
                vi = v_px[vis_idx].astype(np.int32).clip(0, H_n - 1)

                # Look up neural normal in camera space
                n_cam = normals_map[vi, ui].astype(np.float64)  # (M,3)

                # Transform to world space: normal_world = R_c2w @ n_cam
                # c2w[:3,:3] is the rotation from camera to world
                R_c2w = c2w[:3, :3].astype(np.float64)
                n_world = (R_c2w @ n_cam.T).T  # (M,3)

                # Normalise
                norms = np.linalg.norm(n_world, axis=1, keepdims=True) + 1e-9
                n_world /= norms

                # Accumulate (we'll average later)
                normal_sum[vis_idx]   += n_world
                normal_count[vis_idx] += 1

            except Exception as e:
                log.debug(f"  Frame {fr.idx} normal estimation error: {e}")
                continue

        # ── average accumulated normals ───────────────────────────────────────
        has_normals = normal_count > 0
        averaged    = np.zeros_like(normal_sum)
        averaged[has_normals] = (normal_sum[has_normals] /
                                 normal_count[has_normals, np.newaxis])

        # Points with no coverage: fall back to Open3D PCA normal
        n_missing = int((~has_normals).sum())
        if n_missing > 0:
            log.info(f"  {n_missing:,} points with no neural normal "
                     f"— using geometric fallback for those.")
            # Estimate geometric normals just for the missing points
            sub_pcd = pcd.select_by_index(np.where(~has_normals)[0])
            sub_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30))
            geo_norms = np.asarray(sub_pcd.normals)
            averaged[~has_normals] = geo_norms[:len(averaged[~has_normals])]

        # Normalise final normals
        norms = np.linalg.norm(averaged, axis=1, keepdims=True) + 1e-9
        averaged /= norms

        pcd.normals = o3d.utility.Vector3dVector(averaged.astype(np.float64))
        log.info(f"  Neural normals assigned to {int(has_normals.sum()):,} points.")

        o3d.io.write_point_cloud(str(pcd_path), pcd)
        log.info(f"  Saved normal-enhanced point cloud -> {pcd_path.name}")
        return pcd_path

    def _geometric_normals(self, pcd, pcd_path: Path, frames) -> Path:
        """Fallback: Open3D PCA normals oriented toward cameras."""
        import open3d as o3d
        log.info("  Estimating geometric normals (Open3D PCA)...")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Orient toward cameras
        if frames:
            try:
                from scipy.spatial import cKDTree
                cams  = np.array([(fr.c2w @ _FLIP)[:3, 3]
                                  for fr in frames[::3]])  # every 3rd frame
                pts   = np.asarray(pcd.points)
                norms = np.asarray(pcd.normals).copy()
                _, idx = cKDTree(cams).query(pts, k=1, workers=-1)
                to_cam = cams[idx] - pts
                flip   = np.einsum("ij,ij->i", norms, to_cam) < 0
                norms[flip] *= -1
                pcd.normals = o3d.utility.Vector3dVector(norms)
            except ImportError:
                pcd.orient_normals_consistent_tangent_plane(k=15)

        o3d.io.write_point_cloud(str(pcd_path), pcd)
        log.info(f"  Saved geometric-normal point cloud -> {pcd_path.name}")
        return pcd_path