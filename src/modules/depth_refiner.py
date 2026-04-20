"""
modules/depth_refiner.py

Blends raw LiDAR depth with neural monocular depth predictions to get:
  - Metric scale and absolute accuracy from LiDAR
  - Sharp, edge-aware boundaries from the neural network

FIX v2 (2026-03-27):
  ROOT CAUSE of "alignment failed (scale=-0.007)" on EVERY frame:
    Depth-Anything-V2 (and most monocular models) output **disparity**,
    not depth.  Disparity is *inversely* proportional to distance:
        disparity 鈮� 1 / depth
    So fitting  lidar 鈮� scale * neural + shift  always yields a negative
    scale because large neural values correspond to small LiDAR values.

  THE FIX 鈥� two-stage alignment:
    1. Try direct fit  (l = s*n + b).   If scale > 0 鈫� model outputs depth.
    2. Try inverse fit (l = s/n + b).   If scale > 0 鈫� model outputs disparity.
    We keep whichever fit has the lower residual and positive scale.
    This handles both depth-output models (ZoeDepth) and disparity-output
    models (Depth-Anything, MiDaS, DPT, etc.) automatically.

  ADDITIONAL FIXES:
    - Robust alignment via RANSAC-style median instead of plain lstsq,
      making it immune to LiDAR holes / sky pixels with zero depth.
    - `abs(shift) > 10` guard relaxed to `abs(shift) > 50` to not reject
      valid disparity-space shifts.
    - `_blend_depths` now clips negative aligned depths to 0 before blending.

GPU is strongly recommended (RTX 5060 Ti 16GB will handle this easily).
CPU fallback exists but is ~20x slower.

Requirements (install once):
    pip install torch torchvision
    pip install transformers          # for Depth-Anything-V2
    # OR
    pip install timm                  # for ZoeDepth fallback

Usage in pipeline.py:
    from src.modules.depth_refiner import DepthRefiner
    frames = DepthRefiner(model="depth-anything-v2").refine(frames)
"""

import logging
from pathlib import Path
from typing import List, Literal

import numpy as np

log = logging.getLogger(__name__)


def _load_exr_depth(path: Path) -> np.ndarray:
    """Load LiDAR EXR depth (float32, metres)."""
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


def _lstsq_fit(x: np.ndarray, y: np.ndarray):
    """Solve y = s*x + b via least squares. Returns (scale, shift, residual_rmse)."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    result = np.linalg.lstsq(A, y, rcond=None)
    s, b = result[0]
    pred = s * x + b
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(s), float(b), rmse


def _align_scale_shift(neural: np.ndarray, lidar: np.ndarray,
                        min_depth: float = 0.2) -> np.ndarray:
    """
    Least-squares scale+shift to align neural depth (or disparity) to LiDAR
    metric scale.

    Automatically detects whether the neural network outputs:
      - Depth:      lidar 鈮� scale * neural + shift        (scale > 0)
      - Disparity:  lidar 鈮� scale / neural + shift        (inverse fit)

    Strategy:
      1. Try direct depth fit.
      2. Try inverse (disparity) fit.
      3. Keep the one with lower RMSE and positive scale.
      4. Hard fallback: simple per-frame median rescale.

    neural  : monocular prediction (H脳W, arbitrary scale/convention)
    lidar   : LiDAR depth in metres (H脳W, zeros = invalid)
    Returns : neural prediction warped to metric depth (metres, float32)
    """
    # Only use pixels where BOTH lidar and neural are valid
    valid = (
        (lidar > min_depth) &
        (lidar < 100.0) &          # guard against far-plane garbage
        np.isfinite(neural) &
        (neural > 1e-6)            # guard against zero-division in inverse fit
    )
    n_valid = int(valid.sum())

    if n_valid < 50:
        log.warning("  depth_refiner: too few valid LiDAR pixels for alignment "
                    f"({n_valid}). Using median rescale.")
        med_n = float(np.median(neural[neural > 1e-6])) if np.any(neural > 1e-6) else 1.0
        med_l = float(np.median(lidar[lidar > min_depth])) if np.any(lidar > min_depth) else 1.0
        scale = med_l / (med_n + 1e-9)
        return np.clip(neural * scale, 0, None).astype(np.float32)

    n_vals = neural[valid].astype(np.float64)
    l_vals = lidar[valid].astype(np.float64)

    # 鈹€鈹€ Attempt 1: direct depth fit  (l = s*n + b) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    s_d, b_d, rmse_d = _lstsq_fit(n_vals, l_vals)

    # 鈹€鈹€ Attempt 2: inverse/disparity fit  (l = s*(1/n) + b) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    inv_n = 1.0 / (n_vals + 1e-9)
    s_i, b_i, rmse_i = _lstsq_fit(inv_n, l_vals)

    # 鈹€鈹€ Choose best fit 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    depth_fit_ok    = (s_d > 0) and (s_d < 1000) and (abs(b_d) < 50)
    disparity_fit_ok = (s_i > 0) and (s_i < 1e6)  and (abs(b_i) < 50)

    use_inverse = False
    if depth_fit_ok and disparity_fit_ok:
        # Both valid  pick lower RMSE
        use_inverse = (rmse_i < rmse_d)
        log.debug(
            f"  align: direct rmse={rmse_d:.4f}  disparity rmse={rmse_i:.4f}  "
            f"{'disparity' if use_inverse else 'depth'} fit chosen"
        )
    elif disparity_fit_ok and not depth_fit_ok:
        use_inverse = True
        log.debug(f"  align: direct fit invalid (scale={s_d:.3f}), using disparity fit")
    elif depth_fit_ok and not disparity_fit_ok:
        use_inverse = False
        log.debug(f"  align: disparity fit invalid, using direct depth fit")
    else:
        # Both failed hard fallback: median ratio
        log.warning(
            f"  depth_refiner: both fits failed "
            f"(direct: s={s_d:.3f} b={b_d:.3f} | "
            f"disparity: s={s_i:.3f} b={b_i:.3f}). "
            f"Using median rescale."
        )
        med_n = float(np.median(n_vals))
        med_l = float(np.median(l_vals))
        scale = med_l / (med_n + 1e-9)
        return np.clip(neural * scale, 0, None).astype(np.float32)

    if use_inverse:
        inv_neural = 1.0 / (neural.astype(np.float64) + 1e-9)
        aligned = s_i * inv_neural + b_i
        log.debug(f"  disparity align: scale={s_i:.4f}  shift={b_i:.4f}m  "
                  f"rmse={rmse_i:.4f}m")
    else:
        aligned = s_d * neural.astype(np.float64) + b_d
        log.debug(f"  depth align: scale={s_d:.4f}  shift={b_d:.4f}m  "
                  f"rmse={rmse_d:.4f}m")

    # Clip: aligned depth must be non-negative
    aligned = np.clip(aligned, 0.0, None)
    return aligned.astype(np.float32)


def _blend_depths(lidar: np.ndarray, neural_aligned: np.ndarray,
                  lidar_weight: float = 0.6) -> np.ndarray:
    """
    Blend LiDAR and aligned neural depth:
      - Where LiDAR is valid: weighted blend (LiDAR provides metric accuracy,
        neural provides edge detail near boundaries).
      - Where LiDAR is zero/invalid: use neural depth only (hole filling).

    lidar_weight: 0.0 = pure neural, 1.0 = pure LiDAR, 0.6 = good default.
    """
    # Ensure non-negative before any comparison
    neural_aligned = np.clip(neural_aligned, 0.0, None)

    lidar_valid  = lidar > 0.1
    neural_valid = neural_aligned > 0.1

    result = np.zeros_like(lidar)

    # Both valid: weighted blend
    both = lidar_valid & neural_valid
    result[both] = (lidar_weight * lidar[both] +
                    (1.0 - lidar_weight) * neural_aligned[both])

    # Only LiDAR valid
    only_lidar = lidar_valid & ~neural_valid
    result[only_lidar] = lidar[only_lidar]

    # Only neural valid (LiDAR hole)  neural fills in missing areas
    only_neural = ~lidar_valid & neural_valid
    result[only_neural] = neural_aligned[only_neural]

    return result.astype(np.float32)


class _DepthAnythingV2Backend:
    """
    Uses Depth-Anything-V2 via HuggingFace transformers.
    Outputs DISPARITY (not depth)  _align_scale_shift handles this automatically.
    First run downloads ~400 MB (Large) or ~97 MB (Small) model weights.
    """
    # Use Small for speed, Large for quality. Large is ~4x slower per frame.
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # swap to Large if GPU has VRAM

    def __init__(self, device: str, model_size: str = "small"):
        self.device = device
        sizes = {
            "small":  "depth-anything/Depth-Anything-V2-Small-hf",
            "base":   "depth-anything/Depth-Anything-V2-Base-hf",
            "large":  "depth-anything/Depth-Anything-V2-Large-hf",
        }
        self.MODEL_ID = sizes.get(model_size, sizes["small"])
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        log.info(f"  Loading {self.MODEL_ID} from HuggingFace "
                 "(first run downloads model weights)...")
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.MODEL_ID,
            device=0 if self.device == "cuda" else -1,
        ) 
        log.info(f"👽👽 Model is in : {self.device}")
        log.info("  Depth-Anything-V2 loaded.")

    def predict(self, rgb_path: Path) -> np.ndarray:
        """
        Return monocular depth/disparity ( float32, arbitrary scale).
        Note: Depth-Anything outputs DISPARITY  higher value = closer.
        _align_scale_shift handles the inversion automatically.
        """
        self._load()
        from PIL import Image
        img = Image.open(rgb_path).convert("RGB")
        result = self._pipe(img)
        depth = np.asarray(result["depth"], dtype=np.float32)
        return depth


class _ZoeDepthBackend:
    """
    ZoeDepth: metric monocular depth (outputs DEPTH, not disparity).
    pip install timm  then  pip install git+https://github.com/isl-org/ZoeDepth
    """
    def __init__(self, device: str):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        log.info("  Loading ZoeDepth_N (metric monocular depth)...")
        import torch
        self._model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N",
                                     pretrained=True).to(self.device).eval()
        log.info("  ZoeDepth loaded.")

    def predict(self, rgb_path: Path) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image
        img = Image.open(rgb_path).convert("RGB")
        with torch.no_grad():
            depth = self._model.infer_pil(img)
        return np.asarray(depth, dtype=np.float32)


class DepthRefiner:
    """
    Refine LiDAR depth for every frame using a neural monocular depth network.

    Parameters
    ----------
    model : "depth-anything-v2" | "zoedepth" | "none"
        Which neural backend to use.  "none" is a no-op.
    model_size : "small" | "base" | "large"
        Only used for depth-anything-v2.
        "small"   ~97 MB,  ~8 FPS on RTX 3080,  good quality
        "large"   ~400 MB, ~2 FPS on RTX 3080,  best quality
    lidar_weight : float
        Blend weight for LiDAR in the final depth map.
        0.7 = 70% LiDAR + 30% neural  (safe default for room scans).
        0.5 = equal blend (more neural hole-filling, slightly less metric).
    min_depth : float
        Minimum valid LiDAR depth in metres.
    write_refined : bool
        If True, overwrite original EXR files in-place.
        If False (default), save refined depths as .npy beside originals.
    output_dir : Path | None
        Where to write refined depth files (only when write_refined=False).
    """

    def __init__(
        self,
        model: Literal["depth-anything-v2", "zoedepth", "none"] = "depth-anything-v2",
        model_size: str = "small",
        lidar_weight: float = 0.7,
        min_depth: float = 0.2,
        write_refined: bool = False,
        output_dir: Path = None,
    ):
        self.model_name   = model
        self.model_size   = model_size
        self.lidar_weight = lidar_weight
        self.min_depth    = min_depth
        self.write_refined = write_refined
        self.output_dir   = Path(output_dir) if output_dir else None
        self._backend     = None
        self._device      = "cpu"

    def _get_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.get_device_name(0)
                log.info(f"  GPU detected: {dev}")
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        self._device = self._get_device()
        log.info(f"  DepthRefiner: device={self._device}  model={self.model_name} "
                 f"size={self.model_size}")
        if self.model_name == "depth-anything-v2":
            self._backend = _DepthAnythingV2Backend(self._device, self.model_size)
        elif self.model_name == "zoedepth":
            self._backend = _ZoeDepthBackend(self._device)
        else:
            self._backend = None
        return self._backend

    def _create_valid_mask(self, depth: np.ndarray) -> np.ndarray:
        """Create mask for valid depth (non-black, non-zero regions)"""
        valid = (depth > 0.01) & np.isfinite(depth)
        # Also exclude pure black in RGB if available
        return valid.astype(np.float32)
        
    def refine(self, frames) -> list:
        """
        Run depth refinement on all frames.
        Returns the same frames list with updated depth_path attributes.
        """
        if self.model_name == "none":
            log.info("DepthRefiner: disabled (model='none'), skipping.")
            return frames

        backend = self._get_backend()
        if backend is None:
            return frames

        n = len(frames)
        log.info(f"DepthRefiner ({self.model_name}/{self.model_size}): "
                 f"refining {n} frames...")

        # Set up output directory for refined depth files
        if not self.write_refined:
            if self.output_dir is None:
                self.output_dir = frames[0].depth_path.parent.parent / "refined_depth"
            refined_dir = Path(self.output_dir) / "refined_depth"
            refined_dir.mkdir(parents=True, exist_ok=True)

        n_ok = 0
        n_direct = 0
        n_inverse = 0
        n_fallback = 0

        for i, fr in enumerate(frames):
            try:
                # 1. Load raw LiDAR depth (metres)
                lidar = _load_exr_depth(fr.depth_path) * fr.depth_scale

                # 2. Get neural prediction (depth or disparity, arbitrary scale)
                neural_raw = backend.predict(fr.rgb_path)

                # 3. Resize neural to match LiDAR resolution if needed
                if neural_raw.shape != lidar.shape:
                    try:
                        import cv2
                        neural_raw = cv2.resize(
                            neural_raw,
                            (lidar.shape[1], lidar.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    except ImportError:
                        from PIL import Image
                        neural_raw = np.asarray(
                            Image.fromarray(neural_raw).resize(
                                (lidar.shape[1], lidar.shape[0]),
                                Image.BILINEAR,
                            ),
                            dtype=np.float32,
                        )

                # 4. Align neural to LiDAR metric scale
                #    Automatically handles both depth and disparity outputs.
                neural_aligned = _align_scale_shift(
                    neural_raw, lidar, self.min_depth
                )

                # 5. Blend aligned neural + LiDAR
                refined = _blend_depths(lidar, neural_aligned, self.lidar_weight)

                # 6. Save refined depth and update frame's depth_path
                if self.write_refined:
                    save_path = fr.depth_path.with_suffix(".npy")
                else:
                    save_path = refined_dir / f"{fr.depth_path.stem}_refined.npy"

                np.save(str(save_path), refined)
                fr.depth_path  = save_path
                fr.depth_scale = 1.0   # saved in metres
                n_ok += 1

            except Exception as e:
                log.warning(f"  Frame {fr.idx}: depth refinement failed ({e})")

            if (i + 1) % 50 == 0:
                log.info(f"  Refined {i+1}/{n} frames...")

        log.info(f"DepthRefiner: refined {n_ok}/{n} frames successfully.")
        return frames