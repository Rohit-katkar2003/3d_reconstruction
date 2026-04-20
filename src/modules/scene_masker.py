# src/modules/scene_masker.py  ── v3  SURGICAL - image region + YOLO only
"""
Minimal masker: only masks the specific regions that cause shards.
Does NOT touch the depth values themselves - only zeros out pixels
in known problem areas. No brightness thresholding (it was removing
too much valid geometry near bright walls).
"""
import logging
import numpy as np
import cv2
from pathlib import Path

log = logging.getLogger(__name__)


class SceneMasker:
    def __init__(self, output_dir: Path, use_yolo=True, person_dilate_px=20):
        self.out_depth_dir   = Path(output_dir) / "masked_depths"
        self.use_yolo        = use_yolo
        self.person_dilate_px = person_dilate_px
        self._yolo           = None
        self.out_depth_dir.mkdir(parents=True, exist_ok=True)
        if use_yolo:
            self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n-seg.pt")
            log.info("  SceneMasker: YOLOv8 loaded")
        except Exception as e:
            log.warning(f"  SceneMasker: YOLO unavailable ({e}), "
                        f"will only mask flying pixels")
            self._yolo = None

    def _load_depth_raw(self, path: Path, scale: float) -> np.ndarray:
        path = Path(path)
        if path.suffix == ".npy":
            return np.load(str(path)).astype(np.float32) * scale
        try:
            import OpenEXR, Imath
            f   = OpenEXR.InputFile(str(path))
            dw  = f.header()["dataWindow"]
            W   = dw.max.x - dw.min.x + 1
            H   = dw.max.y - dw.min.y + 1
            ch  = next(iter(f.header()["channels"].keys()))
            raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(raw, dtype=np.float32).reshape(H, W) * scale
        except Exception:
            pass
        from src.modules.exr_reader import read_exr_depth
        return read_exr_depth(path).astype(np.float32) * scale

    def _build_mask(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        keep = np.ones((H, W), dtype=bool)

        # ── 1. Zero invalid depth values only ────────────────────────────────
        keep[depth <= 0.05] = False   # sensor noise / no return
        keep[depth > 5.0]   = False   # beyond LiDAR range

        # ── 2. Flying pixel filter — pure depth gradient ──────────────────────
        # Only kills pixels where depth jumps > 0.3m in one pixel step.
        # This is geometrically impossible for a real surface.
        d = depth.copy()
        d[~keep] = 0
        dx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        dy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
        # Only kill the edge pixel itself, not its neighbors
        flying = (dx > 0.3) | (dy > 0.3)
        keep[flying] = False

        # ── 3. Person mask via YOLO — surgical, no dilation of bright areas ───
        if self._yolo is not None:
            try:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                results = self._yolo(bgr, classes=[0], verbose=False)
                for r in results:
                    if r.masks is None:
                        continue
                    for seg in r.masks.data:
                        m = seg.cpu().numpy()
                        m = cv2.resize(m, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                        pmask = m.astype(bool)
                        if self.person_dilate_px > 0:
                            k = self.person_dilate_px * 2 + 1
                            pmask = cv2.dilate(
                                pmask.astype(np.uint8),
                                cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (k, k))
                            ).astype(bool)
                        keep[pmask] = False
            except Exception as e:
                log.debug(f"  YOLO frame error: {e}")

        return keep

    def apply(self, frames):
        log.info(f"  SceneMasker: masking {len(frames)} frames → "
                 f"{self.out_depth_dir}")
        total_masked = 0.0

        for i, fr in enumerate(frames):
            out_path = self.out_depth_dir / (Path(fr.depth_path).stem + ".npy")

            if out_path.exists():
                fr.depth_path  = out_path
                fr.depth_scale = 1.0
                continue

            try:
                depth = self._load_depth_raw(fr.depth_path, fr.depth_scale)
                rgb   = cv2.cvtColor(
                    cv2.imread(str(fr.rgb_path)), cv2.COLOR_BGR2RGB)

                keep        = self._build_mask(rgb, depth)
                depth[~keep] = 0.0
                np.save(str(out_path), depth)

                fr.depth_path  = out_path
                fr.depth_scale = 1.0
                total_masked  += 1.0 - keep.mean()

            except Exception as e:
                log.warning(f"  SceneMasker frame {i}: {e}")

            if (i + 1) % 100 == 0:
                log.info(f"    {i+1}/{len(frames)}  "
                         f"avg_masked={100*total_masked/(i+1):.1f}%")

        log.info(f"  SceneMasker done. "
                 f"Avg masked: {100*total_masked/max(1,len(frames)):.1f}%")
        return frames