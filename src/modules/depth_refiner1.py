# src/modules/depth_refiner1.py — fixed for path-based frames

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class DepthRefiner:
    SUPPORTED = ["depth-pro", "marigold", "unidepth-v2", "depth-anything-v2"]

    def __init__(self, model="depth-pro", lidar_weight=0.3,
                 output_dir="output", ensemble_passes=4, half_precision=True):
        self.model_name = model
        self.lidar_w    = lidar_weight
        self.mono_w     = 1.0 - lidar_weight
        self.out        = Path(output_dir)
        self.passes     = ensemble_passes
        self.dtype      = torch.float16 if half_precision else torch.float32
        self._pipe      = None
        self._transform = None

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: read a frame dict's RGB as uint8 HxWx3 numpy
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_rgb(frame: dict) -> np.ndarray:
        path = frame.get("rgb_path") or frame.get("color_path") or frame.get("image_path")
        if path is None:
            raise KeyError(f"No rgb/color/image path in frame keys: {list(frame.keys())}")
        img = Image.open(str(path)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: read a frame dict's depth as float32 HxW in metres
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_depth(frame: dict) -> np.ndarray:
        path = frame.get("depth_path")
        if path is None:
            raise KeyError(f"No depth_path in frame keys: {list(frame.keys())}")
        path = Path(path)
        scale = frame.get("depth_scale", 1.0)

        suffix = path.suffix.lower()

        if suffix in (".png", ".tiff", ".tif"):
            # 16-bit PNG is the most common iOS depth format
            raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                raise IOError(f"cv2 could not read depth: {path}")
            depth = raw.astype(np.float32) * scale

        elif suffix == ".exr":
            depth = DepthRefiner._load_exr(path, scale)

        elif suffix == ".npy":
            depth = np.load(str(path)).astype(np.float32) * scale

        else:
            # fallback: try PIL (works for 16-bit TIFF etc.)
            raw = np.array(Image.open(str(path)))
            depth = raw.astype(np.float32) * scale

        return depth

    # ──────────────────────────────────────────────────────────────────────────
    # EXR loader — tries multiple backends in order
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_exr(path: Path, scale: float = 1.0) -> np.ndarray:
        path = Path(path)

        # ── Strategy 1: imageio with freeimage plugin (most reliable on Windows)
        try:
            import imageio
            raw = imageio.v3.imread(str(path))          # returns HxW or HxWxC float
            if raw.ndim == 3:
                raw = raw[..., 0]                       # take first channel
            return raw.astype(np.float32) * scale
        except Exception as e1:
            pass

        # ── Strategy 2: OpenCV with EXR env var forced
        try:
            import os, cv2
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            raw = cv2.imread(str(path),
                             cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                return raw.astype(np.float32) * scale
            # try as colour then convert
            raw = cv2.imread(str(path),
                             cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if raw is not None:
                return raw[..., 0].astype(np.float32) * scale
        except Exception as e2:
            pass

        # ── Strategy 3: OpenEXR python binding
        try:
            import OpenEXR, Imath, array as _arr
            f  = OpenEXR.InputFile(str(path))
            dw = f.header()["dataWindow"]
            w  = dw.max.x - dw.min.x + 1
            h  = dw.max.y - dw.min.y + 1
            # try Y (luminance / single-channel depth), then Z, then R
            for ch in ("Y", "Z", "R"):
                if ch in f.header().get("channels", {}):
                    raw = _arr.array(
                        "f", f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                    )
                    return (np.frombuffer(raw, dtype=np.float32)
                            .reshape(h, w) * scale)
        except Exception as e3:
            pass

        # ── Strategy 4: read raw bytes as float32 (last resort)
        try:
            # EXR files: skip the 8-byte magic + header and pray it's packed floats.
            # This only works for very simple single-channel EXRs.
            data = np.fromfile(str(path), dtype=np.float32)
            # guess square-ish shape from typical iOS depth resolutions
            for h, w in [(192, 256), (256, 192), (480, 640), (640, 480),
                         (384, 512), (512, 384)]:
                if data.size >= h * w:
                    return data[:h * w].reshape(h, w) * scale
        except Exception as e4:
            pass

        raise IOError(
            f"Cannot open EXR file: {path}\n"
            "Fix options:\n"
            "  1) pip install imageio[freeimage]  then  imageio.plugins.freeimage.download()\n"
            "  2) pip install openexr  (needs Visual C++ build tools on Windows)\n"
            "  3) Set OPENCV_IO_ENABLE_OPENEXR=1 before launching Python\n"
            "     and rebuild/reinstall opencv-python from source with EXR support\n"
            "  4) Convert your .exr depths to 16-bit PNG beforehand:\n"
            "     python -c \"import imageio, glob, numpy as np; "
            "[imageio.imwrite(p.replace('.exr','.png'), "
            "(imageio.v3.imread(p)[...,0]*1000).astype('uint16')) "
            "for p in glob.glob(r'your\\depth\\*.exr')]\""
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────
    def _load(self):
        if self._pipe is not None:
            return
        m = self.model_name

        if m == "depth-pro":
            from depth_pro import create_model_and_transforms
            self._pipe, self._transform = create_model_and_transforms()
            self._pipe.eval().cuda()

        elif m == "marigold":
            from diffusers import MarigoldDepthPipeline
            self._pipe = MarigoldDepthPipeline.from_pretrained(
                "prs-eth/marigold-lcm-v1-0", torch_dtype=self.dtype
            ).to("cuda")

        elif m == "unidepth-v2":
            from unidepth.models import UniDepthV2
            self._pipe = UniDepthV2.from_pretrained(
                "lpiccinelli-eth/unidepth-v2-vitl14"
            ).to("cuda")

        elif m == "depth-anything-v2":
            from transformers import pipeline
            self._pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device=0,   # -1 = cuda; set 0 if you have a CUDA GPU
            )
        else:
            raise ValueError(f"Unknown model: {m}. Supported: {self.SUPPORTED}")

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────
    def _infer(self, rgb_np: np.ndarray, intrinsics=None) -> np.ndarray:
        m = self.model_name

        if m == "depth-pro":
            rgb_np = cv2.resize(rgb_np, (640, 480))
            img_t = self._transform(Image.fromarray(rgb_np)).cuda()

            with torch.no_grad():
                pred = self._pipe.infer(img_t)

            return pred["depth"].squeeze().cpu().numpy().astype(np.float32)

        elif m == "marigold": 
            pil = Image.fromarray(rgb_np)
            out = self._pipe(pil, num_inference_steps=self.passes, ensemble_size=1)
            return np.array(out.depth, dtype=np.float32)

        elif m == "unidepth-v2":
            rgb_t = (torch.from_numpy(rgb_np)
                     .permute(2, 0, 1).unsqueeze(0).float().cuda())
            cam = (torch.tensor(intrinsics).unsqueeze(0).cuda()
                   if intrinsics is not None else None)
            with torch.no_grad():
                out = self._pipe.infer(rgb_t, cam)
            return out["depth"].squeeze().cuda().numpy().astype(np.float32)

        elif m == "depth-anything-v2":
            rgb_np = cv2.resize(rgb_np, (640, 480)) 
            res = self._pipe(Image.fromarray(rgb_np))
            return np.array(res["depth"], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Blend LiDAR + monocular
    # ──────────────────────────────────────────────────────────────────────────
    def _blend(self, lidar: np.ndarray, mono: np.ndarray) -> np.ndarray:
        """Scale-align mono to LiDAR then do a confidence-weighted blend."""
        # resize mono to match lidar if shapes differ
        if mono.shape != lidar.shape:
            mono = cv2.resize(mono, (lidar.shape[1], lidar.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        valid = (lidar > 0.1) & (lidar < 20.0)   # valid LiDAR pixels
        if valid.sum() > 100:
            l_vals = lidar[valid].reshape(-1, 1)
            m_vals = mono[valid].reshape(-1, 1)
            # least-squares: mono_scaled = a*mono + b  ≈  lidar
            A = np.hstack([m_vals, np.ones_like(m_vals)])
            x, _, _, _ = np.linalg.lstsq(A, l_vals, rcond=None)
            a, b = float(x[0]), float(x[1])
            mono_scaled = np.clip(a * mono + b, 0.0, None)
        else:
            print("    ⚠ Too few valid LiDAR pixels — using mono only")
            mono_scaled = mono

        lidar_conf = valid.astype(np.float32) * self.lidar_w
        mono_conf  = np.ones_like(lidar_conf) * self.mono_w
        total      = lidar_conf + mono_conf
        blended    = (lidar * lidar_conf + mono_scaled * mono_conf) / total
        return blended.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point — works with both dict frames AND object frames
    # ──────────────────────────────────────────────────────────────────────────
    def refine(self, frames: list) -> list:
        self._load()
        n = len(frames)

        for i, fr in enumerate(frames):
            # ── support both dict and object-style frames ──────────────────
            is_dict = isinstance(fr, dict)

            # Load RGB
            if is_dict:
                rgb = self._load_rgb(fr)
            else:
                rgb = (getattr(fr, "rgb", None)
                       or getattr(fr, "color", None)
                       or getattr(fr, "image", None))
                if rgb is None:
                    rgb = self._load_rgb({"rgb_path": getattr(fr, "rgb_path", None)})

            # Load LiDAR depth
            if is_dict:
                lidar = self._load_depth(fr)
            else:
                lidar = (getattr(fr, "depth", None)
                         or getattr(fr, "depth_map", None))
                if lidar is None:
                    lidar = self._load_depth(
                        {"depth_path": getattr(fr, "depth_path", None),
                         "depth_scale": getattr(fr, "depth_scale", 1.0)}
                    )

            intrinsics = (fr.get("intrinsics") if is_dict
                          else getattr(fr, "intrinsics", None))

            # Monocular inference + blend
            mono    = self._infer(rgb, intrinsics)
            blended = self._blend(lidar, mono)

            # Write blended depth back
            if is_dict:
                # Store the refined array directly — downstream consumers
                # should check for "depth_array" before re-reading the file.
                fr["depth_array"] = blended
            else:
                if hasattr(fr, "depth"):
                    fr.depth = blended
                elif hasattr(fr, "depth_map"):
                    fr.depth_map = blended

            if i % 20 == 0:
                print(f"  Refined {i}/{n}  ({self.model_name})")

        print(f"  DepthRefiner done — {n} frames processed")
        return frames