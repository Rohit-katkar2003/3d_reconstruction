"""
auto_tuner.py — v2: Automatic parameter computation from scan data.

Improvements over v1:
  - Detects dynamic objects (people/pets) and compensates
  - Measures scan sharpness / motion blur
  - Measures point cloud density distribution
  - Detects indoor vs outdoor
  - Detects scan completeness (are walls/floor/ceiling all present?)
  - Returns hints for MeshCleaner too (span, has_dynamic)

Drop this file into src/modules/ and call from pipeline.py:
    from src.modules.auto_tuner import AutoTuner
    cfg = AutoTuner(frames, rgb_dir, depth_dir).compute()
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class AutoTuner:
    """
    Measures the scan and returns a cfg dict with all parameters set correctly.

    v2 additions:
      - _measure_motion()       : detects moving objects (person in frame)
      - _measure_sharpness()    : detects blur / fast camera motion
      - _measure_scan_coverage(): checks if floor/walls/ceiling all present
      - _measure_environment()  : indoor / outdoor / garden detection
      - cfg now includes 'scan_span_m' and 'has_dynamic_objects' for MeshCleaner
    """

    SAMPLE_N = 40

    def __init__(self, frames, rgb_dir: Path, depth_dir: Path):
        self.frames    = frames
        self.rgb_dir   = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self._sample   = self._pick_sample()

    # ── public ────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        log.info("AutoTuner v2: measuring scan…")

        room    = self._measure_room()
        depth   = self._measure_depth()
        bright  = self._measure_brightness()
        noise   = self._measure_depth_noise()
        cover   = self._measure_ceiling_coverage()
        motion  = self._measure_motion()
        sharp   = self._measure_sharpness()
        env     = self._measure_environment(bright, depth)
        scan_ok = self._measure_scan_coverage(room, depth)

        log.info(f"  room_span={room['span']:.2f}m  "
                 f"depth=[{depth['p5']:.2f},{depth['p95']:.2f}]m  "
                 f"bright_frac={bright['blown_frac']:.3f}  "
                 f"noise={noise['edge_var']:.4f}  "
                 f"ceiling_ok={cover['ceiling_ok']}")
        log.info(f"  motion={motion['has_dynamic']:.3f}  "
                 f"sharpness={sharp['mean_sharpness']:.1f}  "
                 f"env={env['type']}  "
                 f"scan_complete={scan_ok['complete']}")

        cfg = self._build_cfg(room, depth, bright, noise, cover,
                              motion, sharp, env, scan_ok)
        self._log_cfg(cfg)
        return cfg

    # ── measurement methods ───────────────────────────────────────────────────

    def _pick_sample(self):
        n = min(self.SAMPLE_N, len(self.frames))
        return random.sample(self.frames, n)

    def _measure_room(self) -> dict:
        pos  = np.stack([f.c2w[:3, 3] for f in self.frames])
        ext  = pos.max(0) - pos.min(0)
        span = float(ext.max())
        vol  = float(ext[0] * ext[1] * ext[2])
        return {"span": span, "extents": ext, "volume": vol}

    def _measure_depth(self) -> dict:
        all_vals = []
        for fr in self._sample:
            d = self._load_depth(fr)
            if d is None:
                continue
            valid = d[(d > 0.05) & (d < 20.0)]
            if valid.size > 0:
                all_vals.append(valid)

        if not all_vals:
            return {"p1": 0.10, "p5": 0.15, "p95": 4.0, "p99": 5.0,
                    "median": 2.0, "has_far": False}

        flat = np.concatenate(all_vals)
        p1, p5, p50, p95, p99 = np.percentile(flat, [1, 5, 50, 95, 99])
        return {
            "p1": float(p1), "p5": float(p5), "median": float(p50),
            "p95": float(p95), "p99": float(p99),
            "has_far": float(p95) > 3.5,
        }

    def _measure_brightness(self) -> dict:
        blown_fracs, mean_brights = [], []
        for fr in self._sample:
            img = self._load_rgb(fr)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blown_fracs.append(float((gray > 230).mean()))
            mean_brights.append(float(gray.mean()))

        if not blown_fracs:
            return {"blown_frac": 0.0, "mean_bright": 128.0,
                    "has_window": False, "sky_thresh": 220}

        avg_blown  = float(np.mean(blown_fracs))
        avg_bright = float(np.mean(mean_brights))
        has_window = avg_blown > 0.01
        sky_thresh = int(np.clip(220 - (avg_bright - 100) * 0.3, 190, 235))
        return {
            "blown_frac": avg_blown, "mean_bright": avg_bright,
            "has_window": has_window, "sky_thresh": sky_thresh,
        }

    def _measure_depth_noise(self) -> dict:
        variances = []
        for fr in self._sample:
            d = self._load_depth(fr)
            if d is None:
                continue
            d_clip = np.clip(d, 0, 5.0).astype(np.float32)
            sx = cv2.Sobel(d_clip, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(d_clip, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(sx**2 + sy**2)
            moderate = mag[(mag > 0.05) & (mag < 1.0)]
            if moderate.size > 0:
                variances.append(float(moderate.var()))

        edge_var = float(np.mean(variances)) if variances else 0.02
        return {
            "edge_var":   edge_var,
            "noisy":      edge_var > 0.05,
            "very_noisy": edge_var > 0.15,
        }

    def _measure_ceiling_coverage(self) -> dict:
        up_count = 0
        for fr in self._sample:
            R = fr.c2w[:3, :3]
            cam_forward_world = R @ np.array([0, 0, -1])
            up_y = float(cam_forward_world[1])
            up_z = float(cam_forward_world[2])
            if up_y > 0.3 or up_z > 0.3:
                up_count += 1
        ceiling_frac = up_count / max(len(self._sample), 1)
        return {
            "ceiling_frac": ceiling_frac,
            "ceiling_ok":   ceiling_frac > 0.05,
        }

    # ── NEW v2 measurements ───────────────────────────────────────────────────

    def _measure_motion(self) -> dict:
        """
        Detect temporal inconsistency between consecutive depth frames.
        High variance between consecutive frames at same pixel = moving object.
        This catches people, pets, fans, anything that moved during scan.
        """
        # Take consecutive pairs from sample
        sorted_sample = sorted(self._sample, key=lambda f: getattr(f, 'index', 0))
        pairs = list(zip(sorted_sample[:-1], sorted_sample[1:]))
        random.shuffle(pairs)
        pairs = pairs[:15]   # check 15 pairs max

        motion_fracs = []
        for fa, fb in pairs:
            da = self._load_depth(fa)
            db = self._load_depth(fb)
            if da is None or db is None:
                continue
            # Resize to same shape if needed
            if da.shape != db.shape:
                db = cv2.resize(db, (da.shape[1], da.shape[0]))
            valid_mask = (da > 0.1) & (db > 0.1) & (da < 8.0) & (db < 8.0)
            if valid_mask.sum() < 1000:
                continue
            diff = np.abs(da - db)
            # Large diff (> 0.15m) between consecutive frames = motion
            motion_frac = float((diff[valid_mask] > 0.15).mean())
            motion_fracs.append(motion_frac)

        avg_motion = float(np.mean(motion_fracs)) if motion_fracs else 0.0
        has_dynamic = avg_motion > 0.05  # >5% of pixels changed = someone moving

        log.info(f"    motion: avg_frac={avg_motion:.3f}  has_dynamic={has_dynamic}")
        return {
            "motion_frac": avg_motion,
            "has_dynamic": avg_motion,      # float, used for scaling
            "has_dynamic_objects": has_dynamic,  # bool
        }

    def _measure_sharpness(self) -> dict:
        """
        Laplacian variance of RGB images → how sharp/blurry the scan is.
        Blurry scan = camera moved fast → need looser fusion thresholds.
        """
        sharpness_vals = []
        for fr in self._sample:
            img = self._load_rgb(fr)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize to speed up
            small = cv2.resize(gray, (320, 240))
            lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
            sharpness_vals.append(lap_var)

        if not sharpness_vals:
            return {"mean_sharpness": 500.0, "blurry": False}

        mean_sharp = float(np.median(sharpness_vals))
        # < 100 = very blurry (camera was moving fast or shaking)
        # > 500 = sharp (slow careful scan)
        return {
            "mean_sharpness": mean_sharp,
            "blurry": mean_sharp < 150,
            "very_blurry": mean_sharp < 60,
        }

    def _measure_environment(self, bright, depth) -> dict:
        """
        Classify environment: indoor_small / indoor_large / outdoor / garden.
        Used to pick fundamentally different parameter sets.
        """
        if depth["p95"] > 6.0 and bright["blown_frac"] > 0.05:
            env_type = "outdoor"
        elif depth["p95"] > 6.0:
            env_type = "large_indoor"   # warehouse, hall, gym
        elif bright["blown_frac"] > 0.02:
            env_type = "indoor_windowed"
        else:
            env_type = "indoor_small"   # bedroom, office, cabin

        return {"type": env_type}

    def _measure_scan_coverage(self, room, depth) -> dict:
        """
        Estimate if the scan is geometrically complete.
        Incomplete scans need more aggressive hole filling.
        """
        # Check if depth range covers expected room extents
        # If the max depth << room span, the scanner didn't see far walls
        depth_range = depth["p95"] - depth["p5"]
        room_span   = room["span"]

        # A complete room scan should have depth_range ≈ room_span
        completeness = min(1.0, depth_range / max(room_span, 0.1))

        return {
            "completeness": completeness,
            "complete":     completeness > 0.5,
            "needs_big_hole_fill": completeness < 0.4,
        }

    # ── parameter builder ─────────────────────────────────────────────────────

    def _build_cfg(self, room, depth, bright, noise, cover,
                   motion, sharp, env, scan_ok) -> dict:
        span    = room["span"]
        env_t   = env["type"]
        dynamic = motion["has_dynamic_objects"]
        blurry  = sharp["blurry"]

        log.info(f"  Building cfg for: span={span:.1f}m  env={env_t}  "
                 f"dynamic={dynamic}  blurry={blurry}")

        # ── Voxel size ─────────────────────────────────────────────────────
        # Blurry scans: slightly larger voxel to hide blur artefacts
        if span < 3.0:
            vox = 0.025 if not blurry else 0.030
        elif span < 6.0:
            vox = 0.031 if not blurry else 0.036
        elif span < 10.0:
            vox = 0.037 if not blurry else 0.040
        else:
            vox = 0.041 if not blurry else 0.046

        if env_t == "outdoor":
            vox = max(vox, 0.040)   # outdoor = larger features

        # ── Depth range ────────────────────────────────────────────────────
        min_d = max(0.10, depth["p1"] * 0.8)
        max_d = min(8.0,  depth["p95"] * 1.15)

        if bright["has_window"] and depth["has_far"]:
            max_d = min(max_d, depth["median"] * 2.5)

        if env_t == "outdoor":
            max_d = min(15.0, depth["p99"] * 1.1)

        # ── Dynamic objects: tighten depth to exclude person ───────────────
        # If there's a person, they likely appear in the 0.5-2.0m range
        # We use dynamic_min_weight to blur them rather than exclude them
        dynamic_min_weight = 0.3 if not dynamic else 0.15
        dynamic_motion_thresh = 0.08 if dynamic else 0.12

        # ── SDF truncation ─────────────────────────────────────────────────
        if span < 3.0:
            sdf_trunc = 2.5
        elif span < 6.0:
            sdf_trunc = 3.5
        else:
            sdf_trunc = 5.0

        # ── Poisson depth ──────────────────────────────────────────────────
        if env_t == "outdoor":
            mdepth = 13
        elif span < 3.0:
            mdepth = 10
        elif span < 6.0:
            mdepth = 12
        else:
            mdepth = 13

        # ── Face count ─────────────────────────────────────────────────────
        mfaces = int(np.clip(span * 1_000_000, 2_000_000, 6_000_000))
        if env_t == "outdoor":
            mfaces = min(mfaces, 4_000_000)   # outdoor = less detail needed per m²

        # ── Density quantile ───────────────────────────────────────────────
        n = len(self.frames)
        if n < 300:
            mq = 0.0010
        elif n < 700:
            mq = 0.0005
        else:
            mq = 0.0002

        if dynamic:
            mq = mq * 1.5   # more aggressive culling when person was present

        # ── Component filter ───────────────────────────────────────────────
        # Dynamic scan = more shards from the person → remove more aggressively
        mcomp = 0.004 if not dynamic else 0.008

        # ── Inpaint radius ─────────────────────────────────────────────────
        if not cover["ceiling_ok"]:
            inpaint_r = 45
        elif scan_ok["needs_big_hole_fill"]:
            inpaint_r = 55
        else:
            inpaint_r = 25

        # ── Hole fill size ─────────────────────────────────────────────────
        base_hole = int(300 / (vox / 0.030))
        max_hole  = int(np.clip(base_hole, 100, 600))
        if scan_ok["needs_big_hole_fill"]:
            max_hole = min(max_hole * 2, 800)

        # ── Flying pixel threshold ─────────────────────────────────────────
        if noise["very_noisy"]:
            flying_thresh = 0.15
        elif noise["noisy"]:
            flying_thresh = 0.12
        else:
            flying_thresh = 0.10

        # Blurry scan → more depth noise → loosen flying pixel threshold
        if blurry:
            flying_thresh *= 1.3

        # ── Border fraction ────────────────────────────────────────────────
        border_frac = 0.015 if not blurry else 0.025

        # ── Sky suppression ────────────────────────────────────────────────
        sky_thresh = bright["sky_thresh"]
        sky_depth  = max(1.2, min_d * 8.0)
        use_sky    = bright["has_window"] or env_t == "outdoor"

        # ── Planar snap ────────────────────────────────────────────────────
        n_planes = 6 if span < 5.0 else 8
        if env_t == "outdoor":
            n_planes = 4   # outdoor = ground plane + 3 rough surfaces

        # ── MeshCleaner hints (NEW) ────────────────────────────────────────
        # These are passed through to MeshCleaner
        mesh_smooth_iter   = 3 if not noise["noisy"] else 5
        mesh_smooth_lambda = 0.0 if not noise["noisy"] else 0.1
        mesh_target_faces  = mfaces

        return dict(
            # Core fusion params
            min_d    = round(min_d, 3),
            max_d    = round(max_d, 3),
            vox      = vox,
            scene    = True,
            sdf_trunc_multiplier = sdf_trunc,

            # Meshing
            mdepth   = mdepth,
            mfaces   = mfaces,
            mq       = round(mq, 5),
            mcomp    = mcomp,
            mlargest = 0.04,

            # Depth fusion
            use_bilateral              = True,
            border_frac                = border_frac,
            use_sky_suppress           = use_sky,
            sky_bright_thresh          = sky_thresh,
            sky_depth_min_m            = round(sky_depth, 2),
            flying_pixel_jump_thresh_m = round(flying_thresh, 3),
            inpaint_radius_px          = inpaint_r,
            max_hole_size              = max_hole,
            dynamic_min_weight         = dynamic_min_weight,
            dynamic_motion_thresh_m    = dynamic_motion_thresh,
            n_planes                   = n_planes,
            planar_snap_dist           = 0.06,
            planar_remove_dist         = 0.10,
            planar_ransac_thresh       = 0.020,

            # ── NEW: MeshCleaner hints ─────────────────────────────────────
            scan_span_m          = round(span, 2),
            has_dynamic_objects  = dynamic,
            env_type             = env_t,
            mesh_smooth_iter     = mesh_smooth_iter,
            mesh_smooth_lambda   = mesh_smooth_lambda,
            mesh_target_faces    = mesh_target_faces,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_depth(self, fr):
        try:
            p = Path(fr.depth_path) if hasattr(fr, "depth_path") else None
            if p is None or not p.exists():
                stem = Path(fr.rgb_path).stem
                for ext in [".png", ".exr", ".npy", ".tiff"]:
                    candidate = self.depth_dir / (stem + ext)
                    if candidate.exists():
                        p = candidate
                        break
            if p is None:
                return None
            if p.suffix == ".npy":
                d = np.load(str(p)).astype(np.float32)
            elif p.suffix == ".exr":
                d = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH)
                if d is None:
                    return None
                d = d.astype(np.float32)
            else:
                d = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH)
                if d is None:
                    return None
                d = d.astype(np.float32)
                if d.max() > 100:
                    d = d / 1000.0
            return d
        except Exception as e:
            log.debug(f"  depth load failed for {fr}: {e}")
            return None

    def _load_rgb(self, fr):
        try:
            p = Path(fr.rgb_path)
            if not p.exists():
                return None
            return cv2.imread(str(p))
        except Exception:
            return None

    def _log_cfg(self, cfg: dict):
        log.info("AutoTuner v2 computed config:")
        log.info(f"  depth=[{cfg['min_d']}, {cfg['max_d']}]m  "
                 f"vox={cfg['vox']}  sdf_trunc={cfg['sdf_trunc_multiplier']}")
        log.info(f"  mdepth={cfg['mdepth']}  mfaces={cfg['mfaces']:,}  mq={cfg['mq']}")
        log.info(f"  inpaint_r={cfg['inpaint_radius_px']}px  "
                 f"max_hole={cfg['max_hole_size']}  "
                 f"sky_thresh={cfg['sky_bright_thresh']}")
        log.info(f"  flying_thresh={cfg['flying_pixel_jump_thresh_m']}m  "
                 f"dynamic_min_w={cfg['dynamic_min_weight']}")
        log.info(f"  env={cfg['env_type']}  "
                 f"has_dynamic={cfg['has_dynamic_objects']}  "
                 f"mesh_smooth={cfg['mesh_smooth_iter']} passes")