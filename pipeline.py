"""
pipeline.py — v9 "auto-tuned + mesh repair"
All parameters computed automatically. MeshCleaner runs after meshing.
No manual tuning needed when scan changes.

New in v9:
  - Stage 6b: MeshCleaner  (PyMeshLab + Open3D post-processing)
  - AutoTuner v2            (detects dynamic objects, blur, environment type)
  - cfg now feeds hints to MeshCleaner (no separate config)

Install new deps:
    pip install pymeshlab open3d trimesh
"""

import logging, os, signal, sys, time
from pathlib import Path
import numpy as np

from src.modules.data_loader      import DataLoader
from src.modules.colmap_db        import COLMAPDatabaseBuilder
from src.modules.depth_fusion     import DepthFusion, _load_exr
from src.modules.reconstruction   import Reconstruction
from src.modules.meshing          import Meshing
from src.modules.validator        import Validator
from src.modules.pose_smoother    import PoseSmoother
from src.modules.texture_baker    import TextureBaker
from src.modules.frame_filter     import FrameFilter
# from src.modules.depth_refiner    import DepthRefiner 
from src.modules.depth_refiner1 import DepthRefiner
from src.modules.normal_estimator import NormalEstimator
# from src.modules.auto_tuner       import AutoTuner      # v2
from src.modules.smart_tuner import AutoTuner 
from src.modules.mesh_cleaner     import MeshCleaner    # ★ NEW

os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

_STOP = False
def _sig(s, f):
    global _STOP
    if _STOP: os._exit(1)
    _STOP = True; log.warning("Ctrl+C — stopping. Press again to force.")
signal.signal(signal.SIGINT,  _sig)
signal.signal(signal.SIGTERM, _sig)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, lambda s, f: os._exit(1))

def chk(st):
    if _STOP: log.info(f"Stop before {st}"); sys.exit(0)

# ★ ONLY change this — everything else is automatic
DATA_SUBDIR = r"D:\Major Project\3d reconstruction\3d_reconstruction\Data\cabin"

# ── These never need to change ─────────────────────────────────────────────────
DEPTH_SCALE = 1.0
FRAME_SKIP  = 0
POSE_SMOOTH = 1
SKIP_COLMAP = True
BAKE        = True
TEX_SIZE    = 4096

USE_FRAME_FILTER     = True
USE_DEPTH_REFINER    = False  
USE_NORMAL_ESTIMATOR = False
USE_MESH_CLEANER     = True    # ★ NEW — set False only to debug raw mesh


def find_root(start, sub):
    c = Path(start).resolve()
    for _ in range(5):
        if (c / sub).is_dir(): return c
        c = c.parent
    raise RuntimeError(f"Cannot find {sub}")

def find_d(root, names):
    for n in names:
        p = root / n
        if p.is_dir(): return p

def find_f(root, names):
    for n in names:
        p = root / n
        if p.is_file(): return p

def discover(root):
    r = find_d(root, ["rgb", "images", "color", "frames"])
    d = find_d(root, ["depth", "depths", "depth_maps"])
    p = find_f(root, ["poses.txt", "pose.txt", "transforms.txt", "trajectory.txt"])
    if not r or not d or not p:
        raise RuntimeError(f"Missing rgb/depth/poses in {root}")
    return r, d, p


def main():
    root = find_root(Path(__file__).parent, DATA_SUBDIR)
    src  = root / DATA_SUBDIR
    out  = root / "output"
    out.mkdir(exist_ok=True)
    rgb_d, dep_d, pose_f = discover(src)
    log.info(f"Root:{root}  Data:{src}  Out:{out}")

    # ── Stage 1: Load ──────────────────────────────────────────────────────────
    chk("Load"); log.info("=== Stage 1/8 Load ===")
    frames = DataLoader(
        data_dir=src, rgb_subdir=rgb_d.name,
        depth_subdir=dep_d.name, pose_file=pose_f.name,
        depth_scale=DEPTH_SCALE,
    ).load_all()
    log.info(f"  {len(frames)} frames loaded")

    # ── Stage 1b: Frame filter ─────────────────────────────────────────────────
    if USE_FRAME_FILTER:
        chk("Filter"); log.info("=== Stage 1b Frame Filter ===")
        frames = FrameFilter(
            sharpness_threshold=0.15,
            pose_jump_threshold=0.25,
        ).filter(frames)
        log.info(f"  {len(frames)} frames after filter")

    # ── Stage 1c: AUTO-TUNE v2 ─────────────────────────────────────────────────
    chk("AutoTune"); log.info("=== Stage 1c Auto-Tune Parameters (v2) ===")
    cfg = AutoTuner(
        frames    = frames,
        rgb_dir   = src / rgb_d.name,
        depth_dir = src / dep_d.name,
        enable_verification = True,   # adds ~20s, gains ~5% accuracy
        output_dir = out,
    ).compute()

    log.info(f"  Detected: env={cfg['env_type']}  "
             f"dynamic_objects={cfg['has_dynamic_objects']}  "
             f"span={cfg['scan_span_m']}m")

    # ── Stage 2: Pose smoothing ────────────────────────────────────────────────
    chk("Smooth"); log.info("=== Stage 2/8 Smooth ===")
    if POSE_SMOOTH > 0:
        frames = PoseSmoother(window=POSE_SMOOTH).smooth(frames)
    log.info("  Done")

    # ── Stage 2b: Neural depth refinement ─────────────────────────────────────
    # if USE_DEPTH_REFINER:
    #     chk("DepthRefine"); log.info("=== Stage 2b Depth Refinement ===============")
    #     frames = DepthRefiner(
    #         model="depth-anything-v2",
    #         lidar_weight=0.85,
    #         output_dir=out,
    #     ).refine(frames)
    #     log.info("  Done")

    # In pipeline.py — change this block:
    if USE_DEPTH_REFINER:
        frames = DepthRefiner(
            model="depth-anything-v2",          # ← switch from depth-anything-v2
            lidar_weight=0.4,           # ← was 0.85 — trust the network more
            output_dir=out,
            ensemble_passes=4,          # Marigold: number of diffusion passes
            half_precision=True,        # saves VRAM
        ).refine(frames)

    # ── Stage 3: COLMAP DB ─────────────────────────────────────────────────────
    chk("COLMAP"); log.info("=== Stage 3/8 COLMAP DB ===")
    db = COLMAPDatabaseBuilder(
        output_dir=out,
        colmap_bin=r"D:\Major Project\3d reconstruction\3d_reconstruction\Data\bin\colmap.exe",
        fast_mode=True,
        mapper_timeout_sec=600,
    ).build(frames, skip_feature_matching=SKIP_COLMAP)
    log.info(f"  DB->{db.name}")

    # ── Stage 4: TSDF Fusion ───────────────────────────────────────────────────
    chk("Fuse"); log.info("=== Stage 4/8 TSDF Fusion ===")
    log.info(f"  depth=[{cfg['min_d']},{cfg['max_d']}]  "
             f"vox={cfg['vox']}  sdf_trunc={cfg['sdf_trunc_multiplier']}")

    print("🙌🙌👍👍")
    print(cfg)
    print("🐱‍🐉🐱‍💻🐱‍🐉")
    fusion = DepthFusion(
        output_dir   = out,
        depth_scale  = DEPTH_SCALE,
        min_depth    = cfg["min_d"],
        max_depth    = cfg["max_d"],
        voxel_size   = cfg["vox"],
        frame_skip   = FRAME_SKIP,
        stop_flag    = lambda: _STOP,
        scene_mode   = cfg["scene"],
        sdf_trunc_multiplier = cfg["sdf_trunc_multiplier"],

        use_bilateral              = cfg["use_bilateral"],
        use_border_mask            = True,
        border_frac                =  max(cfg["border_frac"], 0.06),
        use_sky_suppress           = cfg["use_sky_suppress"],
        sky_bright_thresh          = cfg["sky_bright_thresh"],
        sky_depth_min_m            = cfg["sky_depth_min_m"],
        use_flying_pixel_suppress  = True,
        flying_pixel_jump_thresh_m = min(cfg["flying_pixel_jump_thresh_m"], 0.02),
        use_inpaint                = True,
        inpaint_radius_px          = 3,
        use_spike_filter           = True,
        use_glass_weight           = True, ###
        use_soft_consistency       = True, ###
        use_dynamic_weight         = True,
        dynamic_motion_thresh_m    = cfg["dynamic_motion_thresh_m"],
        dynamic_min_weight         = cfg["dynamic_min_weight"],
        use_planar_snap            = True,
        planar_ransac_thresh       = cfg["planar_ransac_thresh"],
        n_planes                   = cfg["n_planes"],
        planar_snap_dist           = cfg["planar_snap_dist"],
        planar_remove_dist         = cfg["planar_remove_dist"],
        use_hole_fill              = False,
        max_hole_size              = max(cfg["max_hole_size"],10),
    )
    pcd_path = fusion.fuse(frames)
    log.info(f"  Point cloud -> {pcd_path.name}")

    if USE_NORMAL_ESTIMATOR:
        chk("Normals"); log.info("=== Stage 4b Normal Estimation ===")
        pcd_path = NormalEstimator(
            output_dir=out, model="dsine", frame_skip=2,
        ).estimate(pcd_path, frames)
        log.info("  Done")

    log.info("=== Stage 5/8 Sparse (skipped) ===")

    # ── Stage 6: Mesh ──────────────────────────────────────────────────────────
    chk("Mesh"); log.info("=== Stage 6/8 Mesh ===")
    mesher = Meshing(
        output_dir             = out,
        depth                  = cfg["mdepth"],
        target_faces           = cfg["mfaces"],
        density_quantile       = cfg["mq"],
        tsdf_voxel_size        = cfg["vox"],
        min_component_fraction = cfg["mcomp"],
        largest_component_ratio= cfg["mlargest"],
    )
    tsdf_mesh = getattr(fusion, "last_tsdf_mesh_path", None)
    if tsdf_mesh and Path(tsdf_mesh).is_file():
        mesh = mesher.cleanup_mesh(tsdf_mesh, frames=frames)
    else:
        mesh = mesher.mesh(pcd_path, frames=frames)
    log.info(f"  Raw mesh -> {mesh.name}")

    # ── Stage 6b: Mesh Cleaning ★ NEW ─────────────────────────────────────────
    if USE_MESH_CLEANER:
        chk("Clean"); log.info("=== Stage 6b/8 Mesh Cleaning ===")
        log.info(f"  PyMeshLab + Open3D repair  "
                 f"(smooth_passes={cfg['mesh_smooth_iter']}  "
                 f"dynamic={cfg['has_dynamic_objects']})")

        cleaner = MeshCleaner(
            output_dir          = out,
            # ── stages ──────────────────────────────────────────────────────
            use_outlier_removal = True,
            use_hole_fill       = True,
            use_smooth          = True,
            use_decimate        = True,
            use_watertight      = False,   # set True for game engine exports
            # ── hints from AutoTuner ─────────────────────────────────────────
            scan_span_m         = cfg["scan_span_m"],
            has_dynamic_objects = cfg["has_dynamic_objects"],
            target_faces        = cfg["mesh_target_faces"],
            # ── smoothing ────────────────────────────────────────────────────
            smooth_iterations   = cfg["mesh_smooth_iter"],
            smooth_lambda       = cfg["mesh_smooth_lambda"],
        )
        mesh = cleaner.clean(mesh)
        log.info(f"  Cleaned mesh -> {mesh.name}")

    # ── Stage 7: Texture ───────────────────────────────────────────────────────
    chk("Texture"); log.info("=== Stage 7/8 Texture ===")
    if BAKE:
        t0  = time.time()
        tex = TextureBaker(
            output_dir   = out,
            texture_size = 1024,   # ↓ big impact
            xatlas_threads = 2,    # ↓ memory
            frame_batch  = 16,
            bake_faces   = 30000,  # ↓ compute
        ).bake(mesh, frames, frame_skip=FRAME_SKIP)
        log.info(f"  Tex->{tex.name}  ({time.time()-t0:.1f}s)")
    else:
        tex = mesh

    log.info(f"Done -> {out}")
    log.info("Preview: sandbox.babylonjs.com")


if __name__ == "__main__":
    s1 = time.time()
    main()
    print(f"Total time: {time.time()-s1:.2f}s")