"""
modules/colmap_db.py  — fixed version 5

Fixes vs v4:
  1. COLMAP mapper requires image `name` to be RELATIVE (filename only),
     not absolute path. Absolute paths cause silent "Loading images..." crash.
  2. image_list_path must also use relative filenames to match DB entries.
  3. GPU flag detection now also checks for newer COLMAP flag styles.
  4. Mapper output is streamed in real-time so we see exactly where it fails.
"""

import logging
import os
import sqlite3
import struct
import subprocess
from pathlib import Path

import numpy as np
COLMAP_PATH_L = os.getenv("COLMAP_PATH" , "D:\Major Project\3d reconstruction\3d_reconstruction\Data\bin\colmap.exe") 
print(f"👍👍 Colmap path : {COLMAP_PATH_L}")
log = logging.getLogger(__name__)
COLMAP_PINHOLE = 1


def _try_import_pycolmap():
    try:
        import pycolmap
        return pycolmap
    except ImportError:
        return None

def _find_qt_plugin_path(colmap_bin: str) -> str | None:
    """
    Walk up from colmap.exe looking for a 'platforms' folder that contains
    qwindows.dll or qoffscreen.dll. Returns the parent of 'platforms' so
    Qt can find it via QT_PLUGIN_PATH, or None if not found.
    """
    colmap_dir = Path(colmap_bin).resolve().parent
    # Search: same dir, ../plugins, ../../plugins, ../lib/qt/plugins, etc.
    candidates = [
        colmap_dir,
        colmap_dir / "plugins",
        colmap_dir.parent / "plugins",          # ★ bin/../plugins  ← COLMAP 4.x lives here
        colmap_dir.parent / "lib" / "qt" / "plugins",
        colmap_dir.parent / "lib" / "plugins",
        colmap_dir / "qt" / "plugins",
        colmap_dir / "Qt" / "plugins",
    ]
    for base in candidates:
        platforms = base / "platforms"
        if platforms.is_dir():
            dlls = list(platforms.glob("q*.dll"))
            if dlls:
                log.info(f"  Qt plugins found at: {base}  "
                         f"({[d.name for d in dlls[:4]]})")
                return str(base)
    return None


def _run(cmd, timeout=None, stream_log=False):
    """Run subprocess. Sets Qt env vars so COLMAP CLI can start headlessly."""
    colmap_bin = cmd[0]
    env = os.environ.copy()

    # ★ FIX: Don't force offscreen — it may not be compiled in.
    #   Instead point Qt at the plugin directory so it can find qwindows.dll,
    #   then request minimal platform. Fall back chain:
    #     1. minimal  (always compiled in, no window needed)
    #     2. offscreen (compiled in on some builds)
    #     3. windows  (needs display but at least tells us it loaded Qt)
    qt_plugin_path = _find_qt_plugin_path(colmap_bin)
    if qt_plugin_path:
        env["QT_PLUGIN_PATH"] = qt_plugin_path
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(Path(qt_plugin_path) / "platforms")

    # 'minimal' is a no-op platform compiled into every Qt build
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Suppress Qt warning spam
    env.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    if stream_log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            log.info(f"  [colmap] {line}")
        proc.wait()
        return proc.returncode, "\n".join(lines)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode, r.stdout + r.stderr


def _probe_mapper_prefix(colmap_bin):
    """Return 'Mapper' or 'mapper' depending on COLMAP version."""
    try:
        _, txt = _run([colmap_bin, "mapper", "--help"], timeout=15)
        if "--mapper." in txt and "--Mapper." not in txt:
            return "mapper"
    except Exception:
        pass
    return "Mapper"


class COLMAPDatabaseBuilder:
    def __init__(self, output_dir: Path,
                 colmap_bin: str = COLMAP_PATH_L,
                 fast_mode: bool = True,
                 mapper_timeout_sec: int = 900):
        self.output_dir = Path(output_dir)
        self.colmap_bin = colmap_bin
        self.fast_mode  = fast_mode
        self.mapper_timeout_sec = int(mapper_timeout_sec)
        self.db_path    = self.output_dir / "colmap.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.image_list = self.output_dir / "image_list.txt"

    def build(self, frames, skip_feature_matching: bool = False) -> Path:
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self._write_image_list(frames)

        pycolmap = _try_import_pycolmap()
        if pycolmap:
            self._build_with_pycolmap(frames, pycolmap)
        else:
            self._build_with_sqlite(frames)

        if not skip_feature_matching:
            self._run_colmap_pipeline(frames)
        else:
            log.info("Skipping COLMAP feature matching — using ARKit poses directly.")
            self._write_known_poses(frames)

        return self.db_path

    def _write_image_list(self, frames):
        """
        ★ FIX: Write RELATIVE filenames (basename only), not absolute paths.
        COLMAP mapper matches images by name against the image_path directory.
        If name = absolute path, COLMAP cannot find the file and silently fails
        at 'Loading images...' with return code 1.
        """
        with open(self.image_list, "w") as f:
            for fr in frames:
                # Relative name = just the filename, e.g. "frame_0001.jpg"
                f.write(fr.rgb_path.name + "\n")
        log.info(f"Image list -> {self.image_list}  (relative names)")

    def _build_with_pycolmap(self, frames, pycolmap):
        log.info("Using pycolmap to build database")
        db = pycolmap.Database(str(self.db_path))
        for fr in frames:
            intr = fr.intrinsics
            cam = pycolmap.Camera(
                model="PINHOLE", width=intr.width, height=intr.height,
                params=[intr.fx, intr.fy, intr.cx, intr.cy],
                camera_id=fr.idx + 1,
            )
            cam_id = db.write_camera(cam, use_camera_id=True)
            img = pycolmap.Image(
                name=fr.rgb_path.name,   # ★ relative name
                camera_id=cam_id, image_id=fr.idx + 1,
            )
            db.write_image(img, use_image_id=True)
        db.close()
        log.info(f"pycolmap DB -> {self.db_path}")

    def _build_with_sqlite(self, frames):
        """
        ★ FIX 1: All frames written (original wrote only last frame).
        ★ FIX 2: name = relative filename, not absolute path.
        ★ FIX 3: ARKit poses stored as prior_q/prior_t.
        """
        log.info("Using sqlite3 to build COLMAP database")
        if self.db_path.exists():
            self.db_path.unlink()

        con = sqlite3.connect(str(self.db_path))
        cur = con.cursor()
        self._create_schema(cur)

        for fr in frames:
            intr = fr.intrinsics
            params_blob = _pack_doubles([intr.fx, intr.fy, intr.cx, intr.cy])
            cur.execute(
                "INSERT OR REPLACE INTO cameras VALUES (?,?,?,?,?,1)",
                (fr.idx + 1, COLMAP_PINHOLE, intr.width, intr.height, params_blob),
            )

        for fr in frames:
            qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)
            cur.execute(
                "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    fr.idx + 1,
                    fr.rgb_path.name,   # ★ relative name only
                    fr.idx + 1,
                    qw, qx, qy, qz,
                    tx, ty, tz,
                ),
            )

        con.commit()
        con.close()
        log.info(f"sqlite3 DB -> {self.db_path}  ({len(frames)} images, relative names)")

    @staticmethod
    def _create_schema(cur):
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL, width INTEGER NOT NULL,
            height INTEGER NOT NULL, params BLOB,
            prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            image_id  INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name      TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx REAL, prior_ty REAL, prior_tz REAL
        );
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL,
            data BLOB, config INTEGER,
            F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
        );
        """)

    # ── COLMAP pipeline ───────────────────────────────────────────────────────

    def _run_colmap_pipeline(self, frames):
        if not self._colmap_available():
            log.warning("COLMAP binary not found — writing known poses as fallback.")
            self._write_known_poses(frames)
            return

        image_dir = str(Path(frames[0].rgb_path).parent)
        n = len(frames)
        log.info(f"COLMAP pipeline: {n} frames  image_dir={image_dir}")

        sift_gpu_on,  sift_gpu_off  = self._detect_sift_gpu_flag()
        match_gpu_on, match_gpu_off = self._detect_matcher_gpu_flag()

        # ── Step 1: Feature extraction ────────────────────────────────────────
        base_feat = [
            self.colmap_bin, "feature_extractor",
            "--database_path",                   str(self.db_path),
            "--image_path",                      image_dir,
            "--image_list_path",                 str(self.image_list),
            "--ImageReader.camera_model",        "PINHOLE",
            "--ImageReader.single_camera",       "1",
            "--SiftExtraction.max_num_features", "4096",
        ]

        feat_variants = []
        # GPU variant — only if flag was detected
        if sift_gpu_on:
            feat_variants.append(("GPU", base_feat + sift_gpu_on))
        # CPU variant — explicitly disable GPU so no OpenGL context needed
        if sift_gpu_off:
            feat_variants.append(("CPU", base_feat + sift_gpu_off))
        # Minimal fallback — no extra flags at all
        feat_variants.append(("CPU-minimal", [
            self.colmap_bin, "feature_extractor",
            "--database_path",   str(self.db_path),
            "--image_path",      image_dir,
            "--image_list_path", str(self.image_list),
        ]))

        log.info("Step 1/3: Feature extraction...")
        feat_ok = False
        for label, cmd in feat_variants:
            log.info(f"  Trying [{label}]: $ {' '.join(cmd)}")
            rc, out = _run(cmd)
            if rc == 0:
                log.info(f"  Feature extraction OK [{label}]")
                feat_ok = True
                break
            else:
                log.warning(f"  [{label}] failed: {out[-600:]}")

        if not feat_ok:
            log.error("All feature_extractor variants failed — falling back to ARKit poses.")
            self._write_known_poses(frames)
            return

        # ── Step 2: Sequential matching ───────────────────────────────────────
        overlap = min(12, max(6, n // 200)) if self.fast_mode else min(20, max(10, n // 100))
        log.info(f"Step 2/3: Sequential matching (overlap={overlap})...")
        seq_cmd = [
            self.colmap_bin, "sequential_matcher",
            "--database_path",                     str(self.db_path),
            "--SequentialMatching.overlap",        str(overlap),
            "--SequentialMatching.loop_detection", "0",
        ] + match_gpu_off  # always CPU for matching too — avoids OpenGL
        log.info(f"  $ {' '.join(seq_cmd)}")
        rc, out = _run(seq_cmd)
        if rc != 0:
            log.error(f"sequential_matcher failed:\n{out[-3000:]}")
            self._write_known_poses(frames)
            return
        log.info("  Sequential matching OK")

        # ── Step 3: Mapper — stream output so we see exactly what fails ───────
        sparse_0 = self.sparse_dir / "0"
        sparse_0.mkdir(parents=True, exist_ok=True)
        mp = _probe_mapper_prefix(self.colmap_bin)
        log.info(f"Step 3/3: Mapper (option prefix '--{mp}.')...")

        map_variants = [
            [
                self.colmap_bin, "mapper",
                "--database_path",                     str(self.db_path),
                "--image_path",                        image_dir,
                "--output_path",                       str(self.sparse_dir),
                f"--{mp}.ba_refine_focal_length",      "0",
                f"--{mp}.ba_refine_principal_point",   "0",
                f"--{mp}.ba_refine_extra_params",      "0",
                f"--{mp}.min_num_matches",             "15",
                f"--{mp}.init_min_num_inliers",        "50",
                f"--{mp}.multiple_models",             "0",
                f"--{mp}.max_num_models",              "1",
                f"--{mp}.num_threads",                 str(os.cpu_count() or 8),
                f"--{mp}.ba_local_max_num_iterations", "15",
                f"--{mp}.ba_global_max_num_iterations", "20",
                f"--{mp}.ba_global_images_ratio",      "1.4",
                f"--{mp}.ba_global_points_ratio",      "1.4",
                f"--{mp}.ba_global_images_freq",       "600",
                f"--{mp}.ba_global_points_freq",       "200000",
            ],
            [
                self.colmap_bin, "mapper",
                "--database_path", str(self.db_path),
                "--image_path",    image_dir,
                "--output_path",   str(self.sparse_dir),
            ],
        ]

        map_ok = False
        for i, cmd in enumerate(map_variants):
            label = "full-flags" if i == 0 else "minimal-flags"
            log.info(f"  Trying mapper [{label}]:")
            log.info(f"  $ {' '.join(cmd)}")
            # ★ stream_log=True: prints each COLMAP line as it runs
            #   so we see EXACTLY where it fails instead of silent crash
            rc, out = _run(cmd, timeout=self.mapper_timeout_sec, stream_log=True)
            if rc == 0:
                log.info(f"  Mapper OK [{label}]")
                map_ok = True
                break
            else:
                log.error(f"  Mapper [{label}] failed (rc={rc})")

        if not map_ok:
            log.warning("All mapper variants failed — falling back to ARKit poses only.")
            self._write_known_poses(frames)
            return

        self._export_sparse_txt(sparse_0)

    def _export_sparse_txt(self, sparse_dir: Path):
        txt_dir = sparse_dir.parent / "0_txt"
        txt_dir.mkdir(exist_ok=True)
        cmd = [
            self.colmap_bin, "model_converter",
            "--input_path",  str(sparse_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ]
        rc, out = _run(cmd)
        if rc == 0:
            log.info(f"  Sparse TXT model -> {txt_dir}")
        else:
            log.warning(f"model_converter failed (non-critical): {out[-300:]}")

    def _write_known_poses(self, frames):
        """Fallback: write ARKit poses as COLMAP images.txt without SfM."""
        images_txt  = self.sparse_dir / "0" / "images.txt"
        cameras_txt = self.sparse_dir / "0" / "cameras.txt"
        points_txt  = self.sparse_dir / "0" / "points3D.txt"
        images_txt.parent.mkdir(parents=True, exist_ok=True)

        with open(cameras_txt, "w") as f:
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            seen = set()
            for fr in frames:
                if fr.idx not in seen:
                    i = fr.intrinsics
                    f.write(f"{fr.idx+1} PINHOLE {i.width} {i.height} "
                            f"{i.fx} {i.fy} {i.cx} {i.cy}\n")
                    seen.add(fr.idx)

        with open(images_txt, "w") as f:
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for fr in frames:
                qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)
                f.write(f"{fr.idx+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                        f"{tx:.9f} {ty:.9f} {tz:.9f} {fr.idx+1} "
                        f"{fr.rgb_path.name}\n\n")

        points_txt.write_text("")
        log.info(f"Known poses written -> {images_txt.parent}")

    def _probe_colmap_version(self):
        """Returns True if this is COLMAP 4.x (new flag namespace)."""
        try:
            _, txt = _run([self.colmap_bin, "feature_extractor", "--help"], timeout=15)
            return "--FeatureExtraction.use_gpu" in txt
        except Exception:
            return False
            
    def _detect_sift_gpu_flag(self):
        """Detect correct GPU flag for this COLMAP version."""
        try:
            _, txt = _run([self.colmap_bin, "feature_extractor", "--help"], timeout=15)
        except Exception:
            return [], []

        # COLMAP 4.x — new top-level namespace
        if "--FeatureExtraction.use_gpu" in txt:
            log.info("  GPU flag: --FeatureExtraction.use_gpu (COLMAP 4.x)")
            return ["--FeatureExtraction.use_gpu", "1"], \
                ["--FeatureExtraction.use_gpu", "0"]

        # COLMAP 3.x — old SiftExtraction namespace
        if "--SiftExtraction.use_gpu" in txt:
            log.info("  GPU flag: --SiftExtraction.use_gpu (COLMAP 3.x)")
            return ["--SiftExtraction.use_gpu", "1"], \
                ["--SiftExtraction.use_gpu", "0"]

        log.warning("  No GPU flag found — will try without flag")
        return [], []

    def _detect_matcher_gpu_flag(self):
        """Detect correct matcher GPU flag for this COLMAP version."""
        try:
            _, txt = _run([self.colmap_bin, "sequential_matcher", "--help"], timeout=15)
        except Exception:
            return [], []

        if "--FeatureMatching.use_gpu" in txt:
            log.info("  Matcher GPU flag: --FeatureMatching.use_gpu (COLMAP 4.x)")
            return ["--FeatureMatching.use_gpu", "1"], \
                ["--FeatureMatching.use_gpu", "0"]

        if "--SiftMatching.use_gpu" in txt:
            log.info("  Matcher GPU flag: --SiftMatching.use_gpu (COLMAP 3.x)")
            return ["--SiftMatching.use_gpu", "1"], \
                ["--SiftMatching.use_gpu", "0"]

        log.warning("  No matcher GPU flag found")
        return [], []

    def _colmap_available(self) -> bool:
        try:
            subprocess.run([self.colmap_bin, "help"],
                           capture_output=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pack_doubles(values):
    return struct.pack(f"{len(values)}d", *values)


def _c2w_to_quat_trans(c2w: np.ndarray):
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s; qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s; qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s;               qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s; qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s
    return qw, qx, qy, qz, t[0], t[1], t[2]