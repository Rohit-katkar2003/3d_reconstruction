"""
modules/reconstruction.py
Runs COLMAP point_triangulator (when poses are known) or mapper (full SfM).
Also exports the sparse model to the output directory.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


class Reconstruction:
    """
    Parameters
    ----------
    output_dir  : parent output directory
    colmap_bin  : COLMAP executable name / path
    """

    def __init__(self, output_dir: Path, colmap_bin: str = r"C:\COLMAP\colmap-x64-windows-cuda\bin\colmap.exe"):
        self.output_dir = Path(output_dir)
        self.colmap_bin = colmap_bin
        self.sparse_dir = self.output_dir / "sparse"

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, db_path: Path, frames, skip: bool = False) -> Path:
        """
        Run reconstruction.

        If skip=True (poses provided externally) we run point_triangulator
        to lift 2-D SIFT matches to 3-D using known poses, which gives a
        sparse point cloud aligned with the metric depth fusion.

        If skip=False we run the full incremental mapper.
        """
        if not self._colmap_available():
            log.warning("COLMAP binary not found – skipping sparse reconstruction.")
            return self.sparse_dir

        sparse_0 = self.sparse_dir / "0"
        sparse_0.mkdir(parents=True, exist_ok=True)

        image_dir = str(frames[0].rgb_path.parent.resolve())

        if skip:
            log.info("Running COLMAP point_triangulator with known poses…")
            self._triangulate(db_path, image_dir, sparse_0)
        else:
            log.info("Running COLMAP incremental mapper (full SfM)…")
            self._mapper(db_path, image_dir)

        return sparse_0

    # ── COLMAP calls ──────────────────────────────────────────────────────────

    def _triangulate(self, db_path: Path, image_dir: str, sparse_input: Path):
        cmd = [
            self.colmap_bin, "point_triangulator",
            "--database_path",    str(db_path),
            "--image_path",       image_dir,
            "--input_path",       str(sparse_input),
            "--output_path",      str(sparse_input),
            "--Mapper.ba_refine_focal_length",      "0",
            "--Mapper.ba_refine_principal_point",   "0",
            "--Mapper.ba_refine_extra_params",      "0",
        ]
        self._run(cmd)

    def _mapper(self, db_path: Path, image_dir: str):
        cmd = [
            self.colmap_bin, "mapper",
            "--database_path",  str(db_path),
            "--image_path",     image_dir,
            "--output_path",    str(self.sparse_dir),
            # Use prior poses as initialisation
            "--Mapper.ba_global_max_refinements",   "5",
            "--Mapper.min_num_matches",             "15",
            "--Mapper.init_min_num_inliers",        "50",
        ]
        self._run(cmd)

    # ── bundle adjustment (optional refinement) ───────────────────────────────

    def bundle_adjust(self, sparse_path: Path):
        """Optional: refine the sparse model with global bundle adjustment."""
        if not self._colmap_available():
            return
        cmd = [
            self.colmap_bin, "bundle_adjuster",
            "--input_path",   str(sparse_path),
            "--output_path",  str(sparse_path),
            "--BundleAdjustment.refine_focal_length",    "1",
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params",    "0",
        ]
        self._run(cmd)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _run(self, cmd: List[str]):
        log.info(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"COLMAP error (exit {result.returncode}):\n{result.stderr[-3000:]}")
            log.warning(
                "COLMAP reconstruction failed. "
                "Consider setting SKIP_COLMAP=True in pipeline.py to use "
                "provided ARKit poses directly and skip SfM entirely."
            )
        else:
            log.info(result.stdout[-300:] if result.stdout else "OK")

    def _colmap_available(self) -> bool:
        try:
            subprocess.run([self.colmap_bin, "help"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False