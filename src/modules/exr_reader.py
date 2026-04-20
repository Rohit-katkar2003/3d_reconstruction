"""
modules/exr_reader.py
Pure-Python EXR reader for single-channel float32/float16 depth images.
No external EXR library needed — uses only Python stdlib + numpy.

Supports:
  - NONE (uncompressed)
  - ZIP  (zlib, 16 scanlines)
  - ZIPS (zlib, 1 scanline)
  - PIZ  (NOT supported — raises clear error)
  - PXR24 (NOT supported — raises clear error)

Bugs fixed vs original:
  1. _exr_zip_unpredict was called unconditionally — it now only runs for
     ZIP/ZIPS (compression 2 or 3), not for NONE. This was causing perfectly
     valid uncompressed floats to be scrambled into garbage.
  2. Channel layout: EXR stores channels CHANNEL-MAJOR within each scanline
     chunk (all pixels of channel A, then all of B …), not pixel-interleaved.
     The original _extract_channel interleaving loop was wrong for all layouts.
"""

import struct
import zlib
import numpy as np
from pathlib import Path

EXR_MAGIC   = 0x762F3101
COMPRESSION = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
               5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}
PIXEL_TYPE  = {0: "UINT", 1: "HALF", 2: "FLOAT"}


class EXRReadError(Exception):
    pass


def read_exr_depth(path) -> np.ndarray:
    """
    Read a single-channel EXR depth image.
    Returns float32 numpy array of shape (H, W) in the file's native units.
    """
    path = Path(path)
    raw  = path.read_bytes()

    # ── 1. Magic check ────────────────────────────────────────────────────────
    magic = struct.unpack_from("<I", raw, 0)[0]
    if magic != EXR_MAGIC:
        raise EXRReadError(f"Not a valid EXR file: {path} (magic={magic:#010x})")

    # ── 2. Parse header ───────────────────────────────────────────────────────
    pos        = 8  # skip magic + version
    width      = None
    height     = None
    compression = 0
    channels   = {}  

    while pos < len(raw):
        end      = raw.index(b'\x00', pos)
        attr_name = raw[pos:end].decode("utf-8", errors="replace")
        pos      = end + 1

        if attr_name == "":
            break

        end      = raw.index(b'\x00', pos)
        pos      = end + 1

        attr_size = struct.unpack_from("<I", raw, pos)[0]
        pos      += 4
        attr_data = raw[pos: pos + attr_size]
        pos      += attr_size

        if attr_name == "dataWindow":
            xmin, ymin, xmax, ymax = struct.unpack_from("<iiii", attr_data)
            width  = xmax - xmin + 1
            height = ymax - ymin + 1

        elif attr_name == "displayWindow" and width is None:
            xmin, ymin, xmax, ymax = struct.unpack_from("<iiii", attr_data)
            width  = xmax - xmin + 1
            height = ymax - ymin + 1

        elif attr_name == "compression":
            compression = attr_data[0]

        elif attr_name == "channels":
            p2 = 0
            while p2 < len(attr_data):
                end2 = attr_data.index(b'\x00', p2)
                ch_name = attr_data[p2:end2].decode("utf-8", errors="replace")
                p2 = end2 + 1
                if ch_name == "":
                    break
                ptype = struct.unpack_from("<i", attr_data, p2)[0]
                channels[ch_name] = ptype
                p2 += 16  # type(4) + reserved(4) + xSamp(4) + ySamp(4)

    if width is None or height is None:
        raise EXRReadError("Could not determine image dimensions from EXR header")
    if not channels:
        raise EXRReadError("No channels found in EXR header")

    # ── 3. Choose depth channel ───────────────────────────────────────────────
    preferred  = ["Z", "depth", "Y", "R", "G", "B"]
    ch_name    = next((c for c in preferred if c in channels), list(channels.keys())[0])
    pixel_type = channels[ch_name]
    bytes_per_pixel = 2 if pixel_type == 1 else 4

    comp_name = COMPRESSION.get(compression, str(compression))
    if compression not in (0, 2, 3):
        raise EXRReadError(
            f"Compression '{comp_name}' not supported by built-in reader.\n"
            f"  Install:  pip install openexr   or   conda install -c conda-forge openexr"
        )

    # ── 4. Read scanline offsets ──────────────────────────────────────────────
    lines_per_chunk = 16 if compression == 3 else 1   # ZIP=16, ZIPS/NONE=1
    n_chunks = (height + lines_per_chunk - 1) // lines_per_chunk
    offsets  = struct.unpack_from(f"<{n_chunks}Q", raw, pos)

    # ── 5. EXR channel layout ─────────────────────────────────────────────────
    # EXR stores channels ALPHABETICALLY within each scanline chunk.
    # Within a chunk the layout is CHANNEL-MAJOR (not pixel-interleaved):
    #   [ all pixels of channel A ] [ all pixels of channel B ] ...
    # Each channel's slice is width * n_lines * bytes_per_pixel bytes.
    sorted_channels = sorted(channels.keys())
    ch_index = sorted_channels.index(ch_name)

    # byte offset to the start of our channel within a decompressed chunk
    ch_slice_start = sum(
        width * lines_per_chunk * (2 if channels[c] == 1 else 4)
        for c in sorted_channels[:ch_index]
    )
    ch_slice_bytes_per_chunk = width * lines_per_chunk * bytes_per_pixel

    # ── 6. Decode chunks ──────────────────────────────────────────────────────
    result = np.zeros((height, width), dtype=np.float32)

    for chunk_idx, offset in enumerate(offsets):
        p     = offset
        _y    = struct.unpack_from("<i", raw, p)[0];  p += 4
        dsize = struct.unpack_from("<I", raw, p)[0];  p += 4
        chunk_data = raw[p: p + dsize]

        n_lines   = min(lines_per_chunk, height - chunk_idx * lines_per_chunk)
        row_start = chunk_idx * lines_per_chunk

        if compression == 0:   # NONE
            decoded = chunk_data
        else:                  # ZIP / ZIPS
            try:
                uncompressed = zlib.decompress(chunk_data)
            except zlib.error as e:
                raise EXRReadError(f"zlib decompress failed at chunk {chunk_idx}: {e}")
            decoded = _exr_zip_unpredict(uncompressed)

        # EXR channel-major layout: extract our channel's contiguous slice.
        # For the last chunk, n_lines < lines_per_chunk — adjust slice size.
        actual_ch_start = sum(
            width * n_lines * (2 if channels[c] == 1 else 4)
            for c in sorted_channels[:ch_index]
        )
        actual_ch_bytes = width * n_lines * bytes_per_pixel
        ch_slice = decoded[actual_ch_start: actual_ch_start + actual_ch_bytes]

        if pixel_type == 1:    # HALF → float32
            row_vals = np.frombuffer(ch_slice, dtype=np.float16).astype(np.float32)
        else:                   # FLOAT
            row_vals = np.frombuffer(ch_slice, dtype=np.float32)

        if len(row_vals) == n_lines * width:
            result[row_start: row_start + n_lines] = row_vals.reshape(n_lines, width)
        else:
            # Fallback: fill what we can
            flat = row_vals.flatten()
            take = min(len(flat), n_lines * width)
            result[row_start: row_start + n_lines] = flat[:take].reshape(n_lines, width) if take == n_lines * width else np.zeros((n_lines, width))

    # ── 7. Sanity check ───────────────────────────────────────────────────────
    finite = result[np.isfinite(result) & (result > 0)]
    if len(finite) > 0 and finite.max() > 1e15:
        raise EXRReadError(
            f"Decoded depth values are garbage (max={finite.max():.2e}). "
            f"The EXR may use an unsupported compression or channel layout. "
            f"Install OpenEXR:  pip install openexr"
        )

    return result


def _exr_zip_unpredict(data: bytes) -> bytes:
    """
    EXR ZIP/ZIPS byte-level predictor (applied AFTER zlib decompression).

    The spec says the entire decompressed buffer (all channels concatenated)
    is treated as one byte array for the predictor:
      1. Reorder: split into two halves (even-index bytes, odd-index bytes),
         then interleave them back.
      2. Delta decode.
    """
    n = len(data)
    b = bytearray(data)

    # Step 1: reorder (un-interleave even/odd bytes)
    tmp  = bytearray(n)
    half = (n + 1) // 2
    for i in range(half):
        tmp[i * 2] = b[i]
    for i in range(n - half):
        tmp[i * 2 + 1] = b[half + i]

    # Step 2: delta decode
    for i in range(1, n):
        tmp[i] = (tmp[i - 1] + tmp[i] - 128) & 0xFF

    return bytes(tmp)