import json, logging, struct, io, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

log = logging.getLogger(__name__)
_FLIP = np.diag([1., -1., -1., 1.])

def _pad4j(b): r=len(b)%4; return b+b'\x20'*(4-r) if r else b
def _pad4b(b): r=len(b)%4; return b+b'\x00'*(4-r) if r else b


# -- MESH CLEANING ------------------------------------------------------------
# FIX v2: Removed the multi-pass normal-deviation shard removal (_clean_mesh).
# That algorithm used cosine-threshold comparisons between adjacent face normals
# and removed any face whose normal disagreed with its neighbors beyond 45-60°.
# On a room scan this is destructive: every sharp corner (wall-floor junction,
# desk edge, door frame) has legitimate 90° normal discontinuities that the
# filter classified as "shards" and deleted, leaving holes.
#
# Replacement: _clean_mesh_safe only removes faces that are:
#   (a) degenerate (zero area), OR
#   (b) true geometric needles (aspect ratio > 20) AND genuinely tiny area
#       (< 0.5% of median face area).
# This kills real confetti/spikes without touching any surface geometry.

def _clean_mesh_safe(mesh):
    """
    Safe mesh cleaning: remove only zero-area degenerate faces and extreme
    needle spikes.  Does NOT use normal-deviation filtering (which creates holes
    at every sharp edge in a room scan).
    """
    import open3d as o3d

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(faces) == 0:
        return mesh

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    e_max = np.maximum(e0, np.maximum(e1, e2))
    e_min = np.minimum(e0, np.minimum(e1, e2))

    cross = np.cross(v1 - v0, v2 - v0)
    area  = np.linalg.norm(cross, axis=1) * 0.5

    med_area = np.median(area) + 1e-12

    # Remove face if:
    #   zero area (degenerate), OR
    #   extreme needle (aspect > 20) AND very small area (< 0.5% of median)
    bad_mask = (
        (area < 1e-10) |
        (
            ((e_max / (e_min + 1e-9)) > 20.0) &
            (area < med_area * 0.005)
        )
    )

    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        mesh.remove_triangles_by_mask(bad_mask)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        log.info(f"  Safe mesh clean: removed {n_bad:,} degenerate/needle faces")
    else:
        log.info("  Safe mesh clean: no bad faces found")

    return mesh


def _remove_tiny_mesh_clusters(mesh, min_ratio=0.002, min_faces=400):
    """
    Remove disconnected tiny triangle clusters (confetti shards) before UV bake.
    Keeps the largest component and any component above an adaptive threshold.
    """
    try:
        tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    except Exception:
        return mesh

    tri_clusters = np.asarray(tri_clusters)
    cluster_n_tri = np.asarray(cluster_n_tri)
    if cluster_n_tri.size == 0:
        return mesh

    largest = int(cluster_n_tri.max())
    keep_thr = max(int(largest * float(min_ratio)), int(min_faces))
    keep_ids = np.where(cluster_n_tri >= keep_thr)[0]
    if keep_ids.size == 0:
        keep_ids = np.array([int(np.argmax(cluster_n_tri))], dtype=np.int64)

    keep_mask = np.isin(tri_clusters, keep_ids)
    n_drop = int((~keep_mask).sum())
    if n_drop > 0:
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        log.info(
            f"  Confetti cleanup: removed {n_drop:,} faces from tiny components"
        )
    return mesh


def _load_metric_depth_for_bake(frame, target_wh=None):
    """
    Load frame depth in metric units for visibility/occlusion checks during texturing.
    """
    path = Path(frame.depth_path)
    try:
        if path.suffix.lower() == ".npy":
            depth = np.load(str(path)).astype(np.float32)
        else:
            try:
                import OpenEXR, Imath
                f = OpenEXR.InputFile(str(path))
                dw = f.header()["dataWindow"]
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                ch = next(iter(f.header()["channels"].keys()))
                raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
            except Exception:
                from src.modules.exr_reader import read_exr_depth
                depth = read_exr_depth(path).astype(np.float32)

        depth = depth * float(getattr(frame, "depth_scale", 1.0))
        depth[~np.isfinite(depth)] = 0.0

        if target_wh is not None:
            tw, th = int(target_wh[0]), int(target_wh[1])
            if tw > 0 and th > 0 and depth.shape != (th, tw):
                try:
                    import cv2
                    depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    from PIL import Image
                    depth = np.asarray(
                        Image.fromarray(depth, mode="F").resize((tw, th), Image.NEAREST),
                        dtype=np.float32,
                    )
        return depth.astype(np.float32)
    except Exception:
        return None


# -- PARALLEL XATLAS UV UNWRAP ------------------------------------------------

def _xatlas_unwrap_chunk(verts_local, faces_local, tex_size):
    """Unwrap one mesh chunk in a worker thread."""
    import xatlas
    chart_opts = xatlas.ChartOptions()
    chart_opts.max_iterations = 1
    pack_opts = xatlas.PackOptions()
    pack_opts.resolution   = tex_size
    pack_opts.padding      = 4
    pack_opts.bilinear     = True
    pack_opts.create_image = False
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts_local.astype(np.float32), faces_local.astype(np.uint32))
    atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
    vmapping, indices, uvs = atlas[0]
    return vmapping, indices.astype(np.uint32), uvs.astype(np.float32), \
           atlas.width, atlas.height


def _xatlas_unwrap(verts, faces, tex_size, n_threads=4):
    """
    Parallel xatlas: split into n_threads chunks, unwrap concurrently,
    pack into a grid in the final atlas.
    """
    try:
        import xatlas
    except ImportError:
        raise RuntimeError("xatlas not found.  pip install xatlas")

    F  = len(faces)
    t0 = time.time()
    log.info(f"  xatlas (parallel {n_threads} threads): {F:,} faces...")

    if n_threads <= 1 or F < 10_000:
        pack_opts = xatlas.PackOptions()
        pack_opts.resolution   = tex_size
        pack_opts.padding      = 4
        pack_opts.bilinear     = True
        pack_opts.create_image = False
        
        chart_opts = xatlas.ChartOptions()
        # FIX: Increase iterations for better chart partitioning
        chart_opts.max_iterations = 4 
        # FIX: Force stricter planar charts (reduces scrambling)
        chart_opts.straighten = True 
        
        atlas = xatlas.Atlas()
        atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32))
        atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
        vmapping, indices, uvs = atlas[0]
        verts_uv = verts[vmapping].astype(np.float32)
        log.info(f"  xatlas done in {time.time()-t0:.1f}s")
        return verts_uv, indices.astype(np.uint32), uvs.astype(np.float32), \
               atlas.width, atlas.height

    chunk_size = (F + n_threads - 1) // n_threads
    chunks = []
    for i in range(n_threads):
        s = i * chunk_size
        e = min(s + chunk_size, F)
        if s >= e:
            break
        sub_faces = faces[s:e]
        used_verts, inv = np.unique(sub_faces, return_inverse=True)
        local_faces = inv.reshape(-1, 3).astype(np.uint32)
        local_verts = verts[used_verts].astype(np.float32)
        chunks.append((used_verts, local_verts, local_faces))

    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        futs = {
            ex.submit(_xatlas_unwrap_chunk, lv, lf, tex_size): ci
            for ci, (_, lv, lf) in enumerate(chunks)
        }
        for fut in as_completed(futs):
            ci = futs[fut]
            vmapping, indices, uvs, aw, ah = fut.result()
            results[ci] = (chunks[ci][0][vmapping], indices, uvs, aw, ah)
            log.info(f"    chunk {ci+1}/{len(chunks)} done "
                     f"(atlas {aw}x{ah}, {len(indices):,} faces)")

    n_chunks = len(results)
    cols = int(np.ceil(np.sqrt(n_chunks)))
    rows = int(np.ceil(n_chunks / cols))

    all_verts, all_uvs, all_faces = [], [], []
    v_offset = 0
    for ci, (gvmap, indices, uvs, aw, ah) in enumerate(results):
        col = ci % cols
        row = ci // cols
        su = 1.0 / cols
        sv = 1.0 / rows
        active_u = aw / tex_size
        active_v = ah / tex_size
        uvs_out = uvs.copy()
        uvs_out[:, 0] = uvs[:, 0] * active_u * su + col * su
        uvs_out[:, 1] = uvs[:, 1] * active_v * sv + row * sv

        all_verts.append(verts[gvmap].astype(np.float32))
        all_uvs.append(uvs_out.astype(np.float32))
        all_faces.append(indices.astype(np.uint32) + v_offset)
        v_offset += len(gvmap)

    verts_uv  = np.concatenate(all_verts, axis=0)
    uvs_final = np.concatenate(all_uvs,   axis=0)
    faces_uv  = np.concatenate(all_faces, axis=0)

    log.info(
        f"  xatlas parallel done in {time.time()-t0:.1f}s | "
        f"merged atlas={tex_size}x{tex_size} | "
        f"UV verts={len(verts_uv):,}  faces={len(faces_uv):,}"
    )
    return verts_uv, faces_uv, uvs_final, tex_size, tex_size


# -- FRAME META PRELOAD -------------------------------------------------------

def _preload_frame_meta(frames):
    """Read (W, H) for every frame in parallel."""
    from PIL import Image as _P
    sizes = [(0, 0)] * len(frames)
    def _read(i):
        try:
            img = _P.open(frames[i].rgb_path)
            sizes[i] = (img.width, img.height)
            img.close()
        except:
            pass
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_read, range(len(frames))))
    return sizes


# -- GPU BAKE -----------------------------------------------------------------

def _gpu_bake(verts_uv, faces_uv, uvs_f, frames, atlas_w, atlas_h,
              frame_batch=32, fill_unseen=False, depth_gate_m=0.06):
    import torch
    from PIL import Image as _P
    torch.set_grad_enabled(False)
    dev = torch.device('cuda')

    F  = len(faces_uv)
    SW = atlas_w
    SH = atlas_h

    log.info(f"  Pre-reading {len(frames)} frame sizes...")
    t_meta = time.time()
    frame_sizes = _preload_frame_meta(frames)
    log.info(f"  Meta read in {time.time()-t_meta:.1f}s")

    scale  = torch.tensor([SW-1, SH-1], device=dev, dtype=torch.float32)
    uv0_px = torch.from_numpy(uvs_f[faces_uv[:, 0]]).to(dev) * scale
    uv1_px = torch.from_numpy(uvs_f[faces_uv[:, 1]]).to(dev) * scale
    uv2_px = torch.from_numpy(uvs_f[faces_uv[:, 2]]).to(dev) * scale

    v0g  = torch.from_numpy(verts_uv[faces_uv[:, 0]]).to(dev)
    v1g  = torch.from_numpy(verts_uv[faces_uv[:, 1]]).to(dev)
    v2g  = torch.from_numpy(verts_uv[faces_uv[:, 2]]).to(dev)
    fc   = (v0g + v1g + v2g) / 3
    fn   = torch.linalg.cross(v1g - v0g, v2g - v0g)
    fn   = fn / (fn.norm(dim=1, keepdim=True) + 1e-9)
    fc_h = torch.cat([fc, torch.ones(F, 1, device=dev)], dim=1)

    best_dist = torch.full((F,), float('inf'), device=dev)
    best_fi   = torch.full((F,), -1, dtype=torch.int32, device=dev)

    # ── pre-compute per-frame sharpness scores ──────────────────────────────
    # Laplacian variance: high = sharp frame, low = blurry/motion-blurred.
    # We normalise to [0.2, 1.0] so the worst frame still contributes a little.
    NF = len(frames)
    log.info("  Scoring frame sharpness...")
    sharp_scores = np.ones(NF, dtype=np.float32)
    def _laplacian_var(path):
        try:
            import cv2 as _cv
            img = _cv.imread(str(path), _cv.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            # Downsample for speed — sharpness estimate doesn't need full res
            h, w = img.shape
            if w > 640:
                img = _cv.resize(img, (640, int(h * 640 / w)))
            return float(_cv.Laplacian(img, _cv.CV_64F).var())
        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=8) as ex:
        vars_ = list(ex.map(lambda fr: _laplacian_var(fr.rgb_path), frames))
    vars_arr = np.array(vars_, dtype=np.float32)
    vmax = vars_arr.max()
    if vmax > 0:
        # Normalise: clamp minimum to 0.2 so even soft frames can contribute
        sharp_scores = np.clip(vars_arr / (vmax + 1e-9), 0.2, 1.0)
    sharp_t = torch.from_numpy(sharp_scores).to(dev)   # (NF,)
    log.info(f"  Sharpness range: min={sharp_scores.min():.3f}  "
             f"max={sharp_scores.max():.3f}  "
             f"mean={sharp_scores.mean():.3f}")

    t0 = time.time()
    log.info(f"  Building {NF} camera matrices...")

    w2c_arr    = np.zeros((NF, 4, 4), dtype=np.float32)
    K_arr      = np.zeros((NF, 3, 3), dtype=np.float32)
    c2w_arr    = np.zeros((NF, 4, 4), dtype=np.float32)
    valid_mask = np.zeros(NF, dtype=bool)

    for fi, fr in enumerate(frames):
        try:
            K              = fr.intrinsics.K().astype(np.float32)
            c2w            = (fr.c2w @ _FLIP).astype(np.float32)
            w2c            = np.linalg.inv(c2w)
            w2c_arr[fi]    = w2c
            K_arr[fi]      = K
            c2w_arr[fi]    = c2w
            valid_mask[fi] = True
        except:
            w2c_arr[fi] = np.eye(4, dtype=np.float32)
            K_arr[fi]   = np.eye(3, dtype=np.float32)
            c2w_arr[fi] = np.eye(4, dtype=np.float32)

    fc_h_b = fc_h.unsqueeze(0)
    fn_b   = fn.unsqueeze(0)
    fc_b   = fc.unsqueeze(0)

    for batch_start in range(0, NF, frame_batch):
        batch_end = min(batch_start + frame_batch, NF)
        B  = batch_end - batch_start
        bs = slice(batch_start, batch_end)

        w2c_t    = torch.from_numpy(w2c_arr[bs]).to(dev)
        cam_batch = fc_h_b.matmul(w2c_t.transpose(-1, -2))
        Zc        = cam_batch[..., 2]

        K_t = torch.from_numpy(K_arr[bs]).to(dev)
        fx  = K_t[:, 0, 0].unsqueeze(1)
        fy  = K_t[:, 1, 1].unsqueeze(1)
        cx  = K_t[:, 0, 2].unsqueeze(1)
        cy  = K_t[:, 1, 2].unsqueeze(1)
        uc  = fx * cam_batch[..., 0] / (Zc + 1e-9) + cx
        vc  = fy * cam_batch[..., 1] / (Zc + 1e-9) + cy

        c2w_t   = torch.from_numpy(c2w_arr[bs]).to(dev)
        cam_pos = c2w_t[:, :3, 3].unsqueeze(1)
        to_c    = cam_pos - fc_b
        dist    = to_c.norm(dim=2)
        face_vis = (fn_b * to_c).sum(dim=2) > 0.15

        for bi in range(B):
            fi = batch_start + bi
            if not valid_mask[fi]:
                continue
            IW, IH = frame_sizes[fi]
            if IW == 0:
                continue
            valid = (
                (Zc[bi]  > 0.05) &
                (uc[bi]  >= 0) & (uc[bi]  < IW) &
                (vc[bi]  >= 0) & (vc[bi]  < IH) &
                face_vis[bi]
            )
            # Composite score = sharpness × cos(view_angle) / distance
            # Higher score = better frame for this face.
            # cos_angle: dot(face_normal, direction_to_camera) — already in face_vis
            # We compute it explicitly here for weighting.
            to_cam_norm = to_c[bi] / (dist[bi].unsqueeze(1) + 1e-9)  # (F,3)
            cos_angle   = (fn * to_cam_norm).sum(dim=1).clamp(0.0, 1.0)  # (F,)
            # score: higher is better.  Use 1/dist so close frames score higher.
            # sharpness weight from pre-computed Laplacian variance.
            score = sharp_t[fi] * (cos_angle ** 2) / (dist[bi] + 1e-3)
            # Convert best_dist to best_score convention:
            # we store negative score in best_dist so "less than" still works.
            neg_score = -score
            improve = valid & (neg_score < best_dist)
            if improve.any():
                best_dist = torch.where(improve, neg_score, best_dist)
                best_fi   = torch.where(
                    improve,
                    torch.tensor(fi, device=dev, dtype=torch.int32),
                    best_fi,
                )

        if batch_start % (frame_batch * 10) == 0:
            cov = (best_fi >= 0).sum().item()
            log.info(f"    Frames {batch_end}/{NF}  covered={cov:,}/{F:,} "
                     f"({100*cov//F}%)")

    log.info(f"  Pass1 done: {time.time()-t0:.1f}s  "
             f"covered={(best_fi>=0).sum().item():,}/{F:,}")

    t1 = time.time()
    CHUNK = 5_000_000
    tex_t = torch.zeros(SH * SW, 3, dtype=torch.uint8, device=dev)
    nw    = 0

    gi   = torch.where(best_fi >= 0)[0]
    bba  = ((torch.ceil(torch.maximum(uv0_px[gi,0], torch.maximum(uv1_px[gi,0], uv2_px[gi,0])))
             - torch.floor(torch.minimum(uv0_px[gi,0], torch.minimum(uv1_px[gi,0], uv2_px[gi,0]))) + 1) *
            (torch.ceil(torch.maximum(uv0_px[gi,1], torch.maximum(uv1_px[gi,1], uv2_px[gi,1])))
             - torch.floor(torch.minimum(uv0_px[gi,1], torch.minimum(uv1_px[gi,1], uv2_px[gi,1]))) + 1)
           ).long()
    bw_  = (torch.ceil(torch.maximum(uv0_px[gi,0], torch.maximum(uv1_px[gi,0], uv2_px[gi,0])))
            - torch.floor(torch.minimum(uv0_px[gi,0], torch.minimum(uv1_px[gi,0], uv2_px[gi,0]))) + 1).long()
    xmn  = torch.floor(torch.minimum(uv0_px[gi,0], torch.minimum(uv1_px[gi,0], uv2_px[gi,0]))).long()
    ymn  = torch.floor(torch.minimum(uv0_px[gi,1], torch.minimum(uv1_px[gi,1], uv2_px[gi,1]))).long()

    ng      = len(gi)
    bba_cpu = bba.cpu().numpy()

    _img_cache  = {}
    _prefetch_q = ThreadPoolExecutor(max_workers=8)

    from PIL import Image as _P

    def _load_rgb(fi_c):
        fr = frames[int(fi_c)]
        try:
            rgb = np.asarray(_P.open(fr.rgb_path).convert("RGB"))
            return torch.from_numpy(rgb.copy())
        except:
            return None

    def _get_cam(fi_c):
        fi_c = int(fi_c)
        if fi_c not in _img_cache:
            K_t      = torch.from_numpy(K_arr[fi_c]).to(dev)
            w2c_t    = torch.from_numpy(w2c_arr[fi_c]).to(dev)
            rgb_cpu  = _load_rgb(fi_c)
            _img_cache[fi_c] = (K_t, w2c_t, rgb_cpu)
        return _img_cache[fi_c]

    pos = 0
    while pos < ng:
        cum = np.cumsum(bba_cpu[pos:])
        end = min(pos + int(np.searchsorted(cum, CHUNK, 'right')) + 1, ng)
        sl  = slice(pos, end)
        cgi = gi[sl]; cb = bba[sl]; cbw = bw_[sl]
        cx0 = xmn[sl]; cy0 = ymn[sl]
        nt  = int(cb.sum().item())
        if nt == 0:
            pos = end; continue

        rp    = cb
        cum_r = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                           cb.cumsum(0)[:-1]])
        loc   = torch.arange(nt, dtype=torch.int64, device=dev) \
                - torch.repeat_interleave(cum_r, rp)
        pw    = torch.repeat_interleave(cbw.to(torch.int64), rp)
        px_x  = (loc % pw + torch.repeat_interleave(cx0.to(torch.int64), rp)).to(torch.int32)
        px_y  = (loc // pw + torch.repeat_interleave(cy0.to(torch.int64), rp)).to(torch.int32)
        fit   = torch.repeat_interleave(cgi, rp)

        vp    = (px_x < SW) & (px_y < SH)
        px_x  = px_x[vp]; px_y = px_y[vp]; fit = fit[vp]

        a0 = uv0_px[fit]; a1 = uv1_px[fit]; a2 = uv2_px[fit]
        e1 = a1 - a0; e2 = a2 - a0
        pts = torch.stack([px_x.float()+0.5, px_y.float()+0.5], 1) - a0
        d00=(e1*e1).sum(1); d01=(e1*e2).sum(1); d11=(e2*e2).sum(1)
        den = d00*d11 - d01*d01 + 1e-10
        d20=(pts*e1).sum(1); d21=(pts*e2).sum(1)
        bv  = (d11*d20 - d01*d21) / den
        bw  = (d00*d21 - d01*d20) / den
        bu  = 1. - bv - bw
        ins = (bu >= -0.02) & (bv >= -0.02) & (bw >= -0.02)
        px_x=px_x[ins]; px_y=px_y[ins]; fit=fit[ins]
        bu=bu[ins]; bv=bv[ins]; bw=bw[ins]
        s = bu+bv+bw+1e-9; bu/=s; bv/=s; bw/=s

        wp0=v0g[fit]; wp1=v1g[fit]; wp2=v2g[fit]
        world = bu[:,None]*wp0 + bv[:,None]*wp1 + bw[:,None]*wp2

        bfp   = best_fi[fit]
        order = torch.argsort(bfp)
        bfp=bfp[order]; world=world[order]
        px_x=px_x[order]; px_y=px_y[order]

        uf_t, cnt_t = torch.unique(bfp, return_counts=True)
        uf  = uf_t.cpu().numpy()
        cnt = cnt_t.cpu().numpy()
        off = np.concatenate([[0], cnt.cumsum()])

        if end < ng:
            next_fis = torch.unique(
                best_fi[gi[end:min(end+500, ng)]]
            ).cpu().numpy()
            for nfi in next_fis:
                if nfi >= 0 and int(nfi) not in _img_cache:
                    _prefetch_q.submit(_load_rgb, int(nfi))

        for idx2, fi_c in enumerate(uf):
            if fi_c < 0: continue
            K_t, w2c_t, rgb_cpu = _get_cam(int(fi_c))
            if rgb_cpu is None: continue
            rgb_t = rgb_cpu.to(dev, non_blocking=True)
            IH, IW = rgb_t.shape[:2]
            sl2    = slice(int(off[idx2]), int(off[idx2+1]))
            wb     = world[sl2]
            ones   = torch.ones(len(wb), 1, device=dev)
            wbh    = torch.cat([wb, ones], 1)
            cc     = (w2c_t @ wbh.T).T
            Zc     = cc[:, 2]
            uc_px  = (K_t[0,0]*cc[:,0]/(Zc+1e-9)+K_t[0,2]).long().clamp(0,IW-1)
            vc_px  = (K_t[1,1]*cc[:,1]/(Zc+1e-9)+K_t[1,2]).long().clamp(0,IH-1)
            rgb_px = rgb_t[vc_px, uc_px]
            flat   = px_y[sl2].to(torch.int64)*SW + px_x[sl2].to(torch.int64)
            tex_t.index_put_((flat,), rgb_px)
            nw    += len(wb)
            del rgb_t

        pos = end
        log.info(f"    raster chunk {end}/{ng}  pixels={nw:,}")

    _prefetch_q.shutdown(wait=False)
    texture = tex_t.cpu().numpy().reshape(SH, SW, 3).astype(np.uint8)
    log.info(f"  GPU raster done: {time.time()-t1:.1f}s  ({nw:,} pixels)")

    import cv2
    empty = (texture.sum(2) == 0).astype(np.uint8) * 255
    ne = int((empty > 0).sum())
    if ne > 0:
        log.info(f"  Inpainting {ne:,} empty pixels...")
        t_d = time.time()
        texture = cv2.inpaint(texture, empty, inpaintRadius=4,
                              flags=cv2.INPAINT_TELEA)
        log.info(f"  Inpaint done in {time.time()-t_d:.1f}s")

    return texture


# -- CPU FALLBACK BAKE --------------------------------------------------------

def _cpu_bake(verts_uv, faces_uv, uvs_f, frames, atlas_w, atlas_h, fill_unseen=False, depth_gate_m=0.06):
    from PIL import Image as _P
    import cv2
    SW = atlas_w; SH = atlas_h; F = len(faces_uv)
    v0=verts_uv[faces_uv[:,0]]; v1=verts_uv[faces_uv[:,1]]; v2=verts_uv[faces_uv[:,2]]
    fc=((v0+v1+v2)/3).astype(np.float32)
    fn=np.cross(v1-v0,v2-v0).astype(np.float32)
    fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-9
    uv0=(uvs_f[faces_uv[:,0]]*np.array([SW-1,SH-1])).astype(np.float32)
    uv1=(uvs_f[faces_uv[:,1]]*np.array([SW-1,SH-1])).astype(np.float32)
    uv2=(uvs_f[faces_uv[:,2]]*np.array([SW-1,SH-1])).astype(np.float32)
    best_fi=np.full(F,-1,dtype=np.int32); best_sc=np.full(F,-np.inf)
    for fi,fr in enumerate(frames):
        K=fr.intrinsics.K().astype(np.float32); c2w=(fr.c2w@_FLIP).astype(np.float32); w2c=np.linalg.inv(c2w)
        to=c2w[:3,3]-fc; d=np.linalg.norm(to,axis=1)+1e-6
        dot=np.einsum("ij,ij->i",fn,to/d[:,None])
        ones=np.ones((F,1),np.float32); cam=(w2c@np.hstack([fc,ones]).T).T
        Z=cam[:,2]; u=K[0,0]*cam[:,0]/(Z+1e-9)+K[0,2]; v_=K[1,1]*cam[:,1]/(Z+1e-9)+K[1,2]
        try: img=_P.open(fr.rgb_path); IH,IW=img.height,img.width; img.close()
        except: continue
        sc=(dot**2)/(d*0.1+1); ok=(dot>0.15)&(Z>0.05)&(u>=0)&(u<IW)&(v_>=0)&(v_<IH)&(sc>best_sc)
        best_sc[ok]=sc[ok]; best_fi[ok]=fi
    texture=np.zeros((SH,SW,3),dtype=np.uint8)
    areas=(uv1[:,0]-uv0[:,0])*(uv2[:,1]-uv0[:,1])-(uv1[:,1]-uv0[:,1])*(uv2[:,0]-uv0[:,0])
    good=(np.abs(areas)>0.01)&(best_fi>=0); gi=np.where(good)[0]
    xmn=np.floor(np.minimum(uv0[gi,0],np.minimum(uv1[gi,0],uv2[gi,0]))).astype(np.int32).clip(0,SW-1)
    xmx=np.ceil(np.maximum(uv0[gi,0],np.maximum(uv1[gi,0],uv2[gi,0]))).astype(np.int32).clip(0,SW-1)
    ymn=np.floor(np.minimum(uv0[gi,1],np.minimum(uv1[gi,1],uv2[gi,1]))).astype(np.int32).clip(0,SH-1)
    ymx=np.ceil(np.maximum(uv0[gi,1],np.maximum(uv1[gi,1],uv2[gi,1]))).astype(np.int32).clip(0,SH-1)
    bba=((xmx-xmn+1)*(ymx-ymn+1)).astype(np.int64); bw2=(xmx-xmn+1)
    CHUNK=30_000_000; pos=0; ng=len(gi); nw=0; cam_cache={}
    def _cam(fi_c):
        if fi_c not in cam_cache:
            fr=frames[int(fi_c)]; K=fr.intrinsics.K().astype(np.float32)
            c2w=(fr.c2w@_FLIP).astype(np.float32); w2c=np.linalg.inv(c2w)
            try: rgb=np.asarray(_P.open(fr.rgb_path).convert("RGB"))
            except: rgb=None
            cam_cache[fi_c]=(K,w2c,rgb)
        return cam_cache[fi_c]
    while pos<ng:
        cum=np.cumsum(bba[pos:]); end=min(pos+int(np.searchsorted(cum,CHUNK,'right'))+1,ng)
        cgi=gi[pos:end]; cb=bba[pos:end]; cbw=bw2[pos:end]; cx0=xmn[pos:end]; cy0=ymn[pos:end]
        nt=int(cb.sum())
        if nt==0: pos=end; continue
        cum_r=np.concatenate([[0],cb.cumsum()[:-1]]); loc=np.arange(nt,dtype=np.int64)-np.repeat(cum_r,cb)
        pw=np.repeat(cbw,cb); px_x=(loc%pw+np.repeat(cx0,cb)).astype(np.int32); px_y=(loc//pw+np.repeat(cy0,cb)).astype(np.int32)
        fit=np.repeat(cgi,cb); vp=(px_x<SW)&(px_y<SH); px_x=px_x[vp]; px_y=px_y[vp]; fit=fit[vp]
        a0=uv0[fit]; a1=uv1[fit]; a2=uv2[fit]; e1=a1-a0; e2=a2-a0
        pts=np.stack([px_x.astype(np.float32)+0.5,px_y.astype(np.float32)+0.5],1)-a0
        d00=(e1*e1).sum(1); d01=(e1*e2).sum(1); d11=(e2*e2).sum(1); den=d00*d11-d01*d01+1e-10
        d20=(pts*e1).sum(1); d21=(pts*e2).sum(1)
        bv3=(d11*d20-d01*d21)/den; bw3=(d00*d21-d01*d20)/den; bu3=1.-bv3-bw3
        ins=(bu3>=-0.02)&(bv3>=-0.02)&(bw3>=-0.02); px_x=px_x[ins]; px_y=px_y[ins]; fit=fit[ins]
        bu3=bu3[ins]; bv3=bv3[ins]; bw3=bw3[ins]; s=bu3+bv3+bw3+1e-9; bu3/=s; bv3/=s; bw3/=s
        wp0=verts_uv[faces_uv[fit,0]]; wp1=verts_uv[faces_uv[fit,1]]; wp2=verts_uv[faces_uv[fit,2]]
        world=(bu3[:,None]*wp0+bv3[:,None]*wp1+bw3[:,None]*wp2).astype(np.float32)
        bfp=best_fi[fit]; order=np.argsort(bfp,kind='stable')
        bfp=bfp[order]; world=world[order]; px_x=px_x[order]; px_y=px_y[order]
        uf,cnt=np.unique(bfp,return_counts=True); off=np.concatenate([[0],cnt.cumsum()])
        for idx2,fi_c in enumerate(uf):
            if fi_c<0: continue
            K,w2c,rgb=_cam(int(fi_c))
            if rgb is None: continue
            IH,IW=rgb.shape[:2]; sl=slice(int(off[idx2]),int(off[idx2+1])); wb=world[sl]
            ones=np.ones((len(wb),1),np.float32); cc=(w2c@np.hstack([wb,ones]).T).T
            Zc=cc[:,2]; uc=K[0,0]*cc[:,0]/(Zc+1e-9)+K[0,2]; vc=K[1,1]*cc[:,1]/(Zc+1e-9)+K[1,2]
            texture[px_y[sl],px_x[sl]]=rgb[vc.astype(int).clip(0,IH-1),uc.astype(int).clip(0,IW-1)]; nw+=len(wb)
        pos=end
    empty=(texture.sum(2)==0).astype(np.uint8)*255
    if empty.any():
        texture=cv2.inpaint(texture,empty,inpaintRadius=4,flags=cv2.INPAINT_TELEA)
    return texture


# -- GLB EXPORT ---------------------------------------------------------------

def _export_glb(output_dir, verts, faces, uvs, texture_np):
    from PIL import Image
    atlas_h, atlas_w = texture_np.shape[:2]
    log.info(f"  Encoding PNG {atlas_w}x{atlas_h}...")
    buf=io.BytesIO(); Image.fromarray(texture_np).save(buf,"PNG")
    png=buf.getvalue(); log.info(f"  PNG: {len(png)//1024} KB")
    verts=verts.astype(np.float32); uvs=uvs.astype(np.float32)
    faces=faces[(faces<len(verts)).all(1)].astype(np.uint32)
    bp=_pad4b(verts.tobytes()); bu=_pad4b(uvs.tobytes())
    bi=_pad4b(faces.tobytes()); bpng=_pad4b(png)
    blob=bp+bu+bi+bpng; nv=len(verts); nf=len(faces)
    vmin=[float(x) for x in verts.min(0)]; vmax=[float(x) for x in verts.max(0)]
    gltf={
        "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},
                   "indices":2,"material":0,"mode":4}]}],
        "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0},
                      "metallicFactor":0.0,"roughnessFactor":1.0},"doubleSided":True}],
        "textures":[{"source":0,"sampler":0}],
        "images":[{"bufferView":3,"mimeType":"image/png"}],
        "samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":33071,"wrapT":33071}],
        "accessors":[
            {"bufferView":0,"byteOffset":0,"componentType":5126,
             "count":int(nv),"type":"VEC3","min":vmin,"max":vmax},
            {"bufferView":1,"byteOffset":0,"componentType":5126,
             "count":int(nv),"type":"VEC2"},
            {"bufferView":2,"byteOffset":0,"componentType":5125,
             "count":int(nf*3),"type":"SCALAR"}],
        "bufferViews":[
            {"buffer":0,"byteOffset":0,                            "byteLength":int(len(bp)),"target":34962},
            {"buffer":0,"byteOffset":int(len(bp)),                 "byteLength":int(len(bu)),"target":34962},
            {"buffer":0,"byteOffset":int(len(bp)+len(bu)),         "byteLength":int(len(bi)),"target":34963},
            {"buffer":0,"byteOffset":int(len(bp)+len(bu)+len(bi)), "byteLength":int(len(bpng))}],
        "buffers":[{"byteLength":int(len(blob))}],
    }
    jb=_pad4j(json.dumps(gltf,separators=(",",":")).encode())
    tot=12+8+len(jb)+8+len(blob)
    out=Path(output_dir)/"mesh_texture.glb"
    with open(out,"wb") as f:
        f.write(struct.pack("<III",0x46546C67,2,tot))
        f.write(struct.pack("<II",len(jb),0x4E4F534A)); f.write(jb)
        f.write(struct.pack("<II",len(blob),0x004E4942)); f.write(blob)
    log.info(f"  GLB -> {out}  ({out.stat().st_size/1024/1024:.1f} MB)")
    with open(out,"rb") as f:
        f.read(12); jl=struct.unpack("<I",f.read(4))[0]; f.read(4)
        p=json.loads(f.read(jl)); assert "textures" in p
    log.info("  GLB validation OK")
    try:
        import open3d as o3d
        m=o3d.geometry.TriangleMesh()
        m.vertices=o3d.utility.Vector3dVector(verts)
        m.triangles=o3d.utility.Vector3iVector(faces)
        m.compute_vertex_normals()
        ply=Path(output_dir)/"mesh_textured.ply"
        o3d.io.write_triangle_mesh(str(ply),m); log.info(f"  PLY -> {ply}")
        return ply
    except:
        return out


# -- MAIN CLASS ---------------------------------------------------------------

class TextureBaker:
    def __init__(self, output_dir, texture_size=4096,
                 xatlas_threads=4, frame_batch=64 , 
                 bake_faces=500_000):
        self.output_dir     = Path(output_dir)
        self.tex_size       = texture_size
        self.xatlas_threads = xatlas_threads
        self.frame_batch    = frame_batch
        self.bake_faces     = bake_faces

    def bake(self, mesh_input, frames, frame_skip=0):
        try:
            import open3d as o3d
        except ImportError:
            log.warning("open3d not available"); return mesh_input

        log.info(f"TextureBaker: Input type {type(mesh_input)}")
        
        # FIX: Handle if mesh_input is already a TriangleMesh object
        if isinstance(mesh_input, o3d.geometry.TriangleMesh):
            mesh = mesh_input
        else:
            # It is a path, try to read it
            mesh_path = str(mesh_input)
            if not Path(mesh_path).exists():
                 log.error(f"Mesh path does not exist: {mesh_path}")
                 return mesh_input
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            
        mesh.compute_vertex_normals()

        sel = [fr for i, fr in enumerate(frames) if i % (frame_skip+1) == 0]
        S   = self.tex_size
        # BAKE_FACES = 5_00_000
        log.info(f"  {len(sel)} frames | tex={S}x{S} | target={self.bake_faces :,} faces")

        # FIX: replaced aggressive normal-deviation _clean_mesh with safe version.
        # The old _clean_mesh removed faces at 45-60° normal differences, which
        # deleted every room corner, desk edge, and door frame, leaving holes.
        mesh = _clean_mesh_safe(mesh)

        # Simplify the clean mesh
        if len(mesh.triangles) > self.bake_faces :
            log.info(f"  Simplify {len(mesh.triangles):,} -> {self.bake_faces :,}...")
            t0 = time.time()
            mesh = mesh.simplify_quadric_decimation(self.bake_faces )
            mesh.compute_vertex_normals()
            log.info(f"  Simplified in {time.time()-t0:.1f}s")

        # Light post-simplify spike removal (safe: needle + tiny area only)
        verts_tmp = np.asarray(mesh.vertices, dtype=np.float32)
        faces_tmp = np.asarray(mesh.triangles, dtype=np.int32)
        if len(faces_tmp) > 0:
            v0t = verts_tmp[faces_tmp[:,0]]
            v1t = verts_tmp[faces_tmp[:,1]]
            v2t = verts_tmp[faces_tmp[:,2]]
            e0t = np.linalg.norm(v1t-v0t, axis=1)
            e1t = np.linalg.norm(v2t-v1t, axis=1)
            e2t = np.linalg.norm(v0t-v2t, axis=1)
            emx = np.maximum(e0t, np.maximum(e1t, e2t))
            emn = np.minimum(e0t, np.minimum(e1t, e2t))
            crt = np.cross(v1t-v0t, v2t-v0t)
            art = np.linalg.norm(crt, axis=1) * 0.5
            med_art = np.median(art) + 1e-12
            med_emx = np.median(emx)
            post_mask = (
                ((emx / (emn + 1e-9)) > 15.0) &
                (emx < med_emx * 2.0) &
                (art < med_art * 0.02)
            ) | (art < 1e-10)
            np_ = int(post_mask.sum())
            if np_ > 0:
                mesh.remove_triangles_by_mask(post_mask)
                mesh.remove_unreferenced_vertices()
                mesh.compute_vertex_normals()
                log.info(f"  Post-simplify spike clean: {np_:,} faces removed")

        verts = np.asarray(mesh.vertices,  dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)

        verts_uv, faces_uv, uvs_f, atlas_w, atlas_h = \
            _xatlas_unwrap(verts, faces, S, n_threads=self.xatlas_threads)

        gpu_ok = False
        try:
            import torch
            if torch.cuda.is_available():
                gpu_ok = True
                log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass

        t_bake = time.time()
        if gpu_ok:
            try:
                texture = _gpu_bake(verts_uv, faces_uv, uvs_f, sel,
                                    atlas_w, atlas_h,
                                    frame_batch=self.frame_batch)
                log.info(f"  GPU bake total: {time.time()-t_bake:.1f}s")
            except Exception as e:
                log.warning(f"  GPU failed ({e}) -> CPU fallback")
                gpu_ok = False
                try:
                    import torch as _t; _t.cuda.empty_cache()
                except: pass

        if not gpu_ok:
            texture = _cpu_bake(verts_uv, faces_uv, uvs_f, sel, atlas_w, atlas_h)
            log.info(f"  CPU bake total: {time.time()-t_bake:.1f}s")

        return _export_glb(self.output_dir, verts_uv, faces_uv, uvs_f, texture)
