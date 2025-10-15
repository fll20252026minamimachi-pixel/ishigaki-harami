#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ishigaki Bulge v1.2 — Better wall detection
(Polygon ROI + HSV masking + GrabCut-init)
"""
import argparse, os, numpy as np, cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--height-mm", type=float, default=None)
    ap.add_argument("--canny1", type=int, default=80)
    ap.add_argument("--canny2", type=int, default=160)
    ap.add_argument("--band-px", type=int, default=100)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--smooth-win", type=int, default=21)
    ap.add_argument("--smooth-poly", type=int, default=3)
    ap.add_argument("--gray-eq", action="store_true")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--min-area", type=float, default=1000.0)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--use-grabcut", action="store_true")
    ap.add_argument("--roi", action="store_true", help="Draw polygon ROI; Enter to finish")
    ap.add_argument("--mask-veg", action="store_true")
    ap.add_argument("--mask-sky", action="store_true")
    ap.add_argument("--veg-h", nargs=2, type=int, default=[35, 85])
    ap.add_argument("--sky-h", nargs=2, type=int, default=[90, 130])
    ap.add_argument("--sat-th", type=int, default=40)
    return ap.parse_args()

def ginput_points(img, title, min_pts=2):
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title + "\nLeft-click to add points, Enter to finish")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close()
    if len(pts) < min_pts:
        raise RuntimeError(f"Need at least {min_pts} points.")
    return np.array(pts, dtype=np.float32)

def rotate_about_point(image, angle_deg, center):
    M = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return rotated, M

def apply_transform_points(pts, M):
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    out = (M @ pts_h.T).T
    return out

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def hsv_suppressions(img_bgr, mask=None, veg_h=(35,85), sky_h=(90,130), sat_th=40, use_veg=False, use_sky=False):
    H,W = img_bgr.shape[:2]
    keep = np.ones((H,W), np.uint8)*255 if mask is None else (mask>0).astype(np.uint8)*255
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hch = hsv[:,:,0]; Sch = hsv[:,:,1]
    if use_veg:
        v1,v2 = veg_h
        veg = (Hch>=v1) & (Hch<=v2) & (Sch>=sat_th)
        keep[veg] = 0
    if use_sky:
        s1,s2 = sky_h
        sky = (Hch>=s1) & (Hch<=s2) & (Sch>=sat_th)
        keep[sky] = 0
    return keep

def canny_edges(img_bgr, c1, c2, dilate_it, erode_it, use_clahe=False, keep_mask=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = clahe_gray(gray)
    if keep_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=keep_mask)
    edges = cv2.Canny(gray, c1, c2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if dilate_it>0: edges = cv2.dilate(edges, kernel, iterations=dilate_it)
    if erode_it>0:  edges = cv2.erode(edges, kernel, iterations=erode_it)
    return edges

def choose_contour_in_band(edges, band_center_x, band_half_width_px, min_area=1000):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    H,W = edges.shape[:2]
    x0 = int(max(0, band_center_x - band_half_width_px))
    x1 = int(min(W-1, band_center_x + band_half_width_px))
    best=None; best_score=-1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        pts = c.reshape(-1,2)
        xs = pts[:,0]; ys = pts[:,1]
        in_band = ((xs>=x0)&(xs<=x1))
        band_ratio = in_band.mean()
        vspan = ys.max() - ys.min()
        score = band_ratio * (1.0 + vspan / max(1,H))
        if score > best_score:
            best_score=score; best=c
    return best

def mask_from_polygon(img_shape, poly_pts):
    mask = np.zeros(img_shape[:2], np.uint8)
    poly_i = np.round(poly_pts).astype(np.int32).reshape(-1,1,2)
    cv2.fillPoly(mask, [poly_i], 255)
    return mask

def grabcut_with_mask(img_bgr, init_keep_mask):
    H,W = img_bgr.shape[:2]
    mask = np.zeros((H,W), np.uint8) + cv2.GC_PR_BGD
    mask[init_keep_mask>0] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    final = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, final
    c = max(cnts, key=cv2.contourArea)
    return c, final

def profile_along_baseline(contour_pts, P_top, P_bot, bins=200, smooth_win=21, smooth_poly=3):
    v = (P_bot - P_top).astype(np.float32)
    L = np.linalg.norm(v)
    if L < 1e-3: return None, None, None, None
    eL = v / L; eN = np.array([-eL[1], eL[0]], dtype=np.float32)
    w = contour_pts.reshape(-1,2).astype(np.float32) - P_top[None,:]
    t = (w @ eL) / L; d = (w @ eN)
    mask = (t >= 0) & (t <= 1)
    t = t[mask]; d = d[mask]
    if len(t) < 10: return None, None, eL, eN
    edges = np.linspace(0, 1, bins+1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    idx = np.digitize(t, edges) - 1
    prof = np.full(bins, np.nan, dtype=np.float32)
    for i in range(bins):
        vals = d[idx == i]
        if len(vals) > 0: prof[i] = np.median(vals)
    valid = ~np.isnan(prof); prof_smooth = prof.copy()
    if valid.sum() > max(smooth_win, smooth_poly+2):
        inds = np.where(valid)[0]
        for i in range(len(prof)):
            if np.isnan(prof[i]): prof[i] = prof[inds[np.argmin(np.abs(inds - i))]]
        if smooth_win % 2 == 0: smooth_win += 1
        prof_smooth = savgol_filter(prof, smooth_win, smooth_poly, mode="interp")
    return centers, prof_smooth, eL, eN

def visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm=None, out_path=None, debug=False, edges=None, contour=None, band=None):
    canvas = img_rot.copy()
    pt1 = tuple(np.round(P_top_rot).astype(int)); pt2 = tuple(np.round(P_bot_rot).astype(int))
    cv2.line(canvas, pt1, pt2, (0,255,0), 2)
    v = (P_bot_rot - P_top_rot).astype(np.float32)
    eL = v / np.linalg.norm(v); eN = np.array([-eL[1], eL[0]], dtype=np.float32)
    if centers is not None and prof_px is not None:
        for t, d in zip(centers, prof_px):
            if not np.isfinite(d): continue
            p = P_top_rot + eL * (t * np.linalg.norm(v)); q = p + eN * d
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (255,0,0), 1)
    if debug and band is not None:
        x0,x1 = band; cv2.rectangle(canvas, (x0,0), (x1,canvas.shape[0]-1), (0,0,255), 1)
    if debug and contour is not None: cv2.drawContours(canvas, [contour], -1, (0,165,255), 2)
    fig1 = plt.figure(figsize=(8,6)); plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title("Overlay (baseline green / offsets blue / band red / contour orange)")
    fig2 = plt.figure(figsize=(7,4))
    if centers is not None and prof_px is not None:
        if scale_px_per_mm:
            prof_mm = prof_px / scale_px_per_mm; plt.plot(centers, prof_mm); plt.ylabel("Lateral offset [mm]")
            ttl=f"Bulge profile (max |offset| ≈ {np.nanmax(np.abs(prof_mm)):.1f} mm)"
        else:
            plt.plot(centers, prof_px); plt.ylabel("Lateral offset [px]")
            ttl=f"Bulge profile (max |offset| ≈ {np.nanmax(np.abs(prof_px)):.1f} px)"
        plt.xlabel("Normalized height (0=top, 1=bottom)"); plt.grid(True); plt.title(ttl)
    else:
        plt.text(0.5,0.5,"No profile (see debug)", ha='center', va='center'); plt.axis('off')
    if out_path:
        fig1.savefig(os.path.join(out_path, "overlay.png"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_path, "profile.png"), dpi=200, bbox_inches="tight")
        if debug and edges is not None: cv2.imwrite(os.path.join(out_path, "edges.png"), edges)
        if debug and contour is not None:
            dbg = img_rot.copy(); cv2.drawContours(dbg, [contour], -1, (0,255,255), 2); cv2.imwrite(os.path.join(out_path, "contour.png"), dbg)
    return fig1, fig2

def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    img = cv2.imread(args.image); 
    if img is None: raise FileNotFoundError(f"Failed to read image: {args.image}")
    # baseline
    print("[INFO] Select TOP then BOTTOM..."); pts_tb = ginput_points(img, "Click TOP then BOTTOM (2 points)", min_pts=2)
    P_top, P_bot = pts_tb[0], pts_tb[1]
    # ROI polygon optional
    roi_mask=None
    if args.roi:
        print("[INFO] Draw polygon ROI (Enter to finish)...")
        poly = ginput_points(img, "Draw polygon ROI (Left-click; Enter to finish)", min_pts=3)
        roi_mask = mask_from_polygon(img.shape, poly)
    # rotate
    v = (P_bot - P_top).astype(np.float32); angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    img_rot, M = rotate_about_point(img, angle_deg, P_top)
    P_top_rot = apply_transform_points(P_top[None,:], M)[0]; P_bot_rot = apply_transform_points(P_bot[None,:], M)[0]
    if roi_mask is not None: roi_mask = cv2.warpAffine(roi_mask, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    # color suppressions
    keep_mask=None
    if args.mask_veg or args.mask_sky or roi_mask is not None:
        base = roi_mask if roi_mask is not None else np.ones(img_rot.shape[:2], np.uint8)*255
        keep_mask = hsv_suppressions(img_rot, mask=base, veg_h=tuple(args.veg_h), sky_h=tuple(args.sky_h),
                                     sat_th=args.sat_th, use_veg=args.mask_veg, use_sky=args.mask_sky)
    # edges & contour
    band_center_x=int(P_top_rot[0])
    edges = canny_edges(img_rot, args.canny1, args.canny2, args.dilate, args.erode, args.gray_eq, keep_mask=keep_mask)
    contour = choose_contour_in_band(edges, band_center_x, args.band_px, min_area=args.min_area)
    # grabcut fallback
    if (contour is None) and args.use_grabcut:
        init_mask = keep_mask if keep_mask is not None else np.ones(img_rot.shape[:2], np.uint8)*255
        contour, gc_mask = grabcut_with_mask(img_rot, init_mask)
        if args.debug: cv2.imwrite(os.path.join(args.output_dir, "grabcut_mask.png"), gc_mask)
    # profile & scale
    centers=None; prof_px=None; scale_px_per_mm=None
    if contour is not None:
        centers, prof_px, eL, eN = profile_along_baseline(contour, P_top_rot, P_bot_rot, bins=args.bins,
                                                          smooth_win=args.smooth_win, smooth_poly=args.smooth_poly)
    if args.height_mm and args.height_mm>0:
        px_len = np.linalg.norm(P_bot_rot - P_top_rot); scale_px_per_mm = px_len / float(args.height_mm)
    # visualize
    used_band = (max(0, band_center_x - args.band_px), min(img_rot.shape[1]-1, band_center_x + args.band_px))
    visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm, out_path=args.output_dir,
              debug=args.debug, edges=edges, contour=contour, band=used_band)
    # csv
    if centers is not None and prof_px is not None:
        csv_path = os.path.join(args.output_dir, "bulge_profile.csv")
        if scale_px_per_mm:
            prof_mm = prof_px / scale_px_per_mm; data = np.vstack([centers, prof_px, prof_mm]).T; header="t_norm,offset_px,offset_mm"
        else:
            data = np.vstack([centers, prof_px]).T; header="t_norm,offset_px"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        if scale_px_per_mm:
            print(f"[RESULT] Max |bulge| ≈ {np.nanmax(np.abs(prof_px/scale_px_per_mm)):.1f} mm")
        else:
            print(f"[RESULT] Max |bulge| ≈ {np.nanmax(np.abs(prof_px)):.1f} px")
    else:
        raise RuntimeError("Failed to get a reliable contour. Try: --roi, --mask-veg, --mask-sky, adjust --band-px / --canny1/2, or add --use-grabcut.")
if __name__ == "__main__":
    main()
