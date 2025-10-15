#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ishigaki Bulge MVP v1.1 (with debug & auto-recovery)
- Adds --debug to save intermediate images (edges, band overlay, chosen contour).
- Auto-widens the vertical band if too few samples.
- Optional fallback to GrabCut seeded by the baseline band (--use-grabcut).
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
    ap.add_argument("--band-px", type=int, default=80)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--smooth-win", type=int, default=21)
    ap.add_argument("--smooth-poly", type=int, default=3)
    ap.add_argument("--gray-eq", action="store_true")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--debug", action="store_true", help="Save edges/contours/band overlays")
    ap.add_argument("--use-grabcut", action="store_true", help="Try GrabCut fallback if contours are weak")
    return ap.parse_args()

def ginput_two_points(img_disp):
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.title("Click TOP, then BOTTOM. Close when done.")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        raise RuntimeError("Two points not selected.")
    P_top = np.array(pts[0], dtype=np.float32)
    P_bot = np.array(pts[1], dtype=np.float32)
    return P_top, P_bot

def rotate_about_point(image, angle_deg, center):
    M = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated, M

def apply_transform_points(pts, M):
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    out = (M @ pts_h.T).T
    return out

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def canny_edges(img_bgr, c1, c2, dilate_it, erode_it, use_clahe=False):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = clahe_gray(gray)
    edges = cv2.Canny(gray, c1, c2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if dilate_it>0:
        edges = cv2.dilate(edges, kernel, iterations=dilate_it)
    if erode_it>0:
        edges = cv2.erode(edges, kernel, iterations=erode_it)
    return edges

def choose_contour_in_band(edges, band_center_x, band_half_width_px):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    H, W = edges.shape[:2]
    x0 = int(max(0, band_center_x - band_half_width_px))
    x1 = int(min(W-1, band_center_x + band_half_width_px))

    best = None
    best_score = -1.0
    for c in cnts:
        c = c.reshape(-1,2)
        xs = c[:,0]; ys = c[:,1]
        in_band = ((xs >= x0) & (xs <= x1))
        band_ratio = in_band.mean()
        vspan = ys.max() - ys.min()
        score = band_ratio * (1.0 + vspan / H)
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        return None
    return best.reshape(-1,1,2).astype(np.int32)

def grabcut_mask(img_bgr, band_center_x, band_half_width_px):
    H, W = img_bgr.shape[:2]
    x0 = int(max(0, band_center_x - band_half_width_px))
    x1 = int(min(W-1, band_center_x + band_half_width_px))
    rect = (max(0,x0-20), 10, min(W-1, x1 - x0 + 40), H-20)  # expand a bit
    mask = np.zeros((H,W), np.uint8)
    bgdModel = np.zeros((1,65), np.float64); fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, mask2
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1,1,2).astype(np.int32), mask2

def profile_along_baseline(contour_pts, P_top, P_bot, bins=200, smooth_win=21, smooth_poly=3):
    v = (P_bot - P_top).astype(np.float32)
    L = np.linalg.norm(v)
    if L < 1e-3: return None, None, None, None
    eL = v / L
    eN = np.array([-eL[1], eL[0]], dtype=np.float32)
    w = contour_pts.reshape(-1,2).astype(np.float32) - P_top[None,:]
    t = (w @ eL) / L
    d = (w @ eN)
    mask = (t >= 0) & (t <= 1)
    t = t[mask]; d = d[mask]
    if len(t) < 10:
        return None, None, eL, eN
    edges = np.linspace(0, 1, bins+1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    idx = np.digitize(t, edges) - 1
    prof = np.full(bins, np.nan, dtype=np.float32)
    for i in range(bins):
        vals = d[idx == i]
        if len(vals) > 0:
            prof[i] = np.median(vals)
    valid = ~np.isnan(prof)
    prof_smooth = prof.copy()
    if valid.sum() > max(smooth_win, smooth_poly+2):
        inds = np.where(valid)[0]
        for i in range(len(prof)):
            if np.isnan(prof[i]):
                j = inds[np.argmin(np.abs(inds - i))]
                prof[i] = prof[j]
        if smooth_win % 2 == 0: smooth_win += 1
        prof_smooth = savgol_filter(prof, smooth_win, smooth_poly, mode="interp")
    return centers, prof_smooth, eL, eN

def visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm=None, out_path=None, debug=False, edges=None, contour=None, band=None):
    canvas = img_rot.copy()
    pt1 = tuple(np.round(P_top_rot).astype(int))
    pt2 = tuple(np.round(P_bot_rot).astype(int))
    cv2.line(canvas, pt1, pt2, (0,255,0), 2)
    v = (P_bot_rot - P_top_rot).astype(np.float32)
    eL = v / np.linalg.norm(v); eN = np.array([-eL[1], eL[0]], dtype=np.float32)
    if centers is not None and prof_px is not None:
        for t, d in zip(centers, prof_px):
            if not np.isfinite(d): continue
            p = P_top_rot + eL * (t * np.linalg.norm(v))
            q = p + eN * d
            cv2.circle(canvas, tuple(np.round(p).astype(int)), 1, (255,0,0), -1)
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (255,0,0), 1)
    if debug and band is not None:
        x0,x1 = band; cv2.rectangle(canvas, (x0,0), (x1,canvas.shape[0]-1), (0,0,255), 1)
    if debug and contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0,165,255), 2)
    fig1 = plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title("Overlay (baseline green / offsets blue / band red / contour orange)")
    fig2 = plt.figure(figsize=(7,4))
    if centers is not None and prof_px is not None:
        if scale_px_per_mm:
            prof_mm = prof_px / scale_px_per_mm
            plt.plot(centers, prof_mm); plt.ylabel("Lateral offset [mm]")
            plt.title(f"Bulge profile (max |offset| ≈ {np.nanmax(np.abs(prof_mm)):.1f} mm)")
        else:
            plt.plot(centers, prof_px); plt.ylabel("Lateral offset [px]")
            plt.title(f"Bulge profile (max |offset| ≈ {np.nanmax(np.abs(prof_px)):.1f} px)")
        plt.xlabel("Normalized height (0=top, 1=bottom)"); plt.grid(True)
    else:
        plt.text(0.5,0.5,"No profile (see debug)", ha='center', va='center'); plt.axis('off')
    if out_path:
        fig1.savefig(os.path.join(out_path, "overlay.png"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_path, "profile.png"), dpi=200, bbox_inches="tight")
        if debug and edges is not None: cv2.imwrite(os.path.join(out_path, "edges.png"), edges)
        if debug and contour is not None:
            dbg = img_rot.copy(); cv2.drawContours(dbg, [contour], -1, (0,255,255), 2)
            cv2.imwrite(os.path.join(out_path, "contour.png"), dbg)
    return fig1, fig2

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None: raise FileNotFoundError(f"Failed to read image: {args.image}")
    print("[INFO] Select TOP then BOTTOM on the displayed image...")
    P_top, P_bot = ginput_two_points(img)
    v = (P_bot - P_top).astype(np.float32)
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    img_rot, M = rotate_about_point(img, angle_deg, P_top)
    P_top_rot = apply_transform_points(P_top[None,:], M)[0]
    P_bot_rot = apply_transform_points(P_bot[None,:], M)[0]
    band_half = max(20, int(args.band_px)); tried = []; contour = None; centers=None; prof_px=None
    for factor in [1, 1.5, 2, 3]:
        band_half_try = int(band_half * factor)
        edges = canny_edges(img_rot, args.canny1, args.canny2, args.dilate, args.erode, args.gray_eq)
        band_center_x = int(P_top_rot[0])
        contour_try = choose_contour_in_band(edges, band_center_x, band_half_try)
        tried.append((band_half_try, contour_try is not None))
        if contour_try is None: continue
        centers_try, prof_px_try, eL, eN = profile_along_baseline(contour_try, P_top_rot, P_bot_rot,
                                                                  bins=args.bins, smooth_win=args.smooth_win, smooth_poly=args.smooth_poly)
        if centers_try is not None:
            contour, centers, prof_px = contour_try, centers_try, prof_px_try
            used_band = (max(0, band_center_x - band_half_try), min(img_rot.shape[1]-1, band_center_x + band_half_try))
            break
    mask_gc=None
    if centers is None and args.use_grabcut:
        contour_gc, mask_gc = grabcut_mask(img_rot, int(P_top_rot[0]), int(band_half*2))
        if contour_gc is not None:
            centers, prof_px, eL, eN = profile_along_baseline(contour_gc, P_top_rot, P_bot_rot,
                                                              bins=args.bins, smooth_win=args.smooth_win, smooth_poly=args.smooth_poly)
            if centers is not None:
                contour = contour_gc
                used_band = (max(0, int(P_top_rot[0]-band_half*2)), min(img_rot.shape[1]-1, int(P_top_rot[0]+band_half*2)))
    scale_px_per_mm=None
    if args.height_mm and args.height_mm>0:
        px_len = np.linalg.norm(P_bot_rot - P_top_rot)
        scale_px_per_mm = px_len / float(args.height_mm)
    if centers is None:
        edges = canny_edges(img_rot, args.canny1, args.canny2, args.dilate, args.erode, args.gray_eq)
        used_band = (max(0, int(P_top_rot[0]-band_half)), min(img_rot.shape[1]-1, int(P_top_rot[0]+band_half)))
        visualize(img_rot, P_top_rot, P_bot_rot, None, None, scale_px_per_mm, out_path=args.output_dir,
                  debug=True, edges=edges, contour=None, band=used_band)
        if args.debug and mask_gc is not None:
            cv2.imwrite(os.path.join(args.output_dir, "grabcut_mask.png"), mask_gc)
        tries = ", ".join([f"{h}px:{'ok' if ok else 'ng'}" for h,ok in tried])
        raise RuntimeError("Failed to get enough contour samples. Saved debug images (edges.png, overlay.png). "
                           f"Try: increase --band-px / adjust --canny1 --canny2 / add --gray-eq / use --use-grabcut. "
                           f"Attempts: {tries}")
    else:
        visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm, out_path=args.output_dir,
                  debug=args.debug, edges=None, contour=contour, band=used_band)
        if scale_px_per_mm:
            print(f"[RESULT] Max |bulge| ≈ {np.nanmax(np.abs(prof_px/scale_px_per_mm)):.1f} mm")
        else:
            print(f"[RESULT] Max |bulge| ≈ {np.nanmax(np.abs(prof_px)):.1f} px")

if __name__ == "__main__":
    main()
