#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ishigaki Bulge MVP (Python/OpenCV)
----------------------------------
Minimal prototype to measure lateral "bulge" of a stone wall relative to a
user-defined baseline (top->bottom).

Features:
- Click top and bottom points on the image to define the baseline.
- Optional: enter real-world height (mm) to convert px -> mm.
- Automatic boundary extraction via Canny + morphology + contour filtering.
- Computes signed lateral offset profile along the baseline.
- Saves CSV and visualization overlay.
- Displays interactive figures (matplotlib).

Usage:
    python ishigaki_bulge_mvp.py --image path/to/image.jpg --height-mm 2000

Dependencies:
    pip install opencv-python numpy matplotlib scipy

Notes:
    - For best results, shoot as fronto-parallel as possible.
    - If extraction fails, adjust thresholds via CLI or pre-mask the wall.
"""

import argparse
import os
import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--height-mm", type=float, default=None, help="Real height between selected top/bottom points (mm) for px->mm scaling")
    ap.add_argument("--canny1", type=int, default=80, help="Canny lower threshold")
    ap.add_argument("--canny2", type=int, default=160, help="Canny upper threshold")
    ap.add_argument("--band-px", type=int, default=80, help="Half width of vertical band around baseline for contour filtering (px)")
    ap.add_argument("--bins", type=int, default=200, help="Number of vertical samples along baseline")
    ap.add_argument("--smooth-win", type=int, default=21, help="Savitzky-Golay window length (odd)")
    ap.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay polyorder")
    ap.add_argument("--gray-eq", action="store_true", help="Apply histogram equalization (CLAHE) before edges")
    ap.add_argument("--dilate", type=int, default=2, help="Morphological dilation iterations on edges")
    ap.add_argument("--erode", type=int, default=1, help="Morphological erosion iterations on edges")
    ap.add_argument("--output-dir", default=".", help="Directory to save results")
    return ap.parse_args()

def ginput_two_points(img_disp):
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.title("Click TOP point, then BOTTOM point. Close the plot if needed.")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        raise RuntimeError("Two points not selected.")
    # matplotlib returns (x,y); convert to numpy array [x,y]
    P_top = np.array([pts[0][0], pts[0][1]], dtype=np.float32)
    P_bot = np.array([pts[1][0], pts[1][1]], dtype=np.float32)
    return P_top, P_bot

def rotate_about_point(image, angle_deg, center):
    M = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated, M

def apply_transform_points(pts, M):
    # pts: (N,2)
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    out = (M @ pts_h.T).T
    return out

def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_wall_contour(img, band_center_x, band_half_width_px, canny1, canny2, dilate_it, erode_it, use_clahe=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = clahe_gray(gray)
    edges = cv2.Canny(gray, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=max(0,dilate_it))
    edges = cv2.erode(edges, kernel, iterations=max(0,erode_it))

    # find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None, edges

    H, W = edges.shape[:2]
    x0 = int(max(0, band_center_x - band_half_width_px))
    x1 = int(min(W-1, band_center_x + band_half_width_px))

    best = None
    best_score = -1.0
    for c in cnts:
        c = c.reshape(-1,2)
        # score contours that significantly intersect the vertical band and span vertically
        xs = c[:,0]
        ys = c[:,1]
        in_band = ((xs >= x0) & (xs <= x1))
        band_ratio = in_band.mean()
        vspan = ys.max() - ys.min()
        score = band_ratio * (1.0 + vspan / H)
        if score > best_score:
            best_score = score
            best = c

    return best.reshape(-1,1,2).astype(np.int32), edges

def profile_along_baseline(contour_pts, P_top, P_bot, bins=200, smooth_win=21, smooth_poly=3):
    v = (P_bot - P_top).astype(np.float32)
    L = np.linalg.norm(v)
    if L < 1e-3:
        raise ValueError("Selected points too close.")
    eL = v / L
    eN = np.array([-eL[1], eL[0]], dtype=np.float32)  # left normal

    w = contour_pts.reshape(-1,2).astype(np.float32) - P_top[None,:]
    t = (w @ eL) / L  # normalized [0,1] along baseline
    d = (w @ eN)      # signed lateral offset (px)

    # keep only points whose projection falls on the segment [0,1]
    mask = (t >= 0) & (t <= 1)
    t = t[mask]; d = d[mask]

    if len(t) < 10:
        raise RuntimeError("Not enough contour samples near the baseline. Try adjusting thresholds or band width.")

    # binning by t
    edges = np.linspace(0, 1, bins+1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    idx = np.digitize(t, edges) - 1
    prof = np.full(bins, np.nan, dtype=np.float32)
    for i in range(bins):
        vals = d[idx == i]
        if len(vals) > 0:
            prof[i] = np.median(vals)  # robust representative

    valid = ~np.isnan(prof)
    prof_smooth = prof.copy()
    if valid.sum() > max(smooth_win, smooth_poly+2):
        # fill gaps by nearest valid then smooth
        # simple gap fill:
        inds = np.where(valid)[0]
        for i in range(len(prof)):
            if np.isnan(prof[i]):
                j = inds[np.argmin(np.abs(inds - i))]
                prof[i] = prof[j]
        # ensure odd window
        if smooth_win % 2 == 0:
            smooth_win += 1
        prof_smooth = savgol_filter(prof, smooth_win, smooth_poly, mode="interp")
    return centers, prof_smooth, eL, eN

def visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm=None, out_path=None):
    # overlay: baseline and small horizontal ticks indicating offset
    canvas = img_rot.copy()
    pt1 = tuple(np.round(P_top_rot).astype(int))
    pt2 = tuple(np.round(P_bot_rot).astype(int))
    cv2.line(canvas, pt1, pt2, (0,255,0), 2)  # baseline

    v = (P_bot_rot - P_top_rot).astype(np.float32)
    eL = v / np.linalg.norm(v)
    eN = np.array([-eL[1], eL[0]], dtype=np.float32)

    for t, d in zip(centers, prof_px):
        if not np.isfinite(d):
            continue
        p = P_top_rot + eL * (t * np.linalg.norm(v))
        q = p + eN * d
        p_i = tuple(np.round(p).astype(int))
        q_i = tuple(np.round(q).astype(int))
        cv2.circle(canvas, p_i, 1, (255,0,0), -1)
        cv2.line(canvas, p_i, q_i, (255,0,0), 1)

    # plots
    fig1 = plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title("Overlay: baseline (green) & lateral offsets (blue)")
    plt.axis('off')

    fig2 = plt.figure(figsize=(7,4))
    y = centers  # 0..1
    if scale_px_per_mm:
        prof_mm = prof_px / scale_px_per_mm
        plt.plot(y, prof_mm)
        plt.xlabel("Normalized height (0=top, 1=bottom)")
        plt.ylabel("Lateral offset [mm]")
        max_abs = np.nanmax(np.abs(prof_mm))
        plt.title(f"Bulge profile (max |offset| ≈ {max_abs:.1f} mm)")
    else:
        plt.plot(y, prof_px)
        plt.xlabel("Normalized height (0=top, 1=bottom)")
        plt.ylabel("Lateral offset [px]")
        max_abs = np.nanmax(np.abs(prof_px))
        plt.title(f"Bulge profile (max |offset| ≈ {max_abs:.1f} px)")
    plt.grid(True)

    if out_path:
        fig1.savefig(os.path.join(out_path, "overlay.png"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_path, "profile.png"), dpi=200, bbox_inches="tight")

    return fig1, fig2, max_abs

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    # pick two points
    print("[INFO] Select TOP then BOTTOM on the displayed image...")
    P_top, P_bot = ginput_two_points(img)

    # rotate so baseline is vertical (pointing downward)
    v = (P_bot - P_top).astype(np.float32)
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    img_rot, M = rotate_about_point(img, angle_deg, P_top)
    P_top_rot = apply_transform_points(P_top[None,:], M)[0]
    P_bot_rot = apply_transform_points(P_bot[None,:], M)[0]

    # extract contour near the baseline band
    band_center_x = P_top_rot[0]
    contour, edges = extract_wall_contour(
        img_rot,
        band_center_x=float(band_center_x),
        band_half_width_px=args.band_px,
        canny1=args.canny1,
        canny2=args.canny2,
        dilate_it=args.dilate,
        erode_it=args.erode,
        use_clahe=args.gray_eq
    )
    if contour is None:
        raise RuntimeError("Contour extraction failed. Try different thresholds or provide clearer image.")

    # compute profile
    centers, prof_px, eL, eN = profile_along_baseline(
        contour, P_top_rot, P_bot_rot, bins=args.bins,
        smooth_win=args.smooth_win, smooth_poly=args.smooth_poly
    )

    # scale
    scale_px_per_mm = None
    if args.height_mm and args.height_mm > 0:
        px_len = np.linalg.norm(P_bot_rot - P_top_rot)
        scale_px_per_mm = px_len / float(args.height_mm)

    # save CSV
    csv_path = os.path.join(args.output_dir, "bulge_profile.csv")
    if scale_px_per_mm:
        prof_mm = prof_px / scale_px_per_mm
        data = np.vstack([centers, prof_px, prof_mm]).T
        header = "t_norm,offset_px,offset_mm"
    else:
        data = np.vstack([centers, prof_px]).T
        header = "t_norm,offset_px"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"[INFO] Saved CSV: {csv_path}")

    # visualize and save
    fig1, fig2, max_abs = visualize(img_rot, P_top_rot, P_bot_rot, centers, prof_px, scale_px_per_mm, out_path=args.output_dir)

    # show plots
    plt.show()

    # summary
    if scale_px_per_mm:
        max_mm = np.nanmax(np.abs(prof_px / scale_px_per_mm))
        print(f"[RESULT] Max |bulge| ≈ {max_mm:.1f} mm")
    else:
        max_px = np.nanmax(np.abs(prof_px))
        print(f"[RESULT] Max |bulge| ≈ {max_px:.1f} px")

if __name__ == "__main__":
    main()
