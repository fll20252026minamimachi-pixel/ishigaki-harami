# app.py — Ishigaki Bulge Analyzer (Streamlit, full working version)

import io
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Ishigaki Bulge Analyzer", layout="wide")
st.title("🧱 Ishigaki Bulge Analyzer")
st.caption("手順: ①画像アップ → ②（任意）ROIで斜面を多角形囲み → ③上端→下端をクリック → 解析")

# ====== Utility functions ======
def rotate_about_point(image, angle_deg, center):
    M = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated, M

def apply_transform_points(pts, M):
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    out = (M @ pts_h.T).T
    return out

def hsv_suppressions(img_bgr, mask=None, veg_h=(30, 90), sky_h=(85, 140),
                     sat_th=40, use_veg=True, use_sky=True):
    H, W = img_bgr.shape[:2]
    keep = np.ones((H, W), np.uint8) * 255 if mask is None else (mask > 0).astype(np.uint8) * 255
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hch = hsv[:, :, 0]; Sch = hsv[:, :, 1]
    if use_veg:
        v1, v2 = veg_h
        veg = (Hch >= v1) & (Hch <= v2) & (Sch >= sat_th)
        keep[veg] = 0
    if use_sky:
        s1, s2 = sky_h
        sky = (Hch >= s1) & (Hch <= s2) & (Sch >= sat_th)
        keep[sky] = 0
    return keep

def canny_edges(img_bgr, c1, c2, dil, ero, gray_eq=True, keep_mask=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray_eq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if keep_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=keep_mask)
    edges = cv2.Canny(gray, c1, c2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if dil > 0: edges = cv2.dilate(edges, k, dil)
    if ero > 0: edges = cv2.erode(edges, k, ero)
    return edges

def choose_contour_in_band(edges, band_center_x, band_half_width_px, min_area=3000):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    H, W = edges.shape[:2]
    x0 = int(max(0, band_center_x - band_half_width_px))
    x1 = int(min(W - 1, band_center_x + band_half_width_px))
    best, best_score = None, -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        pts = c.reshape(-1, 2); xs = pts[:, 0]; ys = pts[:, 1]
        in_band = ((xs >= x0) & (xs <= x1)).mean()
        vspan = float(ys.max() - ys.min())
        score = in_band * (1.0 + vspan / max(1, H))
        if score > best_score:
            best_score, best = score, c
    return best

def profile_along_baseline(contour_pts, P_top, P_bot, bins=200, smooth_win=31, smooth_poly=3):
    v = (P_bot - P_top).astype(np.float32)
    L = np.linalg.norm(v)
    if L < 1e-3: return None, None
    eL = v / L; eN = np.array([-eL[1], eL[0]], dtype=np.float32)
    w = contour_pts.reshape(-1, 2).astype(np.float32) - P_top[None, :]
    t = (w @ eL) / L; d = (w @ eN)
    m = (t >= 0) & (t <= 1); t = t[m]; d = d[m]
    if len(t) < 10: return None, None
    edges = np.linspace(0, 1, bins + 1); centers = (edges[:-1] + edges[1:]) * 0.5
    idx = np.digitize(t, edges) - 1
    prof = np.full(bins, np.nan, np.float32)
    for i in range(bins):
        vals = d[idx == i]
        if len(vals) > 0: prof[i] = np.median(vals)
    valid = ~np.isnan(prof)
    if valid.sum() > max(smooth_win, smooth_poly + 2):
        inds = np.where(valid)[0]
        for i in range(bins):
            if np.isnan(prof[i]):
                j = inds[np.argmin(np.abs(inds - i))]; prof[i] = prof[j]
        if smooth_win % 2 == 0: smooth_win += 1
        prof = savgol_filter(prof, smooth_win, smooth_poly, mode="interp")
    return centers, prof

def analyze(img_bgr, top_xy, bottom_xy, roi_mask=None, band_px=80,
            gray_eq=True, canny1=60, canny2=120, dil=2, ero=1,
            mask_veg=True, mask_sky=True, veg_h=(30, 90), sky_h=(85, 140),
            min_area=3000, height_mm=None):
    P_top = np.array(top_xy, np.float32); P_bot = np.array(bottom_xy, np.float32)
    angle_deg = np.degrees(np.arctan2((P_bot - P_top)[1], (P_bot - P_top)[0]))
    img_rot, M = rotate_about_point(img_bgr, angle_deg, P_top)
    P_top_rot = apply_transform_points(P_top[None, :], M)[0]
    P_bot_rot = apply_transform_points(P_bot[None, :], M)[0]

    if roi_mask is not None:
        roi_mask = cv2.warpAffine(roi_mask, M, (img_bgr.shape[1], img_bgr.shape[0]), flags=cv2.INTER_NEAREST)

    keep_mask = None
    if mask_veg or mask_sky or roi_mask is not None:
        base = roi_mask if roi_mask is not None else np.ones(img_rot.shape[:2], np.uint8) * 255
        keep_mask = hsv_suppressions(img_rot, mask=base, veg_h=veg_h, sky_h=sky_h,
                                     sat_th=40, use_veg=mask_veg, use_sky=mask_sky)

    band_center_x = int(P_top_rot[0])
    edges = canny_edges(img_rot, canny1, canny2, dil, ero, gray_eq, keep_mask)
    contour = choose_contour_in_band(edges, band_center_x, band_px, min_area=min_area)

    centers, prof = (None, None)
    if contour is not None:
        centers, prof = profile_along_baseline(contour, P_top_rot, P_bot_rot)

    scale_px_per_mm = None
    if height_mm and height_mm > 0:
        px_len = float(np.linalg.norm(P_bot_rot - P_top_rot))
        scale_px_per_mm = px_len / float(height_mm)

    return {
        "img_rot": img_rot, "edges": edges, "contour": contour,
        "P_top_rot": P_top_rot, "P_bot_rot": P_bot_rot,
        "centers": centers, "prof_px": prof, "scale_px_per_mm": scale_px_per_mm,
        "band": (max(0, band_center_x - band_px), min(img_rot.shape[1] - 1, band_center_x + band_px))
    }

# ====== Sidebar (parameters) ======
with st.sidebar:
    st.header("設定")
    height_mm = st.number_input("石垣の高さ [mm]（任意）", min_value=0, value=0, step=100)
    band_px   = st.slider("探索帯の幅 [px]", 40, 180, 80, 5)
    canny1    = st.slider("Canny1", 20, 150, 60, 5)
    canny2    = st.slider("Canny2", 50, 250, 120, 5)
    gray_eq   = st.checkbox("明るさ均一化（CLAHE）", True)
    mask_veg  = st.checkbox("緑（草木）を抑制", True)
    mask_sky  = st.checkbox("青（空）を抑制", True)
    st.caption("ヒント: うまく拾わない時は ROI を狭める/探索帯を調整")

# ====== Uploader ======
uploaded = st.file_uploader("石垣画像をアップロード", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("上のボタンから石垣の画像（JPG/PNG）をアップロードしてください。")
    st.stop()

# ====== Load image via Pillow (安定) ======
file_bytes = uploaded.read()
pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")  # 必ず RGB
W, H = pil_img.size
img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 解析用

display_w = int(min(800, W))
display_h = int(H * display_w / W)
bg_pil_disp = pil_img.resize((display_w, display_h), Image.BILINEAR)

# プレビュー（確認用。落ち着いたら消してOK）
st.image(bg_pil_disp, caption="キャンバスに渡す背景（プレビュー）", use_column_width=True)

st.markdown("""
    <style>
    /* st_canvas の内部 canvas 2枚（背景層/描画層）を不透明＆白に固定 */
    [data-testid="stCanvas"] canvas {
        background: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.image(bg_pil_disp, caption="プレビュー（ここに画像が出る）", use_column_width=True)
st.write("DEBUG:", type(bg_pil_disp), getattr(bg_pil_disp, "mode", None), bg_pil_disp.size)
# 期待: <class 'PIL.Image.Image'> , 'RGB' , (display_w, display_h)



# ====== Canvas 1: ROI polygon ======
st.subheader("1) ROI（任意）：石垣の斜面を多角形で囲む → ダブルクリックで確定")
roi_canvas = st_canvas(
    background_image=bg_pil_disp.copy(),   # PIL(RGB).copy()
    background_color="#ffffff",            # ← 透明やrgba(…,0)をやめて白で塗る
    width=display_w, height=display_h,
    drawing_mode="polygon",
    stroke_width=3, stroke_color="#ffa500",
    fill_color="rgba(255,165,0,0.25)",
    display_toolbar=False, update_streamlit=False,
    key="roi_canvas_v1",
)

# ROIマスク生成（無くても解析可能）
roi_mask = None
if roi_canvas.json_data and len(roi_canvas.json_data.get("objects", [])) > 0:
    obj = roi_canvas.json_data["objects"][-1]
    pts = []
    if "path" in obj:
        for cmd in obj["path"]:
            if cmd[0] in ("L", "M"):
                pts.append([cmd[1], cmd[2]])
    elif obj.get("type") == "polygon" and "points" in obj:
        pts = [[p["x"], p["y"]] for p in obj["points"]]
    if len(pts) >= 3:
        pts = np.array(pts, dtype=np.float32)
        scale = W / float(display_w)
        pts_img = (pts * scale).astype(np.int32)
        roi_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(roi_mask, [pts_img.reshape(-1, 1, 2)], 255)

# ====== Canvas 2: Baseline (top→bottom) ======
st.subheader("2) 基準線：上端 → 下端の順に2点をクリック")
click_canvas = st_canvas(
    background_image=bg_pil_disp.copy(),
    background_color="#ffffff",            # ← 同上
    width=display_w, height=display_h,
    drawing_mode="point",
    display_toolbar=False, update_streamlit=False,
    key="click_canvas_v1",
)

# クリック点取り出し
points = []
if click_canvas.json_data:
    for obj in click_canvas.json_data.get("objects", []):
        if obj.get("type") == "circle":
            points.append([obj["left"], obj["top"]])
if len(points) < 2:
    st.warning("上端→下端の順に2点タップしてください。")
    st.stop()

scale = W / float(display_w)
P_top = (np.array(points[0], dtype=np.float32) * scale)
P_bot = (np.array(points[1], dtype=np.float32) * scale)

# ====== Analyze ======
res = analyze(
    img_bgr, top_xy=P_top, bottom_xy=P_bot, roi_mask=roi_mask,
    band_px=band_px, gray_eq=gray_eq, canny1=canny1, canny2=canny2,
    mask_veg=mask_veg, mask_sky=mask_sky
)

img_rot = res["img_rot"]; edges = res["edges"]; contour = res["contour"]
P_top_rot = res["P_top_rot"]; P_bot_rot = res["P_bot_rot"]
centers = res["centers"]; prof_px = res["prof_px"]; scale_px_per_mm = res["scale_px_per_mm"]
x0, x1 = res["band"]

# ====== Overlay / Edges ======
overlay = img_rot.copy()
cv2.line(overlay, tuple(P_top_rot.astype(int)), tuple(P_bot_rot.astype(int)), (0, 255, 0), 2)
cv2.rectangle(overlay, (x0, 0), (x1, overlay.shape[0] - 1), (0, 0, 255), 1)
if contour is not None:
    cv2.drawContours(overlay, [contour], -1, (0, 165, 255), 2)

col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption="Overlay（基準線/探索帯/輪郭）", use_column_width=True)
with col2:
    st.image(edges, caption="Edges（デバッグ）", use_column_width=True)

# ====== Profile ======
st.subheader("3) 孕みプロファイル")
if centers is None or prof_px is None:
    st.error("輪郭が十分に得られませんでした。ROIや探索帯・Cannyを調整して再試行してください。")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    scale_local = None
    if height_mm and height_mm > 0:
        px_len = float(np.linalg.norm(P_bot_rot - P_top_rot))
        scale_local = px_len / float(height_mm)

    if scale_local:
        prof_mm = prof_px / scale_local
        ax.plot(centers, prof_mm); ax.set_ylabel("Lateral offset [mm]")
        mx = float(np.nanmax(np.abs(prof_mm))); unit = "mm"
    else:
        ax.plot(centers, prof_px); ax.set_ylabel("Lateral offset [px]")
        mx = float(np.nanmax(np.abs(prof_px))); unit = "px"
    ax.set_xlabel("Normalized height (0=top, 1=bottom)")
    ax.grid(True); ax.set_title(f"Max |offset| ≈ {mx:.1f} {unit}")
    st.pyplot(fig)

    import pandas as pd
    df = pd.DataFrame({"t_norm": centers, "offset_px": prof_px})
    if scale_local: df["offset_mm"] = prof_px / scale_local
    st.download_button(
        "CSVをダウンロード",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bulge_profile.csv", mime="text/csv"
    )
