import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="centered")
st.title("Canvas BG MWE")

# 400x300 の単色画像を背景にする（確実に PIL / RGB）
bg = Image.new("RGB", (400, 300), (180, 200, 220))
st.image(bg, caption="確認用プレビュー（表示されるはず）")

cv = st_canvas(
    background_image=bg.copy(),      # PIL.Image を copy() で渡す
    background_color="rgba(0,0,0,0)",# 完全透明（上書き防止）
    width=400, height=300,           # int で明示
    drawing_mode="polygon",
    stroke_width=3, stroke_color="#ff7f00",
    fill_color="rgba(255,165,0,0.25)",
    update_streamlit=True,
    display_toolbar=True,
    key="mwe_canvas",
)
st.write("OKならここにキャンバスが出て、背景に薄い水色が表示されます。")
