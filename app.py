"""드론 탐지 Streamlit 데모. 실행: streamlit run app.py"""

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def default_weights():
    """가장 최근에 학습한 best.pt, 학습 결과가 없으면 저장소의 best.pt."""
    runs = sorted((ROOT / "runs" / "detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else ROOT / "best.pt"


@st.cache_resource
def load_model(path):
    return YOLO(path)


st.title("YOLOv8m Drone Detection")

weights = st.sidebar.text_input("모델 가중치", str(default_weights()))
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

if not Path(weights).is_file():
    st.error(f"가중치 파일을 찾을 수 없습니다: {weights}")
    st.stop()

model = load_model(weights)

uploaded_file = st.file_uploader("드론 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    r = model.predict(image, conf=conf)[0]
    # plot()은 OpenCV 형식(BGR) 배열을 반환
    st.image(r.plot(), channels="BGR", caption=f"드론 {len(r.boxes)}개 탐지")
