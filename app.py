import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import plotly.express as px
import time
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Giám Sát Lớp Học", page_icon="🎓", layout="wide")
st.markdown("""
    <style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    .stButton>button { width: 100%; height: 3em; font-weight: bold; background-color: #FF4B4B; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load model best.pt, nếu không có thì dùng yolov8n
    path = "models/best.pt" if os.path.exists("models/best.pt") else "best.pt"
    if not os.path.exists(path): return YOLO("yolov8n.pt")
    return YOLO(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo: {e}")
    st.stop()

# --- CONFIG ---
CLASS_NAMES = {0: 'listening', 1: 'looking_away', 2: 'sleeping', 3: 'using_laptop', 4: 'using_phone', 5: 'writing'}
POSITIVE_ACTIONS = ['listening', 'writing', 'using_laptop']

def calculate_metrics(detections):
    if len(detections) == 0: return 0
    positive = sum(1 for cls_id in detections if CLASS_NAMES.get(int(cls_id), 'unknown') in POSITIVE_ACTIONS)
    return round((positive / len(detections)) * 100, 2)

# ================= GIAO DIỆN CHÍNH =================
st.title("🎓 Phân Tích Lớp Học (Live Monitor)")

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Placeholder hiển thị Video & Thông số
    st_video_spot = st.empty()
    st_metrics_spot = st.empty()

    # TỰ ĐỘNG CHẠY KHI CÓ FILE
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # --- CẤU HÌNH SIÊU MƯỢT ---
    RESIZE_H = 640 
    
    timeline_data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. RESIZE ẢNH (Bí kíp mượt mà)
        h, w = frame.shape[:2]
        if h > RESIZE_H:
            scale = RESIZE_H / h
            frame = cv2.resize(frame, (int(w * scale), RESIZE_H))
        
        # 2. AI DETECT
        results = model.predict(frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        
        # 3. TÍNH TOÁN
        detections = results[0].boxes.cls.cpu().numpy()
        score = calculate_metrics(detections)
        timestamp = round(frame_idx / fps, 2)
        
        # Vẽ thông tin lên video
        color = (0, 255, 0) if score >= 60 else (0, 0, 255)
        cv2.putText(annotated_frame, f"FOCUS: {score}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 4. HIỂN THỊ NGAY
        st_video_spot.image(annotated_frame, channels="BGR", use_container_width=True)
        st_metrics_spot.info(f"⏱ Thời gian: {timestamp}s  |  📊 Độ tập trung: {score}%")
        
        # 5. LƯU DỮ LIỆU
        timeline_data.append({
            "Time": timestamp,
            "Score": score,
            "Status": "Tốt" if score >= 60 else "Mất tập trung"
        })
        
        frame_idx += 1
        
    cap.release()
    st_video_spot.empty() # Xóa video khi xong
    st_metrics_spot.success("✅ Đã hoàn tất! Xem báo cáo bên dưới.")

    # ================= BÁO CÁO (HIỆN RA SAU KHI CHẠY XONG) =================
    if timeline_data:
        st.divider()
        st.header("📈 Báo Cáo Chi Tiết")
        
        df = pd.DataFrame(timeline_data)
        
        # Biểu đồ Timeline
        fig = px.bar(df, x="Time", y="Score", color="Status",
                        color_discrete_map={"Tốt": "#28a745", "Mất tập trung": "#dc3545"},
                        title="Diễn biến độ tập trung",
                        height=300)
        
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True, range=[0, 110]),
            bargap=0
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê & Download
        c1, c2 = st.columns(2)
        avg = df["Score"].mean()
        c1.metric("Độ tập trung trung bình", f"{avg:.1f}%")
        
        csv = df.to_csv(index=False).encode('utf-8')
        c2.download_button("📥 Tải Báo Cáo CSV", csv, "bao_cao.csv", "text/csv")