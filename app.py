import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from PIL import Image

# --- 1. CẤU HÌNH HỆ THỐNG & FIX LỖI ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="AI Vision Sentinel Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS GIAO DIỆN KHOA HỌC (SCI-FI UI) ---
st.markdown("""
    <style>
    /* Nền ứng dụng tối */
    .stApp { background-color: #0e1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #12141c; border-right: 1px solid #2b313e; }
    
    /* Khung Video Placeholder */
    .video-placeholder {
        background-color: #000;
        border: 2px dashed #333;
        border-radius: 12px;
        height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #555;
    }
    
    /* Card thông số (Telemetry Box) */
    .telemetry-card {
        background: #1a1c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00a8ff;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .telemetry-card:hover { transform: scale(1.02); }
    
    .telemetry-label { font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .telemetry-value { font-size: 32px; font-weight: 800; color: #fff; line-height: 1.2; }
    .telemetry-sub { font-size: 12px; color: #666; font-style: italic; }
    
    /* Nút bấm Sci-fi */
    .stButton>button {
        width: 100%;
        height: 55px;
        background: linear-gradient(135deg, #0062cc 0%, #00c6ff 100%);
        border: none;
        color: white;
        font-weight: 700;
        font-size: 16px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.6);
        transform: translateY(-2px);
    }
    .stButton>button:disabled {
        background: #333;
        color: #777;
        cursor: not-allowed;
    }

    /* Tiêu đề section */
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #00d2ff;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    
    /* Ẩn header mặc định */
    header {visibility: hidden;}
    .block-container { padding-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# --- 3. QUẢN LÝ TRẠNG THÁI (STATE MANAGEMENT) ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = None 
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'seek_time' not in st.session_state: st.session_state.seek_time = 0
if 'is_scanning' not in st.session_state: st.session_state.is_scanning = False
if 'current_stats' not in st.session_state: st.session_state.current_stats = {"score": 0, "students": 0, "status": "Waiting"}

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_model():
    path = "models/best.pt" if os.path.exists("models/best.pt") else "best.pt"
    if not os.path.exists(path): return YOLO("yolov8n.pt")
    return YOLO(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Lỗi Model: {e}")
    st.stop()

# --- 5. LOGIC PHÂN TÍCH ---
COLOR_MAP = {
    'listening': '#28a745',    # Xanh lá
    'writing': '#17a2b8',      # Xanh dương nhạt
    'using_laptop': '#007bff', # Xanh dương đậm
    'looking_away': '#ffc107', # Vàng
    'using_phone': '#fd7e14',  # Cam
    'sleeping': '#dc3545'      # Đỏ
}
CLASS_NAMES = {0: 'listening', 1: 'looking_away', 2: 'sleeping', 3: 'using_laptop', 4: 'using_phone', 5: 'writing'}
POSITIVE_ACTIONS = ['listening', 'writing', 'using_laptop']

def calculate_metrics(detections):
    if len(detections) == 0: return 0, {}, 0
    
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for cls_id in detections:
        label = CLASS_NAMES.get(int(cls_id), 'unknown')
        if label in counts: counts[label] += 1
            
    positive = sum(1 for cls_id in detections if CLASS_NAMES.get(int(cls_id), 'unknown') in POSITIVE_ACTIONS)
    score = round((positive / len(detections)) * 100, 2)
    total_students = len(detections)
    
    return score, counts, total_students

def generate_anomaly_report(df, threshold):
    report_lines = []
    if df is None or df.empty: return ""
    is_bad = False; start_bad = 0
    for index, row in df.iterrows():
        if row['Score'] < threshold and not is_bad:
            is_bad = True; start_bad = row['Time']
        elif row['Score'] >= threshold and is_bad:
            is_bad = False; report_lines.append(f"⏱ {start_bad}s - {row['Time']}s: Mất tập trung (Avg: {row['Score']}%)")
    if is_bad: report_lines.append(f"⏱ {start_bad}s - {df.iloc[-1]['Time']}s: Mất tập trung đến hết")
    return "\n".join(report_lines)

def get_empty_chart():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=300, 
        xaxis=dict(showgrid=True, gridcolor='#333', title="Time (s)"),
        yaxis=dict(showgrid=True, gridcolor='#333', range=[0, 100], title="Focus %"),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    return fig

def get_empty_pie():
    fig = go.Figure(go.Pie(labels=[], values=[], hole=0.6))
    fig.update_layout(template="plotly_dark", height=300, 
                      annotations=[dict(text="No Data", x=0.5, y=0.5, font_size=20, showarrow=False)],
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    st.markdown("### ⚙️ SYSTEM CONFIG")
    st.markdown("---")
    conf_threshold = st.slider("👁️ AI Sensitivity", 0.0, 1.0, 0.4, 0.05)
    alert_threshold = st.slider("⚠️ Alert Threshold (%)", 0, 100, 60, 5)
    st.markdown("---")
    if st.button("🔄 FACTORY RESET"):
        st.session_state.clear()
        st.rerun()

# ==================== MAIN UI ====================
st.title("🚀 AI VISION SENTINEL PRO")

# TABS
tab_vid, tab_img = st.tabs(["🎥 LIVE MONITOR (VIDEO)", "📸 STATIC SCAN (IMAGE)"])

# ==================== TAB VIDEO ====================
with tab_vid:
    # --- LAYOUT: 3 Phần Video - 1 Phần Control ---
    col_main, col_ctrl = st.columns([3, 1])

    # 1. CỘT ĐIỀU KHIỂN & THÔNG SỐ (BÊN PHẢI)
    with col_ctrl:
        st.markdown('<div class="section-header">📡 CONTROL PANEL</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Input Source", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
        
        # LOGIC NÚT BẤM
        if st.session_state.is_scanning:
            st.button("⏳ PROCESSING...", disabled=True)
        elif st.session_state.processed_data is not None:
            st.success("✅ COMPLETED")
            if st.button("🔄 RE-SCAN"):
                st.session_state.processed_data = None
                st.rerun()
        else:
            if st.button("▶️ START ANALYSIS"):
                if uploaded_video:
                    st.session_state.is_scanning = True
                    st.rerun()
                else:
                    st.toast("⚠️ Please upload video source first!", icon="🚫")

        # --- LIVE TELEMETRY ---
        st.markdown('<div class="section-header" style="margin-top:20px;">📊 LIVE TELEMETRY</div>', unsafe_allow_html=True)
        
        telemetry_placeholder = st.empty()
        
        def update_telemetry(score, students, status="IDLE"):
            c = "#28a745" if score >= alert_threshold else "#dc3545"
            if status == "IDLE": c = "#555"
            
            telemetry_placeholder.markdown(f"""
                <div class="telemetry-card" style="border-left-color: {c};">
                    <div class="telemetry-label">FOCUS LEVEL (ĐỘ TẬP TRUNG)</div>
                    <div class="telemetry-value" style="color: {c}">{score}%</div>
                    <div class="telemetry-sub">Status: {status}</div>
                </div>
                
                <div class="telemetry-card" style="border-left-color: #f1c40f;">
                    <div class="telemetry-label">ACTIVE STUDENTS (SỐ SV)</div>
                    <div class="telemetry-value" style="color: #f1c40f">{students}</div>
                    <div class="telemetry-sub">Person Detection Count</div>
                </div>
            """, unsafe_allow_html=True)

        # Trạng thái ban đầu
        if not st.session_state.is_scanning:
            if st.session_state.processed_data is not None:
                seek_idx = st.session_state.processed_data['Time'].sub(st.session_state.seek_time).abs().idxmin()
                curr = st.session_state.processed_data.iloc[seek_idx]
                total_std_review = sum(curr['Counts'].values())
                update_telemetry(curr['Score'], total_std_review, "REVIEW MODE")
            else:
                update_telemetry(0, 0, "IDLE")

    # 2. KHU VỰC VIDEO MONITOR (BÊN TRÁI)
    with col_main:
        video_placeholder = st.empty()
        progress_placeholder = st.empty()

        if not uploaded_video:
            video_placeholder.markdown('<div class="video-placeholder"><h3>🎞️ SIGNAL LOST<br>Waiting for input...</h3></div>', unsafe_allow_html=True)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            if st.session_state.video_path != tfile.name and not st.session_state.is_scanning:
                st.session_state.video_path = tfile.name
                st.session_state.processed_data = None
                st.session_state.seek_time = 0

            if not st.session_state.is_scanning:
                if st.session_state.processed_data is not None:
                    video_placeholder.video(st.session_state.video_path, start_time=st.session_state.seek_time)
                else:
                    cap = cv2.VideoCapture(tfile.name)
                    ret, frame = cap.read()
                    if ret:
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Source Ready", use_container_width=True)
                    cap.release()

    # --- KHU VỰC ANALYTICS (BIỂU ĐỒ) ---
    st.divider()
    c_line, c_pie = st.columns([2, 1])

    with c_line:
        st.markdown('<div class="section-header">📈 TIMELINE ANALYTICS (TUA VIDEO)</div>', unsafe_allow_html=True)
        chart_spot = st.empty()
        
    with c_pie:
        st.markdown('<div class="section-header">🥧 BEHAVIOR DISTRIBUTION (CHI TIẾT)</div>', unsafe_allow_html=True)
        pie_spot = st.empty()

    if not st.session_state.is_scanning and st.session_state.processed_data is None:
        chart_spot.plotly_chart(get_empty_chart(), use_container_width=True)
        pie_spot.plotly_chart(get_empty_pie(), use_container_width=True)

    # ==================== LOGIC SCANNING (VÒNG LẶP CHÍNH) ====================
    if st.session_state.is_scanning:
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_idx = 0
        temp_data = []
        RESIZE_H = 480
        SKIP = 3
        CHART_UPDATE_RATE = 10 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % SKIP == 0:
                h, w = frame.shape[:2]
                if h > RESIZE_H: scale = RESIZE_H / h; frame = cv2.resize(frame, (int(w * scale), RESIZE_H))
                
                res = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated_frame = res[0].plot()
                dects = res[0].boxes.cls.cpu().numpy()
                score, counts, total_std = calculate_metrics(dects)
                timestamp = round(frame_idx / fps, 2)
                
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                progress_placeholder.progress(min(frame_idx / total_frames, 1.0))
                
                # CẬP NHẬT LIVE TELEMETRY
                update_telemetry(score, total_std, "SCANNING...")
                
                temp_data.append({"Time": timestamp, "Score": score, "Counts": counts})
                
                if len(temp_data) % CHART_UPDATE_RATE == 0:
                    df_live = pd.DataFrame(temp_data)
                    
                    fig_live = px.area(df_live, x="Time", y="Score", range_y=[0, 110], 
                                       template="plotly_dark", height=300)
                    fig_live.update_traces(line_color='#00d2ff', fillcolor='rgba(0, 210, 255, 0.2)')
                    fig_live.update_layout(xaxis=dict(showgrid=True, gridcolor='#333'), margin=dict(l=0,r=0,t=0,b=0))
                    chart_spot.plotly_chart(fig_live, use_container_width=True, key=f"c_{frame_idx}")
                    
                    if sum(counts.values()) > 0:
                        labels = list(counts.keys())
                        values = list(counts.values())
                        colors = [COLOR_MAP.get(l, '#888') for l in labels]
                        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
                        fig_pie.update_layout(
                            template="plotly_dark", height=300, 
                            margin=dict(l=0,r=0,t=0,b=0),
                            showlegend=True,
                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.0)
                        )
                        pie_spot.plotly_chart(fig_pie, use_container_width=True, key=f"p_{frame_idx}")

            frame_idx += 1
        
        cap.release()
        progress_placeholder.empty()
        st.session_state.processed_data = pd.DataFrame(temp_data)
        st.session_state.is_scanning = False
        st.rerun()

    # ==================== KẾT QUẢ CUỐI (INTERACTIVE) ====================
    if not st.session_state.is_scanning and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # TIMELINE INTERACTIVE
        fig_final = px.area(df, x="Time", y="Score", template="plotly_dark", height=300)
        fig_final.add_hline(y=alert_threshold, line_dash="dash", line_color="red")
        fig_final.update_layout(
            title="💡 Click vào biểu đồ để tua lại",
            xaxis=dict(fixedrange=True, title="Time (s)"),
            yaxis=dict(fixedrange=True, range=[0, 110], title="Focus %"),
            margin=dict(l=0, r=0, t=30, b=0),
            clickmode='event+select', hovermode="x unified"
        )
        
        with chart_spot:
            sel = st.plotly_chart(fig_final, use_container_width=True, on_select="rerun")
            if sel and len(sel["selection"]["points"]) > 0:
                st.session_state.seek_time = int(sel["selection"]["points"][0]["x"])
                
                # Update Info khi tua
                seek_idx = df['Time'].sub(st.session_state.seek_time).abs().idxmin()
                curr = df.iloc[seek_idx]
                cnts = curr['Counts']
                total_std_rev = sum(cnts.values())
                
                if total_std_rev > 0:
                    labels = list(cnts.keys())
                    values = list(cnts.values())
                    colors = [COLOR_MAP.get(l, '#888') for l in labels]
                    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
                    fig_pie.update_layout(
                        template="plotly_dark", height=300,
                        showlegend=True, legend=dict(orientation="v", x=1.0),
                        margin=dict(l=0,r=0,t=0,b=0)
                    )
                    pie_spot.plotly_chart(fig_pie, use_container_width=True)
                
                update_telemetry(curr['Score'], total_std_rev, f"REWIND: {curr['Time']}s")

        # EXPORT REPORT
        st.divider()
        st.markdown("### 📑 EXPORT DATA")
        
        bad_rep = generate_anomaly_report(df, alert_threshold)
        csv_txt = f"""AI CLASSROOM REPORT
====================================
Average Focus Score: {df['Score'].mean():.2f}%
Alert Threshold: {alert_threshold}%
====================================
ANOMALY LOG (THỜI ĐIỂM MẤT TẬP TRUNG):
{bad_rep}
====================================
RAW DATA:
Time,Score,Active_Students
"""
        for i,r in df.iterrows(): 
            std_sum = sum(r['Counts'].values())
            csv_txt += f"{r['Time']},{r['Score']},{std_sum}\n"
        
        c_dl1, c_dl2 = st.columns([1, 2])
        c_dl1.download_button("📥 DOWNLOAD CSV", csv_txt, "classroom_report.csv", "text/csv")
        with c_dl2: 
            with st.expander("📄 Xem nhanh log bất thường"): st.text(bad_rep if bad_rep else "Buổi học tốt!")

# ==================== TAB IMAGE (SỬA LỖI NAME ERROR) ====================
with tab_img:
    st.markdown('<div class="section-header">📸 STATIC IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    img_up = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if img_up:
        im = np.array(Image.open(img_up))
        if st.button("🚀 SCAN IMAGE"):
            r = model.predict(im, conf=conf_threshold)[0]
            st.image(r.plot(), use_container_width=True)
            sc, cnt, tot = calculate_metrics(r.boxes.cls.cpu().numpy())
            
            c1, c2 = st.columns(2)
            c1.metric("FOCUS SCORE", f"{sc}%")
            c2.metric("STUDENTS", tot)
            
            if tot > 0:
                labels = list(cnt.keys())
                values = list(cnt.values())
                colors = [COLOR_MAP.get(l, '#888') for l in labels]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
                fig.update_layout(template="plotly_dark", height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)