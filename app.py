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
import base64
from PIL import Image
from datetime import datetime

# --- 1. CẤU HÌNH HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="Student Engagement Systems",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS GIAO DIỆN ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0d11; }
    [data-testid="stSidebar"] { background-color: #11131a; border-right: 1px solid #2d3342; }
    
    /* Video Placeholder */
    .media-placeholder {
        background: radial-gradient(circle, #1a1c24 0%, #000000 100%);
        border: 2px solid #333;
        border-radius: 12px;
        height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #555;
        box-shadow: inset 0 0 20px #000;
        margin-bottom: 20px;
    }
    
    /* Telemetry Card */
    .telemetry-card {
        background: #161922;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00d2ff;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .telemetry-label { font-size: 12px; color: #8fa1b3; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-weight: 600; }
    .telemetry-value { font-size: 38px; font-weight: 800; color: #fff; line-height: 1; font-family: 'Courier New', monospace; }
    .telemetry-sub { font-size: 13px; color: #555; font-style: italic; margin-top: 5px; }
    
    /* Header */
    .header-container {
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 20px; border-bottom: 1px solid #333; margin-bottom: 20px;
    }
    .app-title { font-size: 32px; font-weight: 900; background: -webkit-linear-gradient(#00c6ff, #0072ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .date-display { font-size: 16px; color: #aaa; font-family: 'Courier New', monospace; font-weight: bold; }
    
    /* Button */
    .stButton>button {
        width: 100%; height: 60px; background: linear-gradient(90deg, #0062cc 0%, #00d2ff 100%);
        border: none; color: white; font-weight: 800; font-size: 15px;
        border-radius: 8px; text-transform: uppercase; letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { box-shadow: 0 0 20px rgba(0, 210, 255, 0.6); }
    .stButton>button:disabled { background: #333; color: #666; cursor: not-allowed; }

    /* Footer */
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0e1117; color: #888; text-align: center;
        padding: 10px; font-size: 12px; border-top: 1px solid #333;
        z-index: 9999; font-family: 'Courier New', monospace;
    }
    
    /* Table Styling */
    .table-container { width: 100%; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); border-radius: 8px; overflow: hidden; }
    thead tr { background-color: #0072ff; color: #ffffff; text-align: left; }
    th, td { padding: 12px 15px; color: #ddd; border-bottom: 1px solid #333; }
    tbody tr:nth-of-type(even) { background-color: #1a1c24; }
    
    .block-container { padding-bottom: 80px; }
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. STATE MANAGEMENT ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = None 
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'output_video_path' not in st.session_state: st.session_state.output_video_path = None
if 'video_name' not in st.session_state: st.session_state.video_name = ""
if 'seek_time' not in st.session_state: st.session_state.seek_time = 0
if 'is_scanning' not in st.session_state: st.session_state.is_scanning = False
if 'student_count' not in st.session_state: st.session_state.student_count = 0
if 'show_report_preview' not in st.session_state: st.session_state.show_report_preview = False

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

# --- 5. LOGIC & DATA ---
COLOR_MAP = {'listening': '#00ff88', 'writing': '#00d2ff', 'using_laptop': '#0072ff', 'looking_away': '#ffcc00', 'using_phone': '#ff6600', 'sleeping': '#ff0055'}
CLASS_NAMES = {0: 'listening', 1: 'looking_away', 2: 'sleeping', 3: 'using_laptop', 4: 'using_phone', 5: 'writing'}
POSITIVE_ACTIONS = ['listening', 'writing', 'using_laptop']

MEMBERS = [
    {"mssv": "23110063", "name": "Lương Thiện Thông", "dob": "26/01/2005", "class": "23110FIE3"},
    {"mssv": "23110006", "name": "Nguyễn Minh Cường", "dob": "13/05/2005", "class": "23110FIE4"},
    {"mssv": "23110022", "name": "Bùi Quang Huy", "dob": "04/05/2005", "class": "23110FIE3"},
    {"mssv": "23110056", "name": "Lê Nguyễn Gia Phúc", "dob": "26/08/2005", "class": "23110FIE3"},
]

def calculate_metrics(detections):
    if len(detections) == 0: return 0, {}, 0
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for cls_id in detections:
        label = CLASS_NAMES.get(int(cls_id), 'unknown')
        if label in counts: counts[label] += 1
    positive = sum(1 for cls_id in detections if CLASS_NAMES.get(int(cls_id), 'unknown') in POSITIVE_ACTIONS)
    score = round((positive / len(detections)) * 100, 2)
    return score, counts, len(detections)

def compress_report_data(df):
    if df is None or df.empty: return []
    compressed = []
    start_time = df.iloc[0]['Time']
    current_score = df.iloc[0]['Score']
    current_students = sum(df.iloc[0]['Counts'].values())
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        if row['Score'] != current_score or sum(row['Counts'].values()) != current_students:
            end_time = df.iloc[i-1]['Time']
            compressed.append({"TimeRange": f"{start_time}s - {end_time}s", "Score": current_score, "Students": current_students})
            start_time = row['Time']; current_score = row['Score']; current_students = sum(row['Counts'].values())
    compressed.append({"TimeRange": f"{start_time}s - {df.iloc[-1]['Time']}s", "Score": current_score, "Students": current_students})
    return compressed

def generate_html_report(df, filename, duration, alert_thresh):
    avg_score = df['Score'].mean()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    compressed_data = compress_report_data(df)
    
    all_counts = pd.DataFrame(df['Counts'].tolist()).sum()
    all_counts = all_counts[all_counts > 0]
    
    fig_pie = go.Figure(data=[go.Pie(labels=all_counts.index.tolist(), values=all_counts.values.tolist(), hole=0.5, marker_colors=[COLOR_MAP.get(l, '#888') for l in all_counts.index])])
    fig_pie.update_layout(title="Tổng hợp hành vi", height=300)
    pie_html = fig_pie.to_html(full_html=False, include_plotlyjs='cdn')
    
    table_rows = ""
    for item in compressed_data:
        color = "red" if item['Score'] < alert_thresh else "black"
        fw = "bold" if item['Score'] < alert_thresh else "normal"
        table_rows += f"<tr style='color:{color}; font-weight:{fw}'><td>{item['TimeRange']}</td><td>{item['Score']}%</td><td>{item['Students']}</td></tr>"

    html_content = f"""
    <html><head><style>
        body {{ font-family: Arial; background: #f4f6f9; padding: 20px; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #0072ff; border-bottom: 2px solid #0072ff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }} th {{ background-color: #0072ff; color: white; }}
    </style></head><body>
    <div class="container">
        <h1>BÁO CÁO PHÂN TÍCH LỚP HỌC</h1>
        <p><b>Video:</b> {filename} | <b>Ngày:</b> {now_str}</p>
        <p><b>GVHD:</b> Hoang Van Dung | <b>Lớp:</b> 01FIE - Nhóm 06</p>
        <hr>
        <h3>TỔNG QUAN</h3>
        <p>Độ tập trung TB: <b>{avg_score:.2f}%</b> | Số SV tối đa: <b>{st.session_state.student_count}</b></p>
        <h3>PHÂN BỐ HÀNH VI (TỔNG HỢP)</h3>{pie_html}
        <h3>DIỄN BIẾN CHI TIẾT (GỘP THEO TRẠNG THÁI)</h3>
        <table><tr><th>Khoảng Thời Gian</th><th>Độ Tập Trung</th><th>Số SV</th></tr>{table_rows}</table>
    </div></body></html>
    """
    return html_content

def get_filled_pie_chart(counts, title=""):
    clean = {k: v for k, v in counts.items() if v > 0}
    if not clean: return get_empty_chart("No Activity")
    labels = list(clean.keys()); values = list(clean.values())
    colors = [COLOR_MAP.get(l, '#888') for l in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.65, marker_colors=colors, textinfo='percent+label', textposition='inside')])
    layout = dict(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    if title: layout['title'] = dict(text=title, x=0.5, font=dict(color="#aaa"))
    fig.update_layout(**layout)
    return fig

def get_empty_chart(title="Waiting for Data"):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=320, xaxis=dict(showgrid=True, gridcolor='#222'), yaxis=dict(showgrid=True, gridcolor='#222'), title=dict(text=title, font=dict(color="#555")))
    return fig

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11698/11698436.png", width=100)
    st.markdown("### ⚙️ CẤU HÌNH HỆ THỐNG")
    st.markdown("---")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.4, 0.05)
    alert_threshold = st.slider("Ngưỡng cảnh báo (%)", 0, 100, 60, 5)
    st.markdown("---")
    st.markdown("**📝 Chú thích trạng thái**")
    for action, color in COLOR_MAP.items():
        st.markdown(f"<span style='color:{color}'>●</span> {action.replace('_', ' ').capitalize()}", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 KHỞI ĐỘNG LẠI"):
        st.session_state.clear(); st.rerun()

# ==================== HEADER ====================
now = datetime.now().strftime("%d/%m/%Y | %H:%M")
st.markdown(f"""<div class="header-container"><div class="app-title">STUDENT ENGAGEMENT SYSTEMS</div><div class="date-display">🔴 LIVE | {now}</div></div>""", unsafe_allow_html=True)

# ==================== TABS ====================
tab_vid, tab_img, tab_mem = st.tabs(["🎥 VIDEO ANALYTICS", "📸 IMAGE SCAN", "👥 GROUP MEMBERS"])

# ==================== TAB 1: VIDEO ====================
with tab_vid:
    # 1. SETUP LAYOUT
    col_main, col_ctrl = st.columns([3, 1])

    # 2. CONTROL PANEL
    with col_ctrl:
        st.markdown("**📁 VIDEO INPUT**")
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
        
        if uploaded_video and uploaded_video.name != st.session_state.video_name:
            st.session_state.video_name = uploaded_video.name
            st.session_state.processed_data = None
            st.session_state.output_video_path = None
            st.session_state.seek_time = 0
            st.session_state.show_report_preview = False

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.is_scanning:
            st.button("⏳ ĐANG PHÂN TÍCH...", disabled=True)
        elif st.session_state.processed_data is not None:
            st.success("✅ ĐÃ HOÀN TẤT")
            if st.button("🔄 QUÉT LẠI"):
                st.session_state.processed_data = None
                st.session_state.show_report_preview = False
                st.rerun()
        else:
            if st.button("▶️ BẮT ĐẦU PHÂN TÍCH"):
                if uploaded_video:
                    st.session_state.is_scanning = True
                    st.rerun()
                else:
                    st.toast("⚠️ Vui lòng upload video trước!", icon="🚫")

        st.markdown("<br><b>📊 THÔNG SỐ LIVE</b>", unsafe_allow_html=True)
        telemetry_placeholder = st.empty()

        def update_telemetry(score, students, status="STANDBY"):
            c = "#28a745" if score >= alert_threshold else "#dc3545"
            if status == "STANDBY": c = "#555"
            telemetry_placeholder.markdown(f"""
                <div class="telemetry-card" style="border-left-color: {c};">
                    <div class="telemetry-label">ĐỘ TẬP TRUNG AVG</div>
                    <div class="telemetry-value" style="color: {c}">{score}%</div>
                    <div class="telemetry-sub">{status}</div>
                </div>
                <div class="telemetry-card" style="border-left-color: #f1c40f;">
                    <div class="telemetry-label">SỐ LƯỢNG SV</div>
                    <div class="telemetry-value" style="color: #f1c40f">{students}</div>
                    <div class="telemetry-sub">Real-time detection</div>
                </div>
            """, unsafe_allow_html=True)

        if not st.session_state.is_scanning:
            if st.session_state.processed_data is not None:
                try:
                    idx = st.session_state.processed_data['Time'].sub(st.session_state.seek_time).abs().idxmin()
                    row = st.session_state.processed_data.iloc[idx]
                    update_telemetry(row['Score'], sum(row['Counts'].values()), "REVIEW MODE")
                except: update_telemetry(0, 0, "DATA ERR")
            else:
                update_telemetry(0, 0, "STANDBY")

    # 3. VIDEO AREA (PLACEHOLDER)
    with col_main:
        video_placeholder = st.empty()
        progress_placeholder = st.empty()

        if not uploaded_video:
            video_placeholder.markdown('<div class="media-placeholder"><h3>📡 NO SIGNAL</h3></div>', unsafe_allow_html=True)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            st.session_state.video_path = tfile.name

            if not st.session_state.is_scanning:
                if st.session_state.processed_data is not None and st.session_state.output_video_path:
                    # DONE -> Hiện video ĐÃ XỬ LÝ (Có box)
                    video_placeholder.video(st.session_state.output_video_path, start_time=st.session_state.seek_time)
                else:
                    # PREVIEW -> Hiện ảnh bìa
                    cap = cv2.VideoCapture(tfile.name)
                    ret, frame = cap.read()
                    if ret:
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Sẵn sàng quét", use_container_width=True)
                    cap.release()

    # 4. ANALYTICS AREA (Placeholder Cố định)
    st.divider()
    c_line, c_pie = st.columns([2, 1])
    with c_line:
        st.markdown("**📈 DIỄN BIẾN THEO THỜI GIAN (CLICK ĐỂ TUA)**")
        chart_spot = st.empty()
    with c_pie:
        st.markdown("**🥧 PHÂN BỐ HÀNH VI**")
        pie_spot = st.empty()

    # LOGIC CHART (KHI KHÔNG QUÉT)
    if not st.session_state.is_scanning:
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            fig_final = px.area(df, x="Time", y="Score", template="plotly_dark", height=320)
            fig_final.update_traces(line_color='#00d2ff', line_shape='spline', fillcolor='rgba(0, 210, 255, 0.2)')
            bad_pts = df[df['Score']<alert_threshold]
            fig_final.add_trace(go.Scatter(x=bad_pts['Time'], y=bad_pts['Score'], mode='markers', marker=dict(color='red', size=6), name='Cảnh báo'))
            fig_final.add_hline(y=alert_threshold, line_dash="dash", line_color="#ff4444")
            fig_final.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True, range=[0, 110]), margin=dict(l=0,r=0,t=20,b=0), clickmode='event+select', hovermode="x unified")
            
            with chart_spot:
                sel = st.plotly_chart(fig_final, use_container_width=True, on_select="rerun", key="timeline_chart")
                if sel and len(sel["selection"]["points"]) > 0:
                    selected_time = int(sel["selection"]["points"][0]["x"])
                    if selected_time != st.session_state.seek_time:
                        st.session_state.seek_time = selected_time
                        st.rerun() 
            
            # Logic Pie Chart (Tổng hợp nếu chưa tua)
            if st.session_state.seek_time == 0:
                all_counts_df = pd.DataFrame(df['Counts'].tolist())
                total_aggregate = all_counts_df.sum().to_dict()
                pie_spot.plotly_chart(get_filled_pie_chart(total_aggregate, title="Tổng hợp toàn bộ Video"), use_container_width=True)
            else:
                idx = df['Time'].sub(st.session_state.seek_time).abs().idxmin()
                current_counts = df.iloc[idx]['Counts']
                pie_spot.plotly_chart(get_filled_pie_chart(current_counts, title=f"Tại giây: {st.session_state.seek_time}"), use_container_width=True)
        else:
            chart_spot.plotly_chart(get_empty_chart(), use_container_width=True)
            pie_spot.plotly_chart(get_empty_chart("No Data"), use_container_width=True)

    # 5. SCANNING LOOP 
    if st.session_state.is_scanning:
        video_placeholder.empty() 
        chart_spot.empty()
        pie_spot.empty()
        # ---------------------------------------------------------

        chart_spot.plotly_chart(get_empty_chart(), use_container_width=True)
        pie_spot.plotly_chart(get_empty_chart("Analyzing..."), use_container_width=True)
        
        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_idx = 0; temp_data = []
        SKIP = 3; CHART_RATE = 15; RESIZE_H = 480
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect = width / height
        new_w = int(RESIZE_H * aspect)
        
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        # Dùng codec AVC1 để chạy được trên Web
        try: fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        except: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out_writer = cv2.VideoWriter(temp_out, fourcc, fps/SKIP, (new_w, RESIZE_H))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % SKIP == 0:
                h, w = frame.shape[:2]
                if h > RESIZE_H: frame = cv2.resize(frame, (new_w, RESIZE_H))
                
                res = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated = res[0].plot()
                out_writer.write(annotated) 
                
                score, counts, tot = calculate_metrics(res[0].boxes.cls.cpu().numpy())
                ts = round(frame_idx/fps, 2)
                st.session_state.student_count = max(st.session_state.student_count, tot)
                
                # Update UI
                video_placeholder.image(annotated, channels="BGR", use_container_width=True)
                progress_placeholder.progress(min(frame_idx/total, 1.0))
                update_telemetry(score, tot, "SCANNING...")
                temp_data.append({"Time": ts, "Score": score, "Counts": counts})
                
                if len(temp_data) % CHART_RATE == 0:
                    df_live = pd.DataFrame(temp_data)
                    fig = px.area(df_live, x="Time", y="Score", range_y=[0,110], template="plotly_dark", height=320)
                    fig.update_traces(line_color='#00d2ff', line_shape='spline')
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
                    chart_spot.plotly_chart(fig, use_container_width=True, key=f"c_{frame_idx}")
                    if sum(counts.values())>0: pie_spot.plotly_chart(get_filled_pie_chart(counts, "Real-time"), use_container_width=True, key=f"p_{frame_idx}")
            frame_idx += 1
            
        cap.release()
        out_writer.release()
        progress_placeholder.empty()
        
        st.session_state.output_video_path = temp_out
        st.session_state.processed_data = pd.DataFrame(temp_data)
        st.session_state.is_scanning = False
        st.rerun()

    # 6. REPORT SECTION
    st.divider()
    with st.container():
        c_rep1, c_rep2, c_rep3 = st.columns([1, 2, 1])
        with c_rep2:
            st.markdown("### 📝 XUẤT BÁO CÁO TỔNG HỢP")
            if st.button("TẠO BÁO CÁO (PREVIEW)", use_container_width=True):
                if not uploaded_video: st.error("🚫 Vui lòng upload video!")
                elif st.session_state.processed_data is None: st.warning("⚠️ Hãy chạy phân tích trước!")
                else: st.session_state.show_report_preview = True

    if st.session_state.show_report_preview and st.session_state.processed_data is not None:
        st.markdown("---")
        df = st.session_state.processed_data
        avg_sc = df['Score'].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Thời lượng", f"{df.iloc[-1]['Time']}s")
        m2.metric("Độ tập trung TB", f"{avg_sc:.1f}%")
        m3.metric("Số cảnh báo", f"{len(df[df['Score'] < alert_threshold])}")
        
        report_html = generate_html_report(df, st.session_state.video_name, df.iloc[-1]['Time'], alert_threshold)
        b64 = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="Report.html"><button style="width:100%; height:50px; background:#28a745; color:white; border-radius:8px; cursor:pointer;">📥 TẢI BÁO CÁO HTML</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# ==================== TAB 2: IMAGE ====================
with tab_img:
    c1, c2 = st.columns([3, 1])
    with c2:
        img_up = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if st.button("🚀 SCAN"): 
            if img_up: st.session_state.img_processed = True
        img_tel = st.empty()
    with c1:
        if img_up:
            im = np.array(Image.open(img_up))
            if 'img_processed' in st.session_state and st.session_state.img_processed:
                r = model.predict(im, conf=conf_threshold)[0]
                st.image(r.plot(), use_container_width=True)
                s, c, t = calculate_metrics(r.boxes.cls.cpu().numpy())
                col = "#28a745" if s>=60 else "#dc3545"
                img_tel.markdown(f"""<div class="telemetry-card" style="border-left-color:{col}"><div class="telemetry-label">SCORE</div><div class="telemetry-value" style="color:{col}">{s}%</div></div>""", unsafe_allow_html=True)
                if t>0: st.plotly_chart(get_filled_pie_chart(c), use_container_width=True)
            else: st.image(im, caption="Original", use_container_width=True)
        else: st.markdown('<div class="media-placeholder"><h3>📸 NO IMAGE</h3></div>', unsafe_allow_html=True)

# ==================== TAB 3: MEMBERS ====================
with tab_mem:
    st.markdown("### 👥 DANH SÁCH THÀNH VIÊN NHÓM 06")
    html_table = "<table><thead><tr><th>STT</th><th>Mã SV</th><th>Họ và Tên</th><th>Lớp</th><th>Ngày Sinh</th><th>Vai trò</th></tr></thead><tbody>"
    for i, m in enumerate(MEMBERS, 1):
        html_table += f"<tr><td>{i}</td><td>{m['mssv']}</td><td><b>{m['name']}</b></td><td>{m['class']}</td><td>{m['dob']}</td><td>Thành viên</td></tr>"
    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.info("**Thông tin nhóm:**\n\n* Nhóm: 06\n* Lớp: Digital Image Processing_ Nhom 01FIE")
    with c2: st.success("**Giảng viên hướng dẫn:**\n\nThầy Hoàng Văn Dũng")

st.markdown("""<div class="footer">Digital Image Processing | Instructor: Hoang Van Dung | Class: Digital Image Processing_ Nhom 01FIE | Group: 06</div>""", unsafe_allow_html=True)