import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Giám Sát Lớp Học",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH CHO GIAO DIỆN ĐẸP ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center; }
    .alert-box { padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; }
    .high-score { background-color: #28a745; }
    .low-score { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# --- KHAI BÁO NHÃN & MÀU SẮC ---
CLASS_NAMES = {
    0: 'listening', 1: 'looking_away', 2: 'sleeping', 
    3: 'using_laptop', 4: 'using_phone', 5: 'writing'
}

# Nhóm hành vi: Tích cực (Positive) & Tiêu cực (Negative)
POSITIVE_ACTIONS = ['listening', 'writing', 'using_laptop']
NEGATIVE_ACTIONS = ['sleeping', 'using_phone', 'looking_away']

# Màu sắc cho Bounding Box
COLORS = {
    'listening': (0, 255, 0), 'writing': (0, 200, 255), 'using_laptop': (0, 255, 255), # Xanh
    'sleeping': (0, 0, 255), 'using_phone': (0, 0, 150), 'looking_away': (0, 100, 255) # Đỏ/Cam
}

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# --- HÀM TÍNH TOÁN CHỈ SỐ ---
def calculate_metrics(detections):
    counts = {name: 0 for name in CLASS_NAMES.values()}
    total_students = len(detections)
    
    if total_students == 0:
        return counts, 0, 0

    for cls_id in detections:
        label = CLASS_NAMES[int(cls_id)]
        counts[label] += 1
    
    positive_count = sum(counts[act] for act in POSITIVE_ACTIONS)
    engagement_score = (positive_count / total_students) * 100
    
    return counts, total_students, round(engagement_score, 2)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Cấu hình Hệ thống")
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    
    # Upload Model nếu chưa có trong folder
    model_file = "models/best.pt"
    if not os.path.exists(model_file):
        st.warning("⚠️ Chưa tìm thấy file models/best.pt")
        uploaded_model = st.file_uploader("Upload file model (.pt)", type=['pt'])
        if uploaded_model:
            with open(model_file, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success("Đã tải model lên!")
            st.rerun()
    
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.4)
    alert_threshold = st.slider("Ngưỡng cảnh báo tập trung (%)", 0, 100, 60)
    st.markdown("---")
    st.info("💡 **Hệ thống phân tích:**\n- **Tích cực:** Listening, Writing, Laptop\n- **Tiêu cực:** Phone, Sleeping, Looking Away")

# --- MAIN APP ---
if os.path.exists(model_file):
    model = load_model(model_file)
    
    st.title("🎓 Hệ Thống Phân Tích Mức Độ Tập Trung Lớp Học")
    
    tab1, tab2 = st.tabs(["🖼️ Phân Tích Ảnh", "🎥 Phân Tích Video"])

    # ================= TAB 1: XỬ LÝ ẢNH =================
    with tab1:
        st.header("Upload Ảnh Lớp Học")
        img_file = st.file_uploader("Chọn ảnh (.jpg, .png)", type=['jpg', 'png', 'jpeg'])
        
        if img_file:
            # Đọc ảnh
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Xử lý AI
            results = model.predict(img, conf=conf_threshold)
            detections = results[0].boxes.cls.cpu().numpy()
            
            # Tính toán
            counts, total, score = calculate_metrics(detections)
            
            # Vẽ Bounding Box
            for result in results:
                res_plotted = result.plot()
                res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # --- HIỂN THỊ KẾT QUẢ ---
            col_img, col_stat = st.columns([2, 1])
            
            with col_img:
                st.image(res_plotted, caption="Kết quả nhận diện", use_column_width=True)
            
            with col_stat:
                st.subheader("📊 Thống Kê")
                
                # Hiển thị điểm số lớn
                color_class = "high-score" if score >= alert_threshold else "low-score"
                st.markdown(f"""
                <div class="alert-box {color_class}" style="text-align: center;">
                    <h2>{score}%</h2>
                    <p>MỨC ĐỘ TẬP TRUNG</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Tổng số sinh viên", f"{total} người")
                
                # Biểu đồ tròn
                df_counts = pd.DataFrame(list(counts.items()), columns=['Hành vi', 'Số lượng'])
                fig = px.pie(df_counts, values='Số lượng', names='Hành vi', hole=0.4, 
                             color='Hành vi',
                             color_discrete_map={'listening':'green', 'sleeping':'red', 'using_phone':'darkred'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Nút xuất báo cáo
                csv = df_counts.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Xuất báo cáo CSV", data=csv, file_name="report_image.csv", mime="text/csv")

    # ================= TAB 2: XỬ LÝ VIDEO =================
    with tab2:
        st.header("Upload Video Giám Sát")
        video_file = st.file_uploader("Chọn video (.mp4, .avi)", type=['mp4', 'avi', 'mov'])
        
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name
            
            col_vid_left, col_vid_right = st.columns([3, 1])
            
            with col_vid_left:
                start_btn = st.button("▶️ Bắt đầu Phân tích Video")
            
            if start_btn:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Thanh tiến trình & Placeholder hiển thị video
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                # Dữ liệu theo thời gian
                timeline_data = []
                frame_idx = 0
                
                # Tạo file output tạm
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_path = temp_output.name
                
                # Codec cho video output (Dùng mp4v cho tương thích cơ bản)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # AI Predict
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    detections = results[0].boxes.cls.cpu().numpy()
                    
                    # Tính toán chỉ số cho frame này
                    _, _, score = calculate_metrics(detections)
                    
                    # Lưu dữ liệu timeline
                    timestamp = round(frame_idx / fps, 2)
                    timeline_data.append({
                        "Time (s)": timestamp,
                        "Engagement (%)": score,
                        "Status": "Low" if score < alert_threshold else "High"
                    })
                    
                    # Vẽ lên frame
                    annotated_frame = results[0].plot()
                    
                    # Hiển thị thông số trực tiếp lên video
                    color = (0, 255, 0) if score >= alert_threshold else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Engagement: {score}%", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    
                    out.write(annotated_frame)
                    
                    # Cập nhật giao diện (mỗi 5 frame update 1 lần cho nhẹ)
                    if frame_idx % 5 == 0:
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Đang xử lý: {timestamp}s")
                        progress = frame_idx / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"⏳ Đang xử lý... {int(progress*100)}%")
                    
                    frame_idx += 1
                
                cap.release()
                out.release()
                progress_bar.progress(100)
                status_text.success("✅ Đã xử lý xong!")
                frame_placeholder.empty() # Xóa ảnh preview để hiện video final

                # --- XỬ LÝ KẾT QUẢ VIDEO ---
                st.divider()
                
                # 1. BIỂU ĐỒ DIỄN BIẾN (Tua được bằng mắt)
                df_timeline = pd.DataFrame(timeline_data)
                
                # Tạo biểu đồ vùng (Area Chart) với Plotly
                fig_timeline = px.area(df_timeline, x='Time (s)', y='Engagement (%)', 
                                       title="📈 Biểu đồ Mức độ tập trung theo thời gian",
                                       color='Status',
                                       color_discrete_map={'High': '#28a745', 'Low': '#dc3545'})
                
                # Thêm đường kẻ ngang mức báo động
                fig_timeline.add_hline(y=alert_threshold, line_dash="dash", line_color="red", 
                                       annotation_text=f"Ngưỡng {alert_threshold}%")
                
                # Đánh dấu các khoảng thời gian nguy hiểm
                low_eng = df_timeline[df_timeline['Engagement (%)'] < alert_threshold]
                if not low_eng.empty:
                    st.error(f"⚠️ CẢNH BÁO: Lớp học mất tập trung trong {len(low_eng)/fps:.1f} giây (Các vạch màu đỏ trên biểu đồ).")
                
                st.plotly_chart(fig_timeline, use_container_width=True)

                # 2. XEM VIDEO & TẢI VỀ
                col_result_video, col_result_stat = st.columns([2, 1])
                
                with col_result_video:
                    st.subheader("🎥 Video Kết Quả")
                    # Lưu ý: Để video mp4 play được trên web, cần convert sang h264. 
                    # Vì ta dùng opencv thuần nên ta sẽ cho user download file để xem chuẩn nhất 
                    # hoặc cố gắng hiển thị (có thể lỗi codec tùy trình duyệt)
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button("⬇️ Tải Video đã phân tích", f, file_name="processed_video.mp4")
                        
                with col_result_stat:
                    st.subheader("📑 Tổng kết Video")
                    avg_score = df_timeline['Engagement (%)'].mean()
                    min_score = df_timeline['Engagement (%)'].min()
                    
                    st.metric("Tập trung trung bình", f"{avg_score:.1f}%")
                    st.metric("Tập trung thấp nhất", f"{min_score:.1f}%", delta_color="inverse")
                    
                    # Xuất báo cáo chi tiết
                    csv_video = df_timeline.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Tải Báo cáo chi tiết (CSV)", csv_video, 
                                       file_name="video_analytics.csv", mime="text/csv")
else:
    st.warning("Vui lòng tải file model best.pt vào thư mục 'models' để bắt đầu.")