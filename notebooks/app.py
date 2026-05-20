import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Cấu hình giao diện trang web
st.set_page_config(page_title="Hệ thống Dự báo Phá sản Doanh nghiệp", layout="wide")

st.title("📊 PHẦN MỀM DỰ BÁO NGUY CƠ KIỆT QUỆ TÀI CHÍNH & PHÁ SẢN")
st.write("---")

# 2. Tải mô hình, bộ chuẩn hóa và file dữ liệu ĐÃ LÀM SẠCH (Đã thêm ../ để sửa đường dẫn)
@st.cache_resource
def load_assets():
    # Thêm ../ vào trước đường dẫn để lùi lại một thư mục ngoài
    model = joblib.load('../models/bankruptcy_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    
    # Thêm ../ vào trước data/dulieuketoan_clean.csv
    df_clean = pd.read_csv('../data/dulieuketoan_clean.csv')
    df_clean.columns = [col.strip() for col in df_clean.columns]
    return model, scaler, df_clean

try:
    model, scaler, df_clean = load_assets()
    st.sidebar.success("🤖 Đã kết nối với mô hình AI thành công!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải dữ liệu/mô hình: {e}")
    st.stop()

# Tự động loại bỏ cột mục tiêu nếu có xuất hiện trong dữ liệu
target_column = 'Bankrupt?' 
X_clean = df_clean.drop(columns=[target_column]) if target_column in df_clean.columns else df_clean

# 3. Định nghĩa tên 3 chỉ số xuất hiện trên giao diện
col_roa = 'ROA(C) before interest and depreciation before interest'
col_debt = 'Debt ratio %'
col_cash = 'Cash flow rate'

# 4. Thiết kế giao diện chia làm 2 cột
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📥 Nhập chỉ số kế toán")
    st.write("Nhập số liệu hiện tại của doanh nghiệp cần kiểm tra:")
    
    # Lấy giá trị Min, Max, Mean trực tiếp từ file sạch
    roa = st.number_input("1. Chỉ số ROA (C):", 
                          min_value=float(X_clean[col_roa].min()), 
                          max_value=float(X_clean[col_roa].max()), 
                          value=float(X_clean[col_roa].mean()), step=0.01)
    
    debt_ratio = st.number_input("2. Tỷ lệ nợ (Debt Ratio %):", 
                                 min_value=float(X_clean[col_debt].min()), 
                                 max_value=float(X_clean[col_debt].max()), 
                                 value=float(X_clean[col_debt].mean()), step=0.01)
    
    cash_flow = st.number_input("3. Tỷ suất dòng tiền (Cash Flow Rate):", 
                                min_value=float(X_clean[col_cash].min()), 
                                max_value=float(X_clean[col_cash].max()), 
                                value=float(X_clean[col_cash].mean()), step=0.01)
    
    predict_btn = st.button("Chạy dự báo nguy cơ", type="primary")

with col2:
    st.header("📈 Kết quả phân tích từ AI")
    
if predict_btn:
    # 1. Tạo một hàng dữ liệu trống có các cột giống bộ dữ liệu gốc
    input_row = X_clean.mean().to_frame().T
    
    # 2. GÁN SỐ THEO VỊ TRÍ CỘT
    input_row.iloc[0, 0] = roa
    input_row.iloc[0, 1] = debt_ratio
    input_row.iloc[0, 2] = cash_flow
        
    # 3. Ép mô hình tính toán lại dựa trên số vừa nhập
    input_scaled = scaler.transform(input_row)
    prob = model.predict_proba(input_scaled)[0][1] * 100
    
    # 4. ĐƯA KẾT QUẢ VÀO ĐÚNG CỘT BÊN PHẢI (with col2)
    # (Lưu ý: Nếu code phía trên của bạn đặt tên cột bên phải là col_right hoặc đặt tên khác, 
    # thì bạn hãy đổi chữ 'col2' này thành đúng tên biến đó nhé!)
    with col2:
        st.write("---")
        st.subheader("📊 Kết quả phân tích từ AI")
        st.markdown(f"### **Xác suất xảy ra rủi ro phá sản: {prob:.2f}%**")
        
        # THAY ĐỔI NGƯỠNG: Vì nền dữ liệu rất thấp, chỉ cần rủi ro > 0.1% 
        # (tức là tăng gấp 5 lần bình thường) là đã phải báo động !
        if prob < 0.10:
            st.success("✅ AN TOÀN: Doanh nghiệp có sức khỏe tài chính ổn định, nguy cơ phá sản thấp.")
            st.info("💡 **Khuyến nghị:** Tiếp tục duy trì chính sách quản lý tài chính hiện tại.")
        else:
            st.error("🚨 CẢNH BÁO: Doanh nghiệp đang có dấu hiệu kiệt quệ tài chính, nguy cơ phá sản ở mức CAO so với bình thường!")
            st.warning("💡 **Khuyến nghị:** Cần rà soát ngay các khoản nợ ngắn hạn và cơ cấu lại dòng tiền gấp.")
# 5. HIỂN THỊ CÁC BIỂU ĐỒ PHÂN TÍCH (Tận dụng thành quả của Bạn 2 & Bạn 4)
# -------------------------------------------------------------------------
st.write("---")
st.header("📊 BIỂU ĐỒ PHÂN TÍCH CHI TIẾT TỪ DỮ LIỆU")

# Tạo 2 tab trên giao diện web để người xem bấm qua lại cho gọn
tab1, tab2 = st.tabs(["📈 Phân tích Xu hướng (Bạn 2)", "🎯 Chỉ số Quan trọng nhất (Bạn 4)"])

with tab1:
    st.subheader("Phân tích phân phối và xu hướng các chỉ số tài chính")
    # Đọc và hiển thị ảnh biểu đồ Boxplot của Bạn 2
    # Vì file app.py nằm chung thư mục notebooks với ảnh nên chỉ cần gọi thẳng tên ảnh
    try:
        st.image("02_boxplots.png", caption="Biểu đồ hộp phân tích các chỉ số kế toán cốt lõi", use_container_width=True)
        st.image("detail_ROA.png", caption="Phân tích chi tiết chỉ số ROA đối với các doanh nghiệp phá sản", use_container_width=True)
    except:
        st.info("Chưa tìm thấy file ảnh biểu đồ trong thư mục. Hãy đảm bảo các file ảnh .png nằm cùng thư mục với file app.py nhé!")

with tab2:
    st.subheader("Top các chỉ số kế toán tác động mạnh nhất đến nguy cơ phá sản")
   
    try:
        st.image("detail_Vòng_quay_tài_sản.png", caption="Biểu đồ mức độ ảnh hưởng của các biến lên mô hình AI", use_container_width=True)
    except:
        st.info("Đang cập nhật biểu đồ chỉ số quan trọng ...")