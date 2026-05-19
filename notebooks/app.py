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
        # Tạo 1 hàng chứa giá trị trung bình (Mean) chuẩn của file sạch
        input_row = X_clean.mean().to_frame().T
        
        # Đè số liệu nhập tay của người dùng vào đúng cột
        input_row[col_roa] = roa
        input_row[col_debt] = debt_ratio
        input_row[col_cash] = cash_flow
        
        # Đưa qua bộ scaler để chuẩn hóa
        input_scaled = scaler.transform(input_row)
        
        # Dự báo xác suất phá sản
        prob = model.predict_proba(input_scaled)[0][1] * 100
        
        # Hiển thị kết quả trực quan
        st.subheader(f"Xác suất xảy ra rủi ro phá sản: {prob:.2f}%")
        st.progress(int(prob))
        
        if prob > 50:
            st.error("🚨 CẢNH BÁO: Doanh nghiệp này đang nằm trong vùng nguy hiểm cao! Nguy cơ kiệt quệ tài chính lớn.")
            st.write("**Khuyến nghị từ hệ thống:** Ban quản trị cần lập tức cơ cấu lại các khoản nợ, cắt giảm chi phí vận hành và tối ưu hóa dòng tiền mặt để tránh mất thanh khoản.")
        else:
            st.success("✅ AN TOÀN: Doanh nghiệp có sức khỏe tài chính ổn định, nguy cơ phá sản thấp.")
            st.write("**Khuyến nghị từ hệ thống:** Tiếp tục duy trì chính sách quản lý tài chính hiện tại và tối ưu hóa hiệu suất sử dụng tài sản.")