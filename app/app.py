import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Hàm trợ giúp ---

# Hàm này giả lập việc tạo ra dữ liệu kết quả TOPSIS
def tao_ket_qua_gia_lap(df):
    """Tạo DataFrame kết quả TOPSIS giả lập."""
    try:
        # Giả sử cột đầu tiên là tên Nhà cung cấp
        suppliers = df.iloc[:, 0].values
        n_suppliers = len(suppliers)
        
        # Tạo điểm Ci Score ngẫu nhiên
        fake_scores = np.random.rand(n_suppliers)
        
        # Tạo DataFrame kết quả
        df_ket_qua = pd.DataFrame({
            'Supplier': suppliers,
            'Ci Score': fake_scores
        })
        
        # Sắp xếp và thêm Ranking
        df_ket_qua = df_ket_qua.sort_values(by='Ci Score', ascending=False)
        df_ket_qua['Ranking'] = range(1, n_suppliers + 1)
        df_ket_qua['Ci Score'] = df_ket_qua['Ci Score'].round(4)
        
        return df_ket_qua.set_index('Supplier')
        
    except Exception as e:
        st.error(f"Lỗi khi tạo dữ liệu giả lập: {e}. Đảm bảo file có cột đầu tiên là tên nhà cung cấp.")
        return None

# Hàm này giả lập dữ liệu đã chuẩn hoá
def tao_du_lieu_chuan_hoa(df):
    """Tạo DataFrame chuẩn hoá giả lập."""
    df_norm = df.copy()
    try:
        # Chỉ chuẩn hoá các cột số
        for col in df_norm.select_dtypes(include=np.number).columns:
            norm = np.linalg.norm(df_norm[col])
            if norm != 0:
                df_norm[col] = df_norm[col] / norm
        return df_norm
    except:
        return df # Trả về df cũ nếu lỗi

# Hàm này giả lập trọng số Entropy
def tao_trong_so_entropy(df):
    """Tạo trọng số entropy giả lập."""
    try:
        # Lấy tên các tiêu chí (bỏ qua cột đầu tiên - tên NCC)
        criteria = df.columns[1:]
        n_criteria = len(criteria)
        
        # Tạo trọng số ngẫu nhiên và chuẩn hoá (tổng = 1)
        weights = np.random.rand(n_criteria)
        weights = weights / weights.sum()
        
        df_weights = pd.DataFrame({
            'Tiêu chí': criteria,
            'Trọng số Entropy': weights.round(4)
        })
        return df_weights
    except:
        return None

# Hàm để chuyển DataFrame sang file Excel (cho việc tải về)
@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='KetQuaTOPSIS')
    processed_data = output.getvalue()
    return processed_data

# --- Cấu hình Trang & Tiêu đề ---
st.set_page_config(page_title="Hệ thống TOPSIS", layout="wide")
st.title("HỆ THỐNG ĐÁNH GIÁ NHÀ CUNG CẤP - PHƯƠNG PHÁP TOPSIS")
st.markdown("---")

# --- Phần [1] Tải dữ liệu ---
st.header("1. Tải dữ liệu đầu vào")
uploaded_file = st.file_uploader("Chọn tệp Excel (.xlsx) hoặc CSV (.csv)", type=["xlsx", "csv"])

# Khởi tạo session state để lưu trữ dữ liệu
if 'data_goc' not in st.session_state:
    st.session_state.data_goc = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.data_goc = df
        st.success("Tải dữ liệu thành công!")
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        st.session_state.data_goc = None

# --- Hiển thị các bước nếu đã có dữ liệu ---
if st.session_state.data_goc is not None:
    df_goc = st.session_state.data_goc

    # Sử dụng Tabs để phân chia các bước theo mô tả
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 [2] Xem dữ liệu gốc", 
        "🔢 [3] Chuẩn hoá dữ liệu", 
        "⚖️ [4] Trọng số Entropy", 
        "🏆 [5 & 6] Kết quả TOPSIS & Xuất báo cáo"
    ])

    # --- Phần [2] Xem dữ liệu gốc ---
    with tab1:
        st.subheader("Bảng dữ liệu đầu vào")
        st.dataframe(df_goc)

    # --- Phần [3] Chuẩn hoá dữ liệu ---
    with tab2:
        st.subheader("Ma trận quyết định đã chuẩn hoá")
        st.write("Dữ liệu được chuẩn hoá (giả lập) bằng phương pháp vector normalization.")
        # Giả lập: Hiển thị dữ liệu đã chuẩn hoá
        df_normalized = tao_du_lieu_chuan_hoa(df_goc)
        st.dataframe(df_normalized)

    # --- Phần [4] Tính trọng số Entropy ---
    with tab3:
        st.subheader("Trọng số tiêu chí (phương pháp Entropy)")
        st.write("Trọng số khách quan (giả lập) được tính từ dữ liệu đầu vào.")
        # Giả lập: Hiển thị trọng số
        df_weights = tao_trong_so_entropy(df_goc)
        if df_weights is not None:
            st.dataframe(df_weights)
        else:
            st.warning("Không thể tính trọng số. Kiểm tra lại định dạng dữ liệu.")

    # --- Phần [5] & [6] Thực hiện TOPSIS và Xem kết quả ---
    with tab4:
        st.subheader("Thực hiện tính toán TOPSIS")
        st.write("Bấm nút bên dưới để chạy phân tích TOPSIS (giả lập) và xem kết quả.")
        
        # Nút thực hiện TOPSIS
        if st.button("🚀 Thực hiện TOPSIS"):
            # Giả lập quá trình tính toán
            with st.spinner("Đang tính toán..."):
                ket_qua = tao_ket_qua_gia_lap(df_goc)
                if ket_qua is not None:
                    # Lưu kết quả vào session state để có thể tải về
                    st.session_state.ket_qua = ket_qua
                    st.success("Đã hoàn tất tính toán TOPSIS!")
                else:
                    st.session_state.ket_qua = None
                    st.error("Tính toán thất bại.")
        
        # Hiển thị kết quả nếu đã tính toán
        if 'ket_qua' in st.session_state and st.session_state.ket_qua is not None:
            ket_qua_df = st.session_state.ket_qua
            
            st.markdown("---")
            st.subheader("Bảng kết quả xếp hạng")
            
            # [6] Hiển thị bảng kết quả
            st.dataframe(ket_qua_df)
            
            st.subheader("Biểu đồ kết quả (Ci Score)")
            
            # [6] Hiển thị biểu đồ
            # Tách riêng Ci Score để vẽ biểu đồ
            chart_data = ket_qua_df[['Ci Score']]
            st.bar_chart(chart_data)
            
            st.markdown("---")
            st.subheader("Xuất báo cáo")
            
            # [Xuất báo cáo]
            col1, col2 = st.columns(2)
            
            with col1:
                # Nút tải Excel
                excel_data = to_excel(ket_qua_df)
                st.download_button(
                    label="📥 Tải kết quả (.xlsx)",
                    data=excel_data,
                    file_name="ket_qua_topsis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                # Nút tải CSV (thay cho PDF để đơn giản)
                csv_data = ket_qua_df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📄 Tải kết quả (.csv)",
                    data=csv_data,
                    file_name="ket_qua_topsis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
else:
    st.info("Vui lòng tải tệp dữ liệu lên để bắt đầu phân tích.")