import pandas as pd

def normalize(df: pd.DataFrame, method: str = "Max") -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame theo phương pháp Min–Max.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào (chỉ chứa các biến số).
        method (str):
        
            'Max': Chuẩn hóa theo hướng Maximize (càng lớn càng tốt).    
            'Min': Chuẩn hóa theo hướng Minimize (càng nhỏ càng tốt).
    
    Returns:
        pd.DataFrame: DataFrame mới sau khi chuẩn hóa về [0, 1].
    """
    
    df_norm = df.copy()
    
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max == col_min:
            # Tránh chia cho 0 nếu tất cả giá trị bằng nhau
            df_norm[col] = 0.0
            continue

        if method.lower() == "max":
            # Càng lớn càng tốt
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        elif method.lower() == "min":
            # Càng nhỏ càng tốt
            df_norm[col] = (col_max - df[col]) / (col_max - col_min)
        else:
            raise ValueError("method phải là 'Max' hoặc 'Min'")
    
    return df_norm
