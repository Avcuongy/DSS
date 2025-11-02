import numpy as np
import pandas as pd

class TOPSIS:
    """
    Lớp triển khai phương pháp ra quyết định đa tiêu chí TOPSIS (Technique for
    Order of Preference by Similarity to Ideal Solution), kết hợp với phương pháp 
    Entropy để xác định trọng số tiêu chí.

    Phương pháp TOPSIS giúp lựa chọn phương án tối ưu nhất bằng cách xác định 
    mức độ gần gũi của từng phương án với giải pháp lý tưởng dương (tốt nhất) 
    và xa nhất với giải pháp lý tưởng âm (xấu nhất).

    Attributes:
        df (pd.DataFrame): Ma trận quyết định ban đầu, trong đó:
            - Index: tên các phương án (ví dụ: nhà cung cấp).
            - Columns: tên các tiêu chí đánh giá.
        criteria_types (dict): Kiểu tối ưu của từng tiêu chí. 
            Key = tên tiêu chí, Value = "max" (càng lớn càng tốt) hoặc "min" (càng nhỏ càng tốt).
        norm_df (pd.DataFrame | None): Ma trận quyết định đã được chuẩn hóa.
        weights (pd.Series | None): Trọng số các tiêu chí tính theo phương pháp Entropy.
        weighted_norm (pd.DataFrame | None): Ma trận chuẩn hóa có trọng số.
        Ci_df (pd.DataFrame | None): Bảng hệ số Ci và xếp hạng các phương án.
    """

    def __init__(self, df, criteria_types):
        """
        Khởi tạo đối tượng TOPSIS với dữ liệu và loại tiêu chí.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu đầu vào, trong đó index là tên các phương án.
            criteria_types (dict): Từ điển chỉ rõ loại tiêu chí ("max" hoặc "min").

        Example:
            >>> data = pd.DataFrame({
            ...     'Giá': [250, 300, 200],
            ...     'Chất lượng': [7, 9, 8],
            ...     'Dịch vụ': [8, 7, 9]
            ... }, index=['A', 'B', 'C'])
            >>> types = {'Giá': 'min', 'Chất lượng': 'max', 'Dịch vụ': 'max'}
            >>> model = TOPSIS(data, types)
        """
        self.df = df.astype(float)
        self.criteria_types = criteria_types

        self.norm_df = None
        self.weights = None
        self.weighted_norm = None
        self.Ci_df = None

    def normalize(self):
        """
        Chuẩn hóa ma trận quyết định theo chuẩn Euclid (vector normalization).

        Returns:
            pd.DataFrame: Ma trận chuẩn hóa (norm_df), trong đó mỗi cột được chia
            cho căn bậc hai của tổng bình phương các phần tử trong cột.

        Formula:
            r_ij = x_ij / sqrt(sum(x_ij^2))

        Raises:
            ValueError: Nếu dữ liệu đầu vào không phải kiểu số.
        """
        norm_df = pd.DataFrame(index=self.df.index)
        for col in self.df.columns:
            denom = np.sqrt((self.df[col] ** 2).sum())
            norm_df[col] = self.df[col] / denom if denom != 0 else 0
        self.norm_df = norm_df
        return norm_df

    def calculate_entropy_weights(self):
        """
        Tính trọng số các tiêu chí theo phương pháp Entropy.

        Quá trình gồm:
            1. Chuẩn hóa dữ liệu (p_ij).
            2. Tính entropy (E_j) của từng tiêu chí.
            3. Xác định độ sai biệt (G_j = 1 - E_j).
            4. Chuẩn hóa trọng số: w_j = G_j / sum(G_j)

        Returns:
            pd.Series: Trọng số các tiêu chí (weights).

        Raises:
            Exception: Nếu dữ liệu chưa được chuẩn hóa (norm_df = None).
        """
        if self.norm_df is None:
            raise Exception("Phải chuẩn hóa dữ liệu (normalize) trước khi tính trọng số.")
        
        pij = self.norm_df.div(self.norm_df.sum(axis=0), axis=1).replace(0, 1e-12)
        m = self.norm_df.shape[0]

        Ej = (-1 / np.log(m) * (pij * np.log(pij)).sum(axis=0)).values
        Gj = 1 - Ej
        aj = Gj / Gj.sum()

        self.weights = pd.Series(aj, index=self.norm_df.columns)
        return self.weights

    def weighted_normalize(self):
        """
        Tạo ma trận chuẩn hóa có trọng số bằng cách nhân từng cột của ma trận 
        chuẩn hóa với trọng số tương ứng.

        Returns:
            pd.DataFrame: Ma trận chuẩn hóa có trọng số (weighted_norm).

        Raises:
            Exception: Nếu chưa tính chuẩn hóa hoặc trọng số.
        """
        if self.norm_df is None or self.weights is None:
            raise Exception("Phải chuẩn hóa và tính trọng số trước.")
        weighted_norm = self.norm_df * self.weights
        self.weighted_norm = weighted_norm
        return weighted_norm

    def calculate_Ci_and_ranking(self):
        """
        Tính hệ số gần với giải pháp lý tưởng (Ci) và xếp hạng các phương án.

        Các bước:
            1. Xác định tập tiêu chí cần tối đa hóa (J+) và tối thiểu hóa (J−).
            2. Xác định giải pháp lý tưởng dương (A+) và âm (A−).
            3. Tính khoảng cách đến A+ (D+) và A− (D−).
            4. Tính hệ số gần gũi Ci = D− / (D+ + D−).
            5. Xếp hạng theo Ci giảm dần.

        Returns:
            pd.DataFrame: Bảng gồm các cột:
                - 'Ci': Hệ số gần với lý tưởng.
                - 'Ranking': Thứ hạng (1 = tốt nhất).

        Raises:
            Exception: Nếu chưa tính ma trận chuẩn hóa có trọng số.
        """
        if self.weighted_norm is None:
            raise Exception("Phải tính ma trận chuẩn hóa trọng số trước.")

        weighted_norm = self.weighted_norm
        J_plus = [c for c, t in self.criteria_types.items() if t == "max"]
        J_minus = [c for c, t in self.criteria_types.items() if t == "min"]

        A_plus = pd.Series({
            col: (weighted_norm[col].max() if col in J_plus else weighted_norm[col].min())
            for col in weighted_norm.columns
        })
        A_minus = pd.Series({
            col: (weighted_norm[col].min() if col in J_plus else weighted_norm[col].max())
            for col in weighted_norm.columns
        })

        D_plus = np.sqrt(((weighted_norm - A_plus) ** 2).sum(axis=1))
        D_minus = np.sqrt(((weighted_norm - A_minus) ** 2).sum(axis=1))

        Ci = D_minus / (D_plus + D_minus)
        rank = Ci.rank(ascending=False, method="min").astype(int)

        self.Ci_df = pd.DataFrame({
            "Supplier": self.df.index,
            "Ci": Ci,
            "Ranking": rank
        }).set_index("Supplier").sort_values(by="Ci", ascending=False)

        return self.Ci_df