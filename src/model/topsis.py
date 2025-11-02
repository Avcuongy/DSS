import numpy as np
import pandas as pd


class TOPSIS:
    """
    Triển khai phương pháp TOPSIS (Technique for Order of Preference by Similarity
    to Ideal Solution) kết hợp với phương pháp Entropy để xác định trọng số.
    """

    def __init__(self, df, criteria_types):
        """
        Khởi tạo đối tượng TOPSIS.

        Args:
            df (pd.DataFrame): DataFrame dữ liệu, với các phương án (hàng)
                               và các tiêu chí (cột).
            criteria_types (dict): Từ điển xác định hướng của tiêu chí.
                                   Key là tên tiêu chí (str),
                                   Value là 'max' (lợi ích) hoặc 'min' (chi phí).
        """
        self.df = df.astype(float)
        self.criteria_types = criteria_types

        self.norm_df = None
        self.weights = None
        self.weighted_norm = None
        self.Ci_df = None

    def normalize(self):
        """
        Chuẩn hóa ma trận quyết định theo chuẩn Euclid (Vector Normalization).

        Công thức: $r_{ij} = x_{ij} / \sqrt{\sum_{i=1}^{m} x_{ij}^2}$
        Kết quả được lưu vào `self.norm_df`.

        Returns:
            pd.DataFrame: DataFrame đã được chuẩn hóa.
        """
        norm_df = pd.DataFrame()
        for col in self.df.columns:
            denom = np.sqrt((self.df[col] ** 2).sum())
            norm_df[col] = self.df[col] / denom if denom != 0 else 0
        self.norm_df = norm_df
        return norm_df

    def calculate_entropy_weights(self):
        """
        Tính trọng số khách quan của các tiêu chí bằng phương pháp Entropy.

        Kết quả được lưu vào `self.weights`.

        Returns:
            pd.Series: Series chứa trọng số của mỗi tiêu chí.

        Raises:
            Exception: Nếu dữ liệu chưa được chuẩn hóa (chưa chạy `normalize()`).
        """
        if self.norm_df is None:
            raise Exception(
                "Phải chuẩn hóa dữ liệu (normalize) trước khi tính trọng số."
            )

        # Tính pij (phân bố xác suất)
        pij = self.norm_df.div(self.norm_df.sum(axis=0), axis=1).replace(0, 1e-12)
        m = self.norm_df.shape[0]
        Ej = (-1 / np.log(m) * (pij * np.log(pij)).sum(axis=0)).values
        Gj = 1 - Ej
        aj = Gj / Gj.sum()
        self.weights = pd.Series(aj, index=self.norm_df.columns)
        return self.weights

    def weighted_normalize(self):
        """
        Tạo ma trận quyết định chuẩn hóa có trọng số.

        Nhân ma trận chuẩn hóa ($r_{ij}$) với trọng số Entropy ($w_j$).
        Công thức: $v_{ij} = r_{ij} \times w_j$
        Kết quả được lưu vào `self.weighted_norm`.

        Returns:
            pd.DataFrame: DataFrame chuẩn hóa đã nhân trọng số.

        Raises:
            Exception: Nếu chưa chuẩn hóa hoặc chưa tính trọng số.
        """
        if self.norm_df is None or self.weights is None:
            raise Exception("Phải chuẩn hóa và tính trọng số trước.")
        weighted_norm = self.norm_df * self.weights
        self.weighted_norm = weighted_norm
        return weighted_norm

    def calculate_Ci_and_ranking(self):
        """
        Tính hệ số gần với giải pháp lý tưởng ($C_i$) và xếp hạng các phương án.

        Xác định giải pháp lý tưởng (PIS, $A^+$) và phi lý tưởng (NIS, $A^-$),
        tính khoảng cách ($D^+, D^-$) và cuối cùng là điểm $C_i$.
        Công thức: $C_i = D_i^- / (D_i^+ + D_i^-)$
        Kết quả được lưu vào `self.Ci_df`.

        Returns:
            pd.DataFrame: DataFrame chứa cột 'Ci' (điểm) và 'Ranking' (thứ hạng),
                          đã được sắp xếp theo thứ hạng.

        Raises:
            Exception: Nếu chưa tính ma trận chuẩn hóa trọng số.
        """
        if self.weighted_norm is None:
            raise Exception("Phải tính ma trận chuẩn hóa trọng số trước.")

        weighted_norm = self.weighted_norm

        J_plus = [c for c, t in self.criteria_types.items() if t == "max"]
        J_minus = [c for c, t in self.criteria_types.items() if t == "min"]

        A_plus = pd.Series(
            {
                col: (
                    weighted_norm[col].max()
                    if col in J_plus
                    else weighted_norm[col].min()
                )
                for col in weighted_norm.columns
            }
        )
        A_minus = pd.Series(
            {
                col: (
                    weighted_norm[col].min()
                    if col in J_plus
                    else weighted_norm[col].max()
                )
                for col in weighted_norm.columns
            }
        )

        D_plus = np.sqrt(((weighted_norm - A_plus) ** 2).sum(axis=1))
        D_minus = np.sqrt(((weighted_norm - A_minus) ** 2).sum(axis=1))

        Ci = D_minus / (D_plus + D_minus)
        rank = Ci.rank(ascending=False, method="min").astype(int)

        self.Ci_df = pd.DataFrame({"Ci": Ci, "Ranking": rank}).sort_values(
            by="Ci", ascending=False
        )
        return self.Ci_df

    # Các hàm in kết quả theo yêu cầu (có thể gọi riêng biệt)
    def print_normalization(self):
        """In ma trận quyết định đã chuẩn hóa."""
        if self.norm_df is None:
            self.normalize()
        print("Ma trận quyết định sau khi chuẩn hóa:")
        print(self.norm_df)

    def print_entropy_weights(self):
        """In trọng số các tiêu chí theo Entropy."""
        if self.weights is None:
            self.calculate_entropy_weights()
        print("Trọng số các tiêu chí theo Entropy:")
        print(self.weights)

    def print_weighted_normalize(self):
        """In ma trận chuẩn hóa có trọng số."""
        if self.weighted_norm is None:
            self.weighted_normalize()
        print("Ma trận chuẩn hóa có trọng số:")
        print(self.weighted_norm)

    def print_Ci_ranking(self):
        """In bảng kết quả Ci và xếp hạng cuối cùng."""
        if self.Ci_df is None:
            self.calculate_Ci_and_ranking()
        print("Bảng kết quả Ci và xếp hạng:")
        print(self.Ci_df)
