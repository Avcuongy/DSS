import numpy as np
import pandas as pd
from typing import Literal, List

class AHP:
    """
    Analytic Hierarchy Process (AHP) class for multi-criteria decision making.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        cols (List[str]): List of column names to be used in AHP analysis
    """
    
    def __init__(self, df: pd.DataFrame, cols: List[str]):
        self.df = df.copy()
        self.cols = cols
        self.pairwise_matrix = None
        self.normalized_matrix = None
        self.weights = None
        self.consistency_ratio = None
        
    def set_pairwise_matrix(self, matrix: np.ndarray):
        """
        Set the pairwise comparison matrix.
        
        Args:
        -----------
        matrix : np.ndarray
            Square matrix of pairwise comparisons (n x n)
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if matrix.shape[0] != len(self.cols):
            raise ValueError(f"Matrix size must match number of columns ({len(self.cols)})")
        
        self.pairwise_matrix = matrix
        self._calculate_weights()
        
    def get_normalized_matrix(self) -> pd.DataFrame:
        """
        Return the normalized pairwise comparison matrix.
        
        Returns:
        --------
        pd.DataFrame
            Normalized matrix with column and row labels
        """
        if self.pairwise_matrix is None:
            raise ValueError("Pairwise matrix not set. Use set_pairwise_matrix() first.")
        
        # Normalize by dividing each element by column sum
        col_sums = self.pairwise_matrix.sum(axis=0)
        self.normalized_matrix = self.pairwise_matrix / col_sums
        
        return pd.DataFrame(
            self.normalized_matrix,
            index=self.cols,
            columns=self.cols
        )
    
    def _calculate_weights(self):
        """Calculate weights from the pairwise matrix (mean of each row)."""
        normalized = self.pairwise_matrix / self.pairwise_matrix.sum(axis=0)
        self.weights = normalized.mean(axis=1)
        self._calculate_consistency_ratio()
    
    def get_weights(self, as_percentage: bool = True) -> pd.DataFrame:
        """
        Get the final weights for each criterion.
        
        Parameters:
        -----------
        as_percentage : bool, default=True
            If True, return weights as percentages
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with criterion names and their weights
        """
        if self.weights is None:
            raise ValueError("Weights not calculated. Use set_pairwise_matrix() first.")
        
        weights_values = self.weights * 100 if as_percentage else self.weights
        
        return pd.DataFrame({
            'Criterion': self.cols,
            'Weight (%)' if as_percentage else 'Weight': weights_values
        }).sort_values(by='Weight (%)' if as_percentage else 'Weight', ascending=False)
    
    def _calculate_consistency_ratio(self):
        """Calculate the Consistency Ratio (CR) to check matrix consistency."""
        n = len(self.cols)
        
        # Random Index (RI) values for different matrix sizes
        ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 
                   7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        
        if n > 10:
            ri = 1.49  # Use max RI for larger matrices
        else:
            ri = ri_dict.get(n, 0)
        
        # Calculate lambda_max
        weighted_sum = self.pairwise_matrix @ self.weights
        lambda_max = np.mean(weighted_sum / self.weights)
        
        # Calculate Consistency Index (CI)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Calculate Consistency Ratio (CR)
        self.consistency_ratio = ci / ri if ri != 0 else 0
    
    def get_consistency_ratio(self) -> float:
        """
        Get the Consistency Ratio (CR).
        CR < 0.1 indicates acceptable consistency.
        
        Returns:
        --------
        float
            Consistency Ratio value
        """
        if self.consistency_ratio is None:
            raise ValueError("Consistency ratio not calculated. Use set_pairwise_matrix() first.")
        return self.consistency_ratio
    
    def normalize_data(self, method: Literal['minmax', 'maxmin'] = 'minmax') -> pd.DataFrame:
        """
        Normalize the data using min-max or max-min normalization.
        
        Parameters:
        -----------
        method : Literal['minmax', 'maxmin'], default='minmax'
            Normalization method:
            - 'minmax': (x - min) / (max - min) - for benefit criteria
            - 'maxmin': (max - x) / (max - min) - for cost criteria
            
        Returns:
        --------
        pd.DataFrame
            Normalized dataframe
        """
        df_normalized = self.df[self.cols].copy()
        
        for col in self.cols:
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            col_range = col_max - col_min
            
            if col_range == 0:
                df_normalized[col] = 0
            else:
                if method == 'minmax':
                    df_normalized[col] = (df_normalized[col] - col_min) / col_range
                elif method == 'maxmin':
                    df_normalized[col] = (col_max - df_normalized[col]) / col_range
                else:
                    raise ValueError("Method must be 'minmax' or 'maxmin'")
        
        return df_normalized
    
    def calculate_scores(self, normalized_df: pd.DataFrame = None) -> pd.Series:
        """
        Calculate final scores by multiplying normalized values with weights.
        
        Parameters:
        -----------
        normalized_df : pd.DataFrame, optional
            Normalized dataframe. If None, will use original df.
            
        Returns:
        --------
        pd.Series
            Final scores for each alternative
        """
        if self.weights is None:
            raise ValueError("Weights not calculated. Use set_pairwise_matrix() first.")
        
        if normalized_df is None:
            normalized_df = self.df[self.cols]
        
        scores = (normalized_df * self.weights).sum(axis=1)
        return scores
    
    def get_solution(self, normalized_df: pd.DataFrame = None, 
                     target_col: str = None) -> pd.DataFrame:
        """
        Get the final solution with scores and ranking.
        
        Parameters:
        -----------
        normalized_df : pd.DataFrame, optional
            Normalized dataframe. If None, will use original df.
        target_col : str, optional
            Target column name to include in the solution
            
        Returns:
        --------
        pd.DataFrame
            Solution dataframe with scores and rankings
        """
        scores = self.calculate_scores(normalized_df)
        
        solution = pd.DataFrame({
            'Score': scores,
            'Rank': scores.rank(ascending=False, method='min').astype(int)
        })
        
        if target_col and target_col in self.df.columns:
            solution.insert(0, target_col, self.df[target_col].values)
        
        return solution.sort_values('Score', ascending=False)
    
    def print_summary(self):
        """Print a summary of the AHP analysis."""
        print("=" * 60)
        print("AHP ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nNumber of criteria: {len(self.cols)}")
        print(f"Criteria: {', '.join(self.cols)}")
        
        if self.weights is not None:
            print("\n" + "-" * 60)
            print("WEIGHTS:")
            print("-" * 60)
            print(self.get_weights(as_percentage=True).to_string(index=False))
            
            print("\n" + "-" * 60)
            print(f"Consistency Ratio (CR): {self.consistency_ratio:.4f}")
            if self.consistency_ratio < 0.1:
                print("Matrix is consistent (CR < 0.1)")
            else:
                print("Matrix is inconsistent (CR >= 0.1) - Review pairwise comparisons")
        
        print("=" * 60)
        
if __name__ == "__main__":
    pass