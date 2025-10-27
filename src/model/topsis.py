import numpy as np
import pandas as pd
from typing import List, Literal

class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) 
    class for multi-criteria decision making.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        cols (List[str]): List of column names to be used in TOPSIS analysis
        weights (List[float], optional): Weights for each criterion (must sum to 1)
        impacts (List[str], optional): '+' for benefit, '-' for cost criterion
    """
    
    def __init__(self, df: pd.DataFrame, cols: List[str], 
                 weights: List[float] = None, impacts: List[str] = None):
        self.df = df.copy()
        self.cols = cols
        self.n_criteria = len(cols)
        
        # Set default weights (equal weights)
        if weights is None:
            self.weights = np.array([1/self.n_criteria] * self.n_criteria)
        else:
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = np.array(weights)
        
        # Set default impacts (all benefit)
        if impacts is None:
            self.impacts = ['+'] * self.n_criteria
        else:
            if len(impacts) != self.n_criteria:
                raise ValueError(f"Number of impacts must match number of criteria ({self.n_criteria})")
            self.impacts = impacts
        
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ideal_best = None
        self.ideal_worst = None
        self.scores = None
        self.rankings = None
        
    def set_weights(self, weights: List[float]):
        """
        Set weights for criteria.
        
        Args:
            weights (List[float]): List of weights (must sum to 1)
        """
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
        self.weights = np.array(weights)
        
    def set_impacts(self, impacts: List[str]):
        """
        Set impacts for criteria.
        
        Args:
            impacts (List[str]): List of '+' (benefit) or '-' (cost)
        """
        if len(impacts) != self.n_criteria:
            raise ValueError(f"Number of impacts must match number of criteria ({self.n_criteria})")
        self.impacts = impacts
        
    def normalize(self, method: Literal['vector', 'minmax'] = 'vector') -> pd.DataFrame:
        """
        Normalize the decision matrix.
        
        Args:
            method (str): 'vector' for vector normalization, 'minmax' for min-max normalization
            
        Returns:
            pd.DataFrame: Normalized matrix
        """
        matrix = self.df[self.cols].values
        
        if method == 'vector':
            # Vector normalization (standard TOPSIS)
            norm = np.sqrt(np.sum(matrix**2, axis=0))
            self.normalized_matrix = matrix / norm
        elif method == 'minmax':
            # Min-max normalization
            min_vals = matrix.min(axis=0)
            max_vals = matrix.max(axis=0)
            self.normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
        else:
            raise ValueError("Method must be 'vector' or 'minmax'")
        
        return pd.DataFrame(self.normalized_matrix, columns=self.cols)
    
    def apply_weights(self) -> pd.DataFrame:
        """
        Apply weights to normalized matrix.
        
        Returns:
            pd.DataFrame: Weighted normalized matrix
        """
        if self.normalized_matrix is None:
            self.normalize()
        
        self.weighted_matrix = self.normalized_matrix * self.weights
        return pd.DataFrame(self.weighted_matrix, columns=self.cols)
    
    def calculate_ideal_solutions(self):
        """
        Calculate ideal best and ideal worst solutions.
        """
        if self.weighted_matrix is None:
            self.apply_weights()
        
        self.ideal_best = np.zeros(self.n_criteria)
        self.ideal_worst = np.zeros(self.n_criteria)
        
        for i, impact in enumerate(self.impacts):
            if impact == '+':
                # Benefit criterion
                self.ideal_best[i] = self.weighted_matrix[:, i].max()
                self.ideal_worst[i] = self.weighted_matrix[:, i].min()
            else:
                # Cost criterion
                self.ideal_best[i] = self.weighted_matrix[:, i].min()
                self.ideal_worst[i] = self.weighted_matrix[:, i].max()
    
    def calculate_distances(self) -> tuple:
        """
        Calculate distances from ideal best and ideal worst solutions.
        
        Returns:
            tuple: (distance_to_best, distance_to_worst)
        """
        if self.ideal_best is None or self.ideal_worst is None:
            self.calculate_ideal_solutions()
        
        # Euclidean distance
        distance_to_best = np.sqrt(np.sum((self.weighted_matrix - self.ideal_best)**2, axis=1))
        distance_to_worst = np.sqrt(np.sum((self.weighted_matrix - self.ideal_worst)**2, axis=1))
        
        return distance_to_best, distance_to_worst
    
    def calculate_scores(self) -> pd.Series:
        """
        Calculate TOPSIS scores (closeness coefficient).
        
        Returns:
            pd.Series: TOPSIS scores for each alternative
        """
        dist_best, dist_worst = self.calculate_distances()
        
        # Closeness coefficient
        self.scores = dist_worst / (dist_best + dist_worst)
        
        return pd.Series(self.scores, name='TOPSIS_Score')
    
    def get_rankings(self) -> pd.DataFrame:
        """
        Get rankings based on TOPSIS scores.
        
        Returns:
            pd.DataFrame: Rankings with scores
        """
        if self.scores is None:
            self.calculate_scores()
        
        rankings_df = pd.DataFrame({
            'TOPSIS_Score': self.scores,
            'Rank': pd.Series(self.scores).rank(ascending=False, method='dense').astype(int)
        })
        
        self.rankings = rankings_df.sort_values('Rank')
        return self.rankings
    
    def get_solution(self, target_col: str = None) -> pd.DataFrame:
        """
        Get complete solution with original data, scores, and rankings.
        
        Args:
            target_col (str, optional): Target column to include in results
            
        Returns:
            pd.DataFrame: Complete solution dataframe
        """
        if self.scores is None:
            self.calculate_scores()
        
        result_df = self.df.copy()
        result_df['TOPSIS_Score'] = self.scores
        result_df['Rank'] = pd.Series(self.scores).rank(ascending=False, method='dense').astype(int)
        
        if target_col and target_col in self.df.columns:
            cols_order = [target_col] + self.cols + ['TOPSIS_Score', 'Rank']
            result_df = result_df[cols_order]
        else:
            cols_order = self.cols + ['TOPSIS_Score', 'Rank']
            result_df = result_df[cols_order]
        
        return result_df.sort_values('Rank')
    
    def get_normalized_matrix(self) -> pd.DataFrame:
        """
        Get the normalized decision matrix.
        
        Returns:
            pd.DataFrame: Normalized matrix
        """
        if self.normalized_matrix is None:
            self.normalize()
        return pd.DataFrame(self.normalized_matrix, columns=self.cols)
    
    def get_weighted_matrix(self) -> pd.DataFrame:
        """
        Get the weighted normalized matrix.
        
        Returns:
            pd.DataFrame: Weighted normalized matrix
        """
        if self.weighted_matrix is None:
            self.apply_weights()
        return pd.DataFrame(self.weighted_matrix, columns=self.cols)
    
    def get_ideal_solutions(self) -> pd.DataFrame:
        """
        Get ideal best and worst solutions.
        
        Returns:
            pd.DataFrame: DataFrame with ideal solutions
        """
        if self.ideal_best is None or self.ideal_worst is None:
            self.calculate_ideal_solutions()
        
        return pd.DataFrame({
            'Criterion': self.cols,
            'Weight': self.weights,
            'Impact': self.impacts,
            'Ideal_Best': self.ideal_best,
            'Ideal_Worst': self.ideal_worst
        })
    
    def print_summary(self):
        """
        Print comprehensive summary of TOPSIS analysis.
        """
        print("=" * 80)
        print("TOPSIS ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Criteria Information:")
        print(f"   Number of criteria: {self.n_criteria}")
        print(f"   Criteria: {', '.join(self.cols)}")
        
        print(f"\n⚖️  Weights and Impacts:")
        weights_df = pd.DataFrame({
            'Criterion': self.cols,
            'Weight': self.weights,
            'Impact': self.impacts
        })
        print(weights_df.to_string(index=False))
        
        print(f"\nIdeal Solutions:")
        ideal_df = self.get_ideal_solutions()
        print(ideal_df.to_string(index=False))
        
        print(f"\nTop 5 Rankings:")
        rankings = self.get_rankings().head()
        print(rankings.to_string())
        
        print("\n" + "=" * 80)