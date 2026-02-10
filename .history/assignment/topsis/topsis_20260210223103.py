"""
TOPSIS Implementation from Scratch

TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
is a multi-criteria decision analysis method that ranks alternatives based
on their geometric distance from ideal best and worst solutions.

This implementation does NOT use any external TOPSIS library - all algorithms
are implemented from scratch using only NumPy for matrix operations.

Author: ML Research Assistant
Date: February 10, 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    
    A multi-criteria decision-making method that ranks alternatives by measuring
    their geometric distance from the ideal best and ideal worst solutions.
    
    Mathematical Steps:
    1. Normalize the decision matrix using vector normalization
    2. Apply weights to normalized values
    3. Determine ideal best (A+) and ideal worst (A-) solutions
    4. Calculate Euclidean distances from ideal solutions
    5. Calculate relative closeness coefficient (TOPSIS score)
    6. Rank alternatives in descending order of TOPSIS score
    """
    
    def __init__(
        self,
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        criteria_types: List[str],
        alternatives: List[str] = None
    ):
        """
        Initialize TOPSIS with decision matrix and parameters.
        
        Args:
            decision_matrix (np.ndarray): m×n matrix where m=alternatives, n=criteria
            weights (np.ndarray): 1D array of criterion weights (must sum to 1)
            criteria_types (List[str]): '+' for benefit, '-' for cost criteria
            alternatives (List[str]): Names of alternatives (optional)
        
        Raises:
            ValueError: If weights don't sum to 1 or dimensions mismatch
        """
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.criteria_types = criteria_types
        
        # Validate inputs
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1. Current sum: {self.weights.sum()}")
        
        if len(self.weights) != self.decision_matrix.shape[1]:
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of criteria ({self.decision_matrix.shape[1]})"
            )
        
        if len(self.criteria_types) != self.decision_matrix.shape[1]:
            raise ValueError(
                f"Number of criteria types ({len(self.criteria_types)}) must match "
                f"number of criteria ({self.decision_matrix.shape[1]})"
            )
        
        # Set alternative names
        if alternatives is None:
            self.alternatives = [f"Alt_{i+1}" for i in range(self.decision_matrix.shape[0])]
        else:
            self.alternatives = alternatives
        
        # Placeholders for intermediate results
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ideal_best = None
        self.ideal_worst = None
        self.distance_to_best = None
        self.distance_to_worst = None
        self.topsis_scores = None
        self.rankings = None
    
    def normalize(self) -> np.ndarray:
        """
        Step 1: Normalize the decision matrix using vector normalization.
        
        Formula: r_ij = x_ij / sqrt(sum(x_ij^2))
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        # Calculate denominator: sqrt of sum of squares for each column
        denominators = np.sqrt(np.sum(self.decision_matrix**2, axis=0))
        
        # Avoid division by zero
        denominators = np.where(denominators == 0, 1, denominators)
        
        # Normalize
        self.normalized_matrix = self.decision_matrix / denominators
        
        return self.normalized_matrix
    
    def apply_weights(self) -> np.ndarray:
        """
        Step 2: Calculate weighted normalized decision matrix.
        
        Formula: v_ij = w_j * r_ij
        
        Returns:
            np.ndarray: Weighted normalized matrix
        """
        if self.normalized_matrix is None:
            self.normalize()
        
        # Multiply each column by its weight
        self.weighted_matrix = self.normalized_matrix * self.weights
        
        return self.weighted_matrix
    
    def determine_ideal_solutions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Determine ideal best (A+) and ideal worst (A-) solutions.
        
        For benefit criteria (+): 
            A+ = max value, A- = min value
        For cost criteria (-):
            A+ = min value, A- = max value
        
        Returns:
            tuple: (ideal_best, ideal_worst) arrays
        """
        if self.weighted_matrix is None:
            self.apply_weights()
        
        self.ideal_best = np.zeros(self.weighted_matrix.shape[1])
        self.ideal_worst = np.zeros(self.weighted_matrix.shape[1])
        
        for j, criterion_type in enumerate(self.criteria_types):
            if criterion_type == '+':  # Benefit criterion
                self.ideal_best[j] = np.max(self.weighted_matrix[:, j])
                self.ideal_worst[j] = np.min(self.weighted_matrix[:, j])
            elif criterion_type == '-':  # Cost criterion
                self.ideal_best[j] = np.min(self.weighted_matrix[:, j])
                self.ideal_worst[j] = np.max(self.weighted_matrix[:, j])
            else:
                raise ValueError(f"Invalid criterion type: {criterion_type}. Use '+' or '-'")
        
        return self.ideal_best, self.ideal_worst
    
    def calculate_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4: Calculate Euclidean distances from ideal solutions.
        
        Formula for distance to best:
            S_i+ = sqrt(sum((v_ij - v_j+)^2))
        
        Formula for distance to worst:
            S_i- = sqrt(sum((v_ij - v_j-)^2))
        
        Returns:
            tuple: (distance_to_best, distance_to_worst) arrays
        """
        if self.ideal_best is None or self.ideal_worst is None:
            self.determine_ideal_solutions()
        
        # Calculate squared differences
        diff_to_best = (self.weighted_matrix - self.ideal_best)**2
        diff_to_worst = (self.weighted_matrix - self.ideal_worst)**2
        
        # Sum across criteria (axis=1) and take square root
        self.distance_to_best = np.sqrt(np.sum(diff_to_best, axis=1))
        self.distance_to_worst = np.sqrt(np.sum(diff_to_worst, axis=1))
        
        return self.distance_to_best, self.distance_to_worst
    
    def calculate_topsis_scores(self) -> np.ndarray:
        """
        Step 5: Calculate relative closeness coefficient (TOPSIS score).
        
        Formula: C_i = S_i- / (S_i+ + S_i-)
        
        Where:
            - C_i ∈ [0, 1]
            - C_i = 1 means identical to ideal best
            - C_i = 0 means identical to ideal worst
            - Higher C_i indicates better performance
        
        Returns:
            np.ndarray: TOPSIS scores for all alternatives
        """
        if self.distance_to_best is None or self.distance_to_worst is None:
            self.calculate_distances()
        
        # Calculate relative closeness
        denominator = self.distance_to_best + self.distance_to_worst
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        self.topsis_scores = self.distance_to_worst / denominator
        
        return self.topsis_scores
    
    def rank_alternatives(self) -> pd.DataFrame:
        """
        Step 6: Rank alternatives in descending order of TOPSIS score.
        
        Returns:
            pd.DataFrame: Ranked alternatives with scores
        """
        if self.topsis_scores is None:
            self.calculate_topsis_scores()
        
        # Create ranking DataFrame
        results = pd.DataFrame({
            'Alternative': self.alternatives,
            'TOPSIS Score': self.topsis_scores
        })
        
        # Sort by TOPSIS score (descending)
        results = results.sort_values('TOPSIS Score', ascending=False)
        results['Rank'] = range(1, len(results) + 1)
        
        # Reorder columns
        results = results[['Rank', 'Alternative', 'TOPSIS Score']]
        
        self.rankings = results.reset_index(drop=True)
        
        return self.rankings
    
    def get_detailed_results(self) -> Dict:
        """
        Get detailed intermediate results for analysis.
        
        Returns:
            dict: All intermediate matrices and final results
        """
        if self.rankings is None:
            self.rank_alternatives()
        
        return {
            'decision_matrix': self.decision_matrix,
            'normalized_matrix': self.normalized_matrix,
            'weighted_matrix': self.weighted_matrix,
            'ideal_best': self.ideal_best,
            'ideal_worst': self.ideal_worst,
            'distance_to_best': self.distance_to_best,
            'distance_to_worst': self.distance_to_worst,
            'topsis_scores': self.topsis_scores,
            'rankings': self.rankings
        }
    
    def print_summary(self):
        """Print a formatted summary of TOPSIS results."""
        if self.rankings is None:
            self.rank_alternatives()
        
        print("\n" + "="*60)
        print("TOPSIS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nNumber of Alternatives: {len(self.alternatives)}")
        print(f"Number of Criteria: {len(self.weights)}")
        
        print("\nCriteria Weights:")
        for i, weight in enumerate(self.weights):
            criterion_type = "Benefit" if self.criteria_types[i] == '+' else "Cost"
            print(f"  Criterion {i+1}: {weight:.4f} ({criterion_type})")
        
        print("\nIdeal Best Solution:")
        print(f"  {self.ideal_best}")
        
        print("\nIdeal Worst Solution:")
        print(f"  {self.ideal_worst}")
        
        print("\n" + "-"*60)
        print("FINAL RANKINGS")
        print("-"*60)
        print(self.rankings.to_string(index=False))
        print("="*60 + "\n")


def apply_topsis_to_dataframe(
    df: pd.DataFrame,
    criteria_columns: List[str],
    weights: Dict[str, float],
    criteria_types: Dict[str, str],
    alternative_column: str = None
) -> pd.DataFrame:
    """
    Convenience function to apply TOPSIS to a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        criteria_columns (List[str]): Column names representing criteria
        weights (Dict[str, float]): Mapping of criterion to weight
        criteria_types (Dict[str, str]): Mapping of criterion to type ('+' or '-')
        alternative_column (str): Column name for alternative labels
        
    Returns:
        pd.DataFrame: Original DataFrame with added TOPSIS Score and Rank columns
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Model': ['A', 'B', 'C'],
        ...     'ROUGE-1': [0.45, 0.42, 0.48],
        ...     'Latency': [300, 200, 400]
        ... })
        >>> weights = {'ROUGE-1': 0.7, 'Latency': 0.3}
        >>> types = {'ROUGE-1': '+', 'Latency': '-'}
        >>> result = apply_topsis_to_dataframe(df, ['ROUGE-1', 'Latency'], weights, types, 'Model')
    """
    # Extract decision matrix
    decision_matrix = df[criteria_columns].values
    
    # Extract weights and types in correct order
    weights_array = np.array([weights[col] for col in criteria_columns])
    types_list = [criteria_types[col] for col in criteria_columns]
    
    # Extract alternative names
    if alternative_column:
        alternatives = df[alternative_column].tolist()
    else:
        alternatives = None
    
    # Run TOPSIS
    topsis = TOPSIS(decision_matrix, weights_array, types_list, alternatives)
    rankings = topsis.rank_alternatives()
    
    # Merge results back to original DataFrame
    result_df = df.copy()
    result_df['TOPSIS Score'] = topsis.topsis_scores
    
    # Add rank based on TOPSIS score
    result_df = result_df.sort_values('TOPSIS Score', ascending=False)
    result_df['Rank'] = range(1, len(result_df) + 1)
    
    return result_df


def main():
    """Example usage of TOPSIS class."""
    # Example: Laptop selection
    print("Example: TOPSIS for Laptop Selection\n")
    
    # Decision matrix: [Price, Performance, Battery Life, Weight]
    laptops = np.array([
        [1200, 85, 8,  1.5],  # Laptop A
        [1500, 92, 6,  1.8],  # Laptop B
        [1000, 75, 10, 1.3],  # Laptop C
        [1800, 95, 7,  2.0],  # Laptop D
    ])
    
    # Weights (must sum to 1)
    weights = np.array([0.3, 0.4, 0.2, 0.1])
    
    # Criteria types: Price(-), Performance(+), Battery(+), Weight(-)
    criteria_types = ['-', '+', '+', '-']
    
    # Alternative names
    alternatives = ['Laptop A', 'Laptop B', 'Laptop C', 'Laptop D']
    
    # Run TOPSIS
    topsis = TOPSIS(laptops, weights, criteria_types, alternatives)
    rankings = topsis.rank_alternatives()
    topsis.print_summary()


if __name__ == "__main__":
    main()
