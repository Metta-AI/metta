"""
Correlation analyzer for policy performance data.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Analyzer for computing and analyzing correlations between task performances."""

    def __init__(self):
        self.correlation_matrix = None

    def calculate_correlations(self, performance_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix from performance matrix."""
        # Handle missing values by filling with mean
        performance_matrix_filled = performance_matrix.fillna(performance_matrix.mean())

        # Calculate correlation matrix
        self.correlation_matrix = performance_matrix_filled.corr()

        return self.correlation_matrix

    def get_average_correlation(self) -> float:
        """Get average correlation coefficient."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Call calculate_correlations first.")

        # Get upper triangle values (excluding diagonal)
        upper_triangle = self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)]
        return np.mean(upper_triangle)

    def get_correlation_range(self) -> Tuple[float, float]:
        """Get range of correlation coefficients."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Call calculate_correlations first.")

        # Get upper triangle values (excluding diagonal)
        upper_triangle = self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)]
        return float(np.min(upper_triangle)), float(np.max(upper_triangle))

    def get_most_correlated_pairs(self, n: int = 3) -> List[Tuple[Tuple[str, str], float]]:
        """Get the n most correlated task pairs."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Call calculate_correlations first.")

        # Get all pairs and their correlations
        pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                task1 = self.correlation_matrix.columns[i]
                task2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                pairs.append(((task1, task2), corr))

        # Sort by absolute correlation value and return top n
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:n]

    def get_least_correlated_pairs(self, n: int = 3) -> List[Tuple[Tuple[str, str], float]]:
        """Get the n least correlated task pairs."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Call calculate_correlations first.")

        # Get all pairs and their correlations
        pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                task1 = self.correlation_matrix.columns[i]
                task2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                pairs.append(((task1, task2), corr))

        # Sort by absolute correlation value and return bottom n
        pairs.sort(key=lambda x: abs(x[1]))
        return pairs[:n]

    def get_task_clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """Get clusters of highly correlated tasks."""
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Call calculate_correlations first.")

        # Create adjacency matrix for tasks with correlation above threshold
        adjacency = (self.correlation_matrix.abs() > threshold).astype(int)
        np.fill_diagonal(adjacency.values, 0)  # Remove self-connections

        # Simple clustering: find connected components
        clusters = []
        visited = set()

        for task in self.correlation_matrix.columns:
            if task in visited:
                continue

            # Start new cluster
            cluster = [task]
            visited.add(task)

            # Find all connected tasks
            to_visit = [task]
            while to_visit:
                current = to_visit.pop()
                for other_task in self.correlation_matrix.columns:
                    if other_task not in visited and adjacency.loc[current, other_task] == 1:
                        cluster.append(other_task)
                        visited.add(other_task)
                        to_visit.append(other_task)

            if len(cluster) > 1:  # Only include clusters with multiple tasks
                clusters.append(cluster)

        return clusters
