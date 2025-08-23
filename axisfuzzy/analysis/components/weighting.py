#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:33
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import TYPE_CHECKING

import numpy as np

from .base import AnalysisComponent
from ..contracts import contract, FuzzyTable, WeightVector

from ..dataframe import FuzzyDataFrame


class EntropyWeightTool(AnalysisComponent):
    """
    Calculates attribute weights using the Entropy Weight Method (EWM).

    This method determines objective weights based on the amount of information
    (or entropy) present in each attribute column of a decision matrix.
    Higher entropy (less variance) implies less information and thus a lower weight.
    """
    def __init__(self, epsilon: float = 1e-12):
        """
        Initializes the tool.

        Parameters
        ----------
        epsilon : float, default 1e-12
            A small value to add to prevent log(0) errors during entropy calculation.
        """
        self.epsilon = epsilon

    @contract(inputs={'matrix': 'FuzzyTable'}, outputs={'weights': 'WeightVector'})
    def run(self, matrix: FuzzyDataFrame) -> WeightVector:
        """
        Computes the entropy weights for the given fuzzy decision matrix.

        Parameters
        ----------
        matrix : FuzzyTable
            The fuzzy decision matrix (alternatives x attributes).

        Returns
        -------
        WeightVector
            A 1D numpy array containing the calculated weights for each attribute.
        """
        # Step 1: Defuzzify the matrix to a crisp representation.
        # This involves converting each Fuzznum in the FuzzyTable to a single crisp score.
        crisp_matrix = np.zeros(matrix.shape, dtype=float)
        for j, col_name in enumerate(matrix.columns):
            fuzz_array = matrix[col_name] # This is a Fuzzarray (collection of Fuzznum)
            scores = np.zeros(len(fuzz_array), dtype=float)
            for i in range(len(fuzz_array)):
                fuzz_num = fuzz_array[i] # This is an individual Fuzznum

                # [Crucial Point] Attempt to get a crisp score from the Fuzznum.
                # In a complete system, Fuzznum objects would have a 'score()' method
                # (likely added via the Extension system) that returns a float.
                if hasattr(fuzz_num, 'score'): # Preferred: a dedicated scoring function for Fuzznum
                    scores[i] = fuzz_num.score
                elif hasattr(fuzz_num, 'mean'): # Alternative: if Fuzznum has a crisp mean method
                    scores[i] = fuzz_num.mean()
                else:
                    # Fallback for testing: a simple, basic score if no specific method is available.
                    # This ensures the code runs even if the 'score' extension is not yet implemented.
                    scores[i] = (getattr(fuzz_num, 'md', 0.0) - getattr(fuzz_num, 'nmd', 0.0) + 1) / 2

            crisp_matrix[:, j] = scores

        # Step 2: Normalize the crisp matrix column-wise (with non-negativity handling).
        # Ensures all values are non-negative for entropy calculation.
        if np.any(crisp_matrix < 0):
             crisp_matrix = (crisp_matrix - crisp_matrix.min()) / (crisp_matrix.max() - crisp_matrix.min() + self.epsilon)

        col_sums = crisp_matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1 # Avoid division by zero for columns that sum to zero
        p_matrix = crisp_matrix / col_sums

        # Step 3: Calculate the entropy for each attribute.
        num_alternatives = p_matrix.shape[0]
        if num_alternatives <= 1: # Handle edge case for small datasets
            return np.full(p_matrix.shape[1], 1 / p_matrix.shape[1])

        k = -1 / np.log(num_alternatives)
        entropy = k * np.sum(p_matrix * np.log(p_matrix + self.epsilon), axis=0)

        # Step 4: Calculate the degree of divergence (information content).
        divergence = 1 - entropy

        # Step 5: Normalize the divergence to get the final weights.
        total_divergence = divergence.sum()
        if total_divergence == 0: # Handle case where all divergences are zero
            return np.full(len(divergence), 1 / len(divergence)) # All equal if no info

        weights = divergence / total_divergence
        return weights
