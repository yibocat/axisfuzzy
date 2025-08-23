#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:37
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import numpy as np

from .base import AnalysisComponent
from ..contracts import contract, FuzzyTable, WeightVector, ScoreVector
from ..dataframe import FuzzyDataFrame


class WeightedAggregationTool(AnalysisComponent):
    """
    Aggregates a fuzzy decision matrix into a crisp score vector using
    a weighted sum approach.
    """
    def __init__(self, scoring_method: str = 'mean'):
        """
        Initializes the aggregation tool.

        Parameters
        ----------
        scoring_method : str, default 'mean'
            The method used to convert a fuzzy number to a crisp score.
            Currently supports 'mean'.
        """
        self.scoring_method = scoring_method

    @contract(
        inputs={'matrix': 'FuzzyTable', 'weights': 'WeightVector'},
        outputs={'scores': 'ScoreVector'}
    )
    def run(self, matrix: FuzzyDataFrame, weights: WeightVector) -> ScoreVector:
        """
        Performs weighted aggregation.

        Parameters
        ----------
        matrix : FuzzyTable
            The fuzzy decision matrix.
        weights : WeightVector
            The weights for each attribute.

        Returns
        -------
        ScoreVector
            A 1D numpy array of crisp scores for each alternative.
        """
        crisp_matrix = np.zeros(matrix.shape, dtype=float)
        for j, col_name in enumerate(matrix.columns):
            fuzz_array = matrix[col_name]       # type: ignore
            scores = fuzz_array.score
            crisp_matrix[:, j] = scores

        # Perform weighted dot product for each alternative (row)
        scores = np.dot(crisp_matrix, weights)
        return scores















