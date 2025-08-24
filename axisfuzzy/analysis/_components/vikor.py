#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 14:34
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Dict, Union, List, TYPE_CHECKING
import numpy as np

from axisfuzzy.config import get_config

from .base import AnalysisComponent
from ..contracts import contract, FuzzyTable, WeightVector, ScoreVector, RankingResult

if TYPE_CHECKING:
    from ..dataframe import FuzzyDataFrame

from axisfuzzy.utils import experimental


@experimental
class VIKORTool(AnalysisComponent):
    """
    A component for multi-criteria decision analysis using the fuzzy VIKOR method.

    VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje) determines a
    compromise solution from a set of alternatives evaluated against
    conflicting criteria. This implementation is adapted for fuzzy data,
    operating on a `FuzzyDataFrame`.

    Notes
    -----
    The VIKOR method involves the following steps:
    1.  Determine the Fuzzy Positive Ideal Solution (FPIS) and Fuzzy Negative
        Ideal Solution (FNIS) for each criterion. This implementation assumes
        all criteria are benefit criteria (more is better).
    2.  Calculate the group utility (S) and individual regret (R) for each
        alternative based on the weighted and normalized distances to the FPIS.

        .. math::

            S_i = \\sum_{j=1}^{n} w_j \\frac{d(f_j^*, f_{ij})}{d(f_j^*, f_j^-)} \\
            R_i = \\max_{j} \\left( w_j \\frac{d(f_j^*, f_{ij})}{d(f_j^*, f_j^-)} \\right)

    3.  Compute the VIKOR index (Q) for each alternative, which balances the
        group utility and individual regret.

        .. math::

            Q_i = v \\frac{S_i - S^*}{S^- - S^*} + (1-v) \\frac{R_i - R^*}{R^- - R^*}

    4.  Rank the alternatives based on the ascending order of their Q values.
        A lower Q value indicates a better compromise solution.

    Attributes
    ----------
    v : float, default 0.5
        The decision-making mechanism coefficient, representing the weight of
        the strategy of "the majority of criteria" (or "group utility").
        It ranges from 0 to 1. `v > 0.5` prioritizes group utility,
        `v < 0.5` prioritizes individual regret, and `v = 0.5` represents
        a balanced approach.
    epsilon : float, default 1e-9
        A small constant to avoid division by zero when calculating the
        normalized distances, especially if FPIS and FNIS are identical
        for a criterion.
    """
    def __init__(self, v: float = 0.5):
        if not 0 <= v <= 1:
            raise ValueError("The coefficient 'v' must be between 0 and 1.")
        self.v = v
        self.epsilon = get_config().DEFAULT_EPSILON

    @contract(
        inputs={
            'matrix': 'FuzzyTable',
            'weights': 'WeightVector',
            'alternative_names': 'RankingResult'},
        outputs={
            'scores': 'ScoreVector',
            'ranking': 'RankingResult'})
    def run(self,
            matrix: 'FuzzyDataFrame',
            weights: WeightVector,
            alternative_names: List[str]) -> Dict[str, Union[ScoreVector, RankingResult]]:
        """
        Executes the fuzzy VIKOR analysis.

        Parameters
        ----------
        matrix : FuzzyTable
            The fuzzy decision matrix where rows are alternatives and columns
            are criteria. Each cell contains a `Fuzznum`.
        weights : WeightVector
            A 1D array or pandas Series of weights for each criterion. The
            order must match the columns of the matrix.
        alternative_names : RankingResult
            A list of names for the alternatives, corresponding to the rows
            of the matrix.

        Returns
        -------
        dict[str, Union[ScoreVector, RankingResult]]
            A dictionary containing:
            - 'scores': A `numpy.ndarray` of the final Q-scores for each alternative.
            - 'ranking': A list of alternative names sorted by their rank (best to worst).

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.analysis import FuzzyPipeline
            from axisfuzzy.analysis._components.vikor import VIKORTool
            # Assume `fuzzy_decision_matrix` and `criterion_weights` are pre-defined

            # 1. Instantiate the component
            vikor_tool = VIKORTool(v=0.5)

            # 2. Build and run a pipeline
            p = FuzzyPipeline()
            matrix_input = p.input("matrix")
            weights_input = p.input("weights")
            names_input = p.input("names")

            vikor_results = p.add(
                vikor_tool.run,
                matrix=matrix_input,
                weights=weights_input,
                alternative_names=names_input
            )

            results = p.run({
                "matrix": fuzzy_decision_matrix,
                "weights": criterion_weights,
                "names": list(fuzzy_decision_matrix.index)
            })

            print(f"VIKOR Scores: {results['scores']}")
            print(f"Final Ranking: {results['ranking']}")

        """
        num_alternatives, num_criteria = matrix.shape
        if isinstance(weights, list):
            weights = np.array(weights)

        # Step 1: Determine FPIS and FNIS for each criterion
        # Assuming all are benefit criteria
        fpis = [matrix[col].max() for col in matrix.columns]
        fnis = [matrix[col].min() for col in matrix.columns]

        # Calculate normalized distance matrix
        norm_dist_matrix = np.zeros(matrix.shape, dtype=float)

        for j, col_name in enumerate(matrix.columns):
            fuzz_array = matrix[col_name]               # type: ignore

            # d(f_j^*, f_j^-)
            # Assuming the core library provides a distance method on Fuzznum
            dist_ideal_span = fpis[j].distance(fnis[j])

            # d(f_j^*, f_{ij}) for all i
            # Assuming Fuzzarray has a vectorized distance method
            dist_to_fpis = fuzz_array.distance(fpis[j])

            # Normalized distance: w_j * d(f_j^*, f_{ij}) / d(f_j^*, f_j^-)
            norm_dist_matrix[:, j] = weights[j] * (dist_to_fpis / (dist_ideal_span + self.epsilon))

        # Step 2: Calculate S_i (group utility) and R_i (individual regret)
        s_scores = np.sum(norm_dist_matrix, axis=1)
        r_scores = np.max(norm_dist_matrix, axis=1)

        # Step 3: Calculate Q_i (VIKOR index)
        s_star, s_minus = np.min(s_scores), np.max(s_scores)
        r_star, r_minus = np.min(r_scores), np.max(r_scores)

        s_range = s_minus - s_star + self.epsilon
        r_range = r_minus - r_star + self.epsilon

        q_scores = self.v * (s_scores - s_star) / s_range + (1 - self.v) * (r_scores - r_star) / r_range

        # Step 4: Rank alternatives based on Q scores (ascending)
        sorted_indices = np.argsort(q_scores)
        ranked_names = [alternative_names[i] for i in sorted_indices]

        return {'scores': q_scores, 'ranking': ranked_names}
