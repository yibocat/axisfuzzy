#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 15:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


from typing import Dict, List, Union
import numpy as np
import pandas as pd

from .base import AnalysisComponent
from ..contracts import contract, ScoreVector, RankingResult, ThreeWayResult


class ClassicThreeWayTool(AnalysisComponent):
    """
    A component for making three-way decisions based on crisp scores and thresholds.

    This tool partitions a set of alternatives into three regions: acceptance,
    rejection, and deferment, based on two predefined thresholds, alpha and beta.

    Notes
    -----
    The decision rules are as follows for an alternative `x` with score `S(x)`:
    1.  **Accept**: If `S(x) >= alpha`, the alternative is accepted.
    2.  **Reject**: If `S(x) <= beta`, the alternative is rejected.
    3.  **Defer**: If `beta < S(x) < alpha`, the decision is deferred.

    Parameters
    ----------
    alpha : float, default 0.7
        The acceptance threshold. Scores greater than or equal to alpha will be
        categorized as 'accept'. Must be greater than beta.
    beta : float, default 0.3
        The rejection threshold. Scores less than or equal to beta will be
        categorized as 'reject'. Must be less than alpha.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        if not (0 <= beta < alpha <= 1):
            raise ValueError("Thresholds must satisfy 0 <= beta < alpha <= 1.")
        self.alpha = alpha
        self.beta = beta

    @contract(
        inputs={'scores': 'ScoreVector', 'alternative_names': 'RankingResult'},
        outputs={'result': 'ThreeWayResult'}
    )
    def run(self,
            scores: ScoreVector,
            alternative_names: List[str]) -> ThreeWayResult:
        """
        Partitions alternatives into accept, reject, and defer regions.

        Parameters
        ----------
        scores : ScoreVector
            A 1D numpy array or pandas Series of scores for each alternative.
        alternative_names : RankingResult
            A list of names for the alternatives, in the same order as the scores.

        Returns
        -------
        ThreeWayResult
            A dictionary with three keys: 'accept', 'reject', 'defer', each
            containing a list of alternative names.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.analysis.components.threeway import ClassicThreeWayTool
            import numpy as np

            scores = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
            names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']

            # 1. Instantiate the component with custom thresholds
            decider = ClassicThreeWayTool(alpha=0.75, beta=0.25)

            # 2. Run the tool
            decision = decider.run(scores=scores, alternative_names=names)

            print(decision)
            # Expected output:
            # {'accept': ['A1', 'A2'], 'defer': ['A3', 'A4'], 'reject': ['A5', 'A6']}
        """
        if isinstance(scores, pd.Series):
            scores = scores.values

        results: ThreeWayResult = {'accept': [], 'reject': [], 'defer': []}

        for i, score in enumerate(scores):
            name = alternative_names[i]
            if score >= self.alpha:
                results['accept'].append(name)
            elif score <= self.beta:
                results['reject'].append(name)
            else:
                results['defer'].append(name)

        return results


class FuzzyThreeWayTool(AnalysisComponent):
    """
    A component for making three-way decisions based on a loss function.

    This tool implements the Decision-Theoretic Rough Sets (DTRS) model.
    Instead of taking explicit thresholds, it derives them from a set of loss
    parameters that represent the costs of making correct or incorrect decisions.
    This provides a more interpretable and business-oriented approach to
    three-way decisions.

    Notes
    -----
    The tool calculates the decision thresholds `alpha` and `beta` based on
    the provided loss function parameters.
    - `lambda_pp`: Loss of taking action `P` (accept) on an object in state `P`.
    - `lambda_pn`: Loss of taking action `P` (accept) on an object in state `N`.
    - `lambda_np`: Loss of taking action `N` (reject) on an object in state `P`.
    - `lambda_nn`: Loss of taking action `N` (reject) on an object in state `N`.
    - `lambda_bp`: Loss of taking action `B` (defer) on an object in state `P`.
    - `lambda_bn`: Loss of taking action `B` (defer) on an object in state `N`.

    The thresholds are calculated as:

    .. math::

        \\alpha = \\frac{\\lambda_{BN} - \\lambda_{PN}}{(\\lambda_{BN} - \\lambda_{PN}) + (\\lambda_{PP} - \\lambda_{BP})} \\
        \\beta = \\frac{\\lambda_{NN} - \\lambda_{BN}}{(\\lambda_{NN} - \\lambda_{BN}) + (\\lambda_{BP} - \\lambda_{NP})}

    The decision rules are then applied identically to `ClassicThreeWayTool`.
    This implementation assumes the input scores are conditional probabilities `P(State P | evidence)`.

    Parameters
    ----------
    loss_function : dict[str, float]
        A dictionary containing the six lambda parameters for the loss function:
        'lambda_pp', 'lambda_pn', 'lambda_np', 'lambda_nn', 'lambda_bp', 'lambda_bn'.
    """
    def __init__(self, loss_function: Dict[str, float]):
        self._validate_loss_function(loss_function)
        self.loss_function = loss_function
        self.alpha, self.beta = self._calculate_thresholds()

        # We reuse ClassicThreeWayTool for the final partitioning logic
        self._classic_decider = ClassicThreeWayTool(self.alpha, self.beta)

    def _validate_loss_function(self, lf: Dict[str, float]):
        """Validates the structure and logical constraints of the loss function."""
        required_keys = {'lambda_pp', 'lambda_pn', 'lambda_np', 'lambda_nn', 'lambda_bp', 'lambda_bn'}
        if not required_keys.issubset(lf.keys()):
            missing = required_keys - set(lf.keys())
            raise ValueError(f"Loss function is missing required keys: {missing}")

        # Check DTRS model's rationality conditions for a non-empty defer region
        if not (lf['lambda_pn'] > lf['lambda_bn'] and lf['lambda_np'] > lf['lambda_bp']):
             print("Warning: Loss function values do not guarantee a non-empty deferment region. "
                   "This may lead to a two-way decision scenario (alpha <= beta).")

    def _calculate_thresholds(self) -> tuple[float, float]:
        """Calculates alpha and beta thresholds from the loss function."""
        lf = self.loss_function

        # Alpha calculation
        alpha_num = lf['lambda_bn'] - lf['lambda_pn']
        alpha_den = (lf['lambda_bn'] - lf['lambda_pn']) + (lf['lambda_pp'] - lf['lambda_bp'])

        if alpha_den == 0:
            # This is a degenerate case, implies R(a_P|x) and R(a_B|x) are parallel.
            # Decision depends only on the sign of the numerator.
            alpha = float('inf') if alpha_num > 0 else float('-inf')
        else:
            alpha = alpha_num / alpha_den

        # Beta calculation
        beta_num = lf['lambda_nn'] - lf['lambda_bn']
        beta_den = (lf['lambda_nn'] - lf['lambda_bn']) + (lf['lambda_bp'] - lf['lambda_np'])

        if beta_den == 0:
            beta = float('inf') if beta_num > 0 else float('-inf')
        else:
            beta = beta_num / beta_den

        return alpha, beta

    @contract(
        inputs={'scores': 'ScoreVector', 'alternative_names': 'RankingResult'},
        outputs={'result': 'ThreeWayResult'}
    )
    def run(self,
            scores: ScoreVector,
            alternative_names: List[str]) -> ThreeWayResult:
        """
        Partitions alternatives using DTRS-derived thresholds.

        Parameters
        ----------
        scores : ScoreVector
            A 1D numpy array or pandas Series of scores, interpreted as
            conditional probabilities P(State P | evidence).
        alternative_names : RankingResult
            A list of names for the alternatives, in the same order as the scores.

        Returns
        -------
        ThreeWayResult
            A dictionary with three keys: 'accept', 'reject', 'defer', each
            containing a list of alternative names.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.analysis.components.threeway import FuzzyThreeWayTool
            import numpy as np

            # Define a loss function for a loan application scenario
            # High cost for wrong approval, moderate cost for wrong rejection
            loss_func = {
                'lambda_pp': 0,    # Correctly approving a good loan (no loss)
                'lambda_pn': 100,  # Incorrectly approving a bad loan (high loss)
                'lambda_np': 20,   # Incorrectly rejecting a good loan (opportunity loss)
                'lambda_nn': 0,    # Correctly rejecting a bad loan (no loss)
                'lambda_bp': 5,    # Cost to investigate a good loan
                'lambda_bn': 5,    # Cost to investigate a bad loan
            }

            # 1. Instantiate the component
            dtrs_decider = FuzzyThreeWayTool(loss_function=loss_func)

            # The tool internally calculates alpha and beta:
            # alpha = (5 - 100) / ((5 - 100) + (0 - 5)) = -95 / -100 = 0.95
            # beta = (0 - 5) / ((0 - 5) + (5 - 20)) = -5 / -20 = 0.25

            print(f"Derived Alpha: {dtrs_decider.alpha:.2f}")
            print(f"Derived Beta: {dtrs_decider.beta:.2f}")

            # 2. Run the tool with some probability scores
            # (e.g., from a machine learning model predicting default probability)
            prob_scores = np.array([0.98, 0.95, 0.75, 0.30, 0.25, 0.10])
            names = ['App1', 'App2', 'App3', 'App4', 'App5', 'App6']

            decision = dtrs_decider.run(scores=prob_scores, alternative_names=names)

            print(decision)
            # Expected output:
            # {'accept': ['App1', 'App2'], 'defer': ['App3', 'App4'], 'reject': ['App5', 'App6']}
        """
        # The core logic is delegated to the internal ClassicThreeWayTool instance,
        # which already has the correctly calculated alpha and beta.
        return self._classic_decider.run(scores, alternative_names)
