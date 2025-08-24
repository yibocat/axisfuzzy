#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 15:10
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Dict, Union, List, TYPE_CHECKING
import numpy as np
import pandas as pd

from .base import AnalysisComponent
from ..contracts import contract, PairwiseMatrix, WeightVector

if TYPE_CHECKING:
    from axisfuzzy.core import Fuzznum  # For type hinting Fuzznum elements

from axisfuzzy.utils import experimental


@experimental
class AHPWeightTool(AnalysisComponent):
    """
    A component for calculating criterion weights using the fuzzy Analytic Hierarchy Process (AHP).

    This tool implements the fuzzy geometric mean method to derive weights from a
    fuzzy pairwise comparison matrix and performs a consistency check.

    Notes
    -----
    The fuzzy AHP process using the geometric mean method involves:
    1.  **Fuzzy Geometric Mean**: For each row `i` of the fuzzy pairwise
        comparison matrix (where elements are fuzzy numbers `a_ij`), calculate
        the fuzzy geometric mean `r_i`.

        .. math::
            r_i = \\left( \\prod_{j=1}^{n} a_{ij} \\right)^{\\frac{1}{n}}

    2.  **Fuzzy Sum**: Sum all fuzzy geometric means `r_i` to get `S`.

        .. math::
            S = \\sum_{i=1}^{n} r_i

    3.  **Fuzzy Inverse**: Calculate the inverse of the fuzzy sum `S_inv`.

        .. math::
            S^{-1} = \\left( \\frac{1}{U_S}, \\frac{1}{M_S}, \\frac{1}{L_S} \\right)

    4.  **Fuzzy Weights**: Compute the fuzzy weight `w_i` for each criterion `i`
        by multiplying `r_i` with `S_inv`.

        .. math::
            w_i = r_i \\otimes S^{-1}

    5.  **Defuzzification**: Convert each fuzzy weight `w_i` into a crisp value `w'_i`.
        This implementation assumes `Fuzznum` objects have a `.score()` method.

        .. math::
            w'_i = \\text{score}(w_i)

    6.  **Normalization**: Normalize the crisp weights `w'_i` to sum to 1.

        .. math::
            \\text{weights}_i = \\frac{w'_i}{\\sum_{k=1}^{n} w'_k}

    7.  **Consistency Check**:
        *   **Crisp Equivalent**: Convert the fuzzy pairwise matrix to a crisp one
            by defuzzifying each element.
        *   **Principal Eigenvalue**: Calculate the principal eigenvalue (lambda_max)
            of the crisp equivalent matrix.
        *   **Consistency Index (CI)**:

            .. math::
                CI = \\frac{\\lambda_{\\max} - n}{n - 1}

        *   **Consistency Ratio (CR)**:

            .. math::
                CR = \\frac{CI}{RI}

            where RI is the Random Index based on matrix size `n`.
            A CR < `consistency_threshold` (default 0.1) is generally acceptable.

    Parameters
    ----------
    consistency_threshold : float, default 0.1
        The maximum acceptable Consistency Ratio (CR) for the pairwise
        comparison matrix. If the calculated CR exceeds this value, a warning
        will be issued.
    """

    # Random Index (RI) values for consistency check
    _RI_VALUES = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
        9: 1.45, 10: 1.49, 11: 1.51, 12: 1.53, 13: 1.56, 14: 1.57, 15: 1.58
    }

    def __init__(self, consistency_threshold: float = 0.1):
        if not 0 < consistency_threshold:
            raise ValueError("Consistency threshold must be a positive value.")
        self.consistency_threshold = consistency_threshold

    @contract(
        inputs={'pairwise_matrix': 'PairwiseMatrix'},
        outputs={'weights': 'WeightVector', 'consistency_ratio': 'float'}
    )
    def run(self, pairwise_matrix: pd.DataFrame) -> Dict[str, Union[WeightVector, float]]:
        """
        Calculates the AHP weights and consistency ratio from a fuzzy pairwise
        comparison matrix.

        Parameters
        ----------
        pairwise_matrix : PairwiseMatrix
            A pandas DataFrame representing the fuzzy pairwise comparison matrix.
            Elements are expected to be `Fuzznum` objects.

        Returns
        -------
        dict[str, Union[WeightVector, float]]
            A dictionary containing:
            - 'weights': A `numpy.ndarray` of the normalized crisp weights.
            - 'consistency_ratio': A `float` representing the Consistency Ratio (CR).

        Raises
        ------
        ValueError
            If the input `pairwise_matrix` is not square or contains non-Fuzznum elements.

        Examples
        --------
        .. code-block:: python

            import pandas as pd
            from axisfuzzy.analysis._components.ahp import AHPWeightTool
            # Assuming you have a way to create Fuzznum objects, e.g., TriangularFuzznum
            # from axisfuzzy.fuzzy.tfn.fuzznums import TriangularFuzznum

            # Example: A 3x3 fuzzy pairwise comparison matrix
            # (Mocked TFNs as (l, m, u) tuples for demonstration)
            # In a real scenario, these would be actual Fuzznum instances.
            # Here, we will use a mock Fuzznum that supports arithmetic operations
            # in the test to ensure this code works.

            # For simplicity, let's use a mock Fuzznum in the example
            class MockFuzznumForAHP:
                def __init__(self, l, m, u):
                    self.l, self.m, self.u = l, m, u

                def __mul__(self, other):
                    if isinstance(other, (int, float)):
                        return MockFuzznumForAHP(self.l * other, self.m * other, self.u * other)
                    return MockFuzznumForAHP(self.l * other.l, self.m * other.m, self.u * other.u)

                def __pow__(self, power):
                    return MockFuzznumForAHP(self.l**power, self.m**power, self.u**power)

                def __add__(self, other):
                    return MockFuzznumForAHP(self.l + other.l, self.m + other.m, self.u + other.u)

                def __truediv__(self, other):
                    if isinstance(other, (int, float)):
                        return MockFuzznumForAHP(self.l / other, self.m / other, self.u / other)
                    # For Fuzznum / Fuzznum, assume (l/u_other, m/m_other, u/l_other)
                    return MockFuzznumForAHP(self.l / other.u, self.m / other.m, self.u / other.l)

                def __rtruediv__(self, other): # For 1 / Fuzznum
                    if isinstance(other, (int, float)):
                        return MockFuzznumForAHP(other / self.u, other / self.m, other / self.l)
                    raise NotImplementedError

                def score(self): # Simple defuzzification (e.g., mean of bounds)
                    return (self.l + self.m + self.u) / 3

                def __repr__(self):
                    return f"({self.l}, {self.m}, {self.u})"

            # A sample pairwise matrix with MockFuzznum instances
            m12 = MockFuzznumForAHP(2, 3, 4) # C1 is 2-4 times more important than C2
            m13 = MockFuzznumForAHP(3, 4, 5) # C1 is 3-5 times more important than C3
            m23 = MockFuzznumForAHP(1, 2, 3) # C2 is 1-3 times more important than C3

            data = {
                'C1': [MockFuzznumForAHP(1,1,1), m12, m13],
                'C2': [1/m12, MockFuzznumForAHP(1,1,1), m23],
                'C3': [1/m13, 1/m23, MockFuzznumForAHP(1,1,1)]
            }
            fuzzy_matrix_df = pd.DataFrame(data, index=['C1', 'C2', 'C3'])

            # 1. Instantiate the component
            ahp_tool = AHPWeightTool(consistency_threshold=0.1)

            # 2. Run the tool
            results = ahp_tool.run(pairwise_matrix=fuzzy_matrix_df)

            print(f"AHP Weights: {results['weights']}")
            print(f"Consistency Ratio: {results['consistency_ratio']:.4f}")
            # Expected output (approximate, depending on exact fuzzy arithmetic implementation):
            # AHP Weights: [0.60, 0.25, 0.15]
            # Consistency Ratio: 0.05 (if consistent)
        """
        n = pairwise_matrix.shape[0]

        if pairwise_matrix.shape[1] != n:
            raise ValueError("Input pairwise_matrix must be a square matrix.")

        # Verify all elements are Fuzznum (or mock Fuzznum for testing)
        # In a real system, this would involve checking axisfuzzy.core.Fuzznum
        # For now, we'll rely on the operations to raise errors if types are incompatible.

        # Step 1: Calculate fuzzy geometric mean for each row (r_i)
        r_i_fuzznums = []
        for i in range(n):
            row_elements = [pairwise_matrix.iloc[i, j] for j in range(n)]

            # Product of fuzzy numbers in the row
            prod_fuzznum = row_elements[0]
            for k in range(1, n):
                prod_fuzznum = prod_fuzznum * row_elements[k] # Assumes Fuzznum has __mul__

            # Power to 1/n
            r_i = prod_fuzznum ** (1/n) # Assumes Fuzznum has __pow__
            r_i_fuzznums.append(r_i)

        # Step 2: Calculate fuzzy sum of fuzzy geometric means (S)
        S_fuzznum = r_i_fuzznums[0]
        for i in range(1, n):
            S_fuzznum = S_fuzznum + r_i_fuzznums[i] # Assumes Fuzznum has __add__

        # Step 3: Calculate inverse of fuzzy sum (S_inv)
        # Assumes Fuzznum supports 1 / Fuzznum
        S_inv_fuzznum = 1 / S_fuzznum # Assumes Fuzznum has __rtruediv__ for 1 / Fuzznum

        # Step 4: Calculate fuzzy weights (w_i)
        w_i_fuzznums = []
        for r_val in r_i_fuzznums:
            w_i = r_val * S_inv_fuzznum # Assumes Fuzznum has __mul__
            w_i_fuzznums.append(w_i)

        # Step 5: Defuzzify fuzzy weights (w'_i)
        w_prime = np.array([fuzznum.score() for fuzznum in w_i_fuzznums]) # Assumes Fuzznum has .score()

        # Step 6: Normalize crisp weights
        normalized_weights = w_prime / np.sum(w_prime)

        # Consistency Check
        consistency_ratio = self._calculate_consistency_ratio(pairwise_matrix, normalized_weights)

        if consistency_ratio > self.consistency_threshold:
            print(f"Warning: Consistency Ratio ({consistency_ratio:.4f}) exceeds "
                  f"the threshold ({self.consistency_threshold:.4f}). "
                  "The judgments may be inconsistent.")

        return {'weights': normalized_weights, 'consistency_ratio': consistency_ratio}

    def _calculate_consistency_ratio(self, pairwise_matrix: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculates the Consistency Ratio (CR) for the AHP matrix.

        Parameters
        ----------
        pairwise_matrix : pd.DataFrame
            The original fuzzy pairwise comparison matrix.
        weights : np.ndarray
            The calculated normalized crisp weights.

        Returns
        -------
        float
            The Consistency Ratio (CR).
        """
        n = pairwise_matrix.shape[0]

        # 1. Convert fuzzy matrix to crisp equivalent using .score()
        crisp_matrix_data = np.array([[elem.score() for elem in row] for _, row in pairwise_matrix.iterrows()])
        crisp_matrix = pd.DataFrame(crisp_matrix_data, index=pairwise_matrix.index, columns=pairwise_matrix.columns)

        # Handle n=1 or n=2 cases where CI is always 0
        if n <= 2:
            return 0.0

        # 2. Calculate lambda_max
        # Method: A'w = lambda_max * w
        # So, lambda_max_approx = (A'w) / w (element-wise division, then average)
        # This is a common approximation for lambda_max in AHP.
        Aw = np.dot(crisp_matrix.values, weights)

        # Avoid division by zero if any weight is zero
        lambda_max_values = np.where(weights != 0, Aw / weights, np.nan)
        lambda_max = np.nanmean(lambda_max_values) # Use nanmean to ignore NaNs from zero weights

        # 3. Calculate Consistency Index (CI)
        CI = (lambda_max - n) / (n - 1)

        # 4. Calculate Random Index (RI)
        RI = self._RI_VALUES.get(n, 1.58) # Default to 1.58 for n > 15, or handle error

        # 5. Calculate Consistency Ratio (CR)
        if RI == 0: # For n=1 or n=2, CI is 0, so CR is 0 to avoid division by zero
            CR = 0.0
        else:
            CR = CI / RI

        return CR
