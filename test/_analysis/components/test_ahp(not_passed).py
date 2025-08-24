#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 15:14
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np
import pandas as pd

from axisfuzzy.analysis._components.ahp import AHPWeightTool


# --- Mock Object for Testing Fuzznum Arithmetic ---
class MockFuzznumForAHP:
    """
    A mock fuzzy number (simulating TFN) with basic arithmetic operations
    and a .score() method for AHP testing.
    (l, m, u) represents lower, middle, upper bounds.
    """

    def __init__(self, l, m, u):
        if not (l <= m <= u):
            raise ValueError(f"Invalid TFN: ({l}, {m}, {u}). Must be l <= m <= u.")
        self.l, self.m, self.u = float(l), float(m), float(u)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockFuzznumForAHP(self.l * other, self.m * other, self.u * other)
        return MockFuzznumForAHP(self.l * other.l, self.m * other.m, self.u * other.u)

    def __rmul__(self, other):
        return self.__mul__(other)  # Commutative for scalar multiplication

    def __pow__(self, power):
        # Assuming power is positive (e.g., 1/n)
        return MockFuzznumForAHP(self.l ** power, self.m ** power, self.u ** power)

    def __add__(self, other):
        return MockFuzznumForAHP(self.l + other.l, self.m + other.m, self.u + other.u)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError("division by zero")
            return MockFuzznumForAHP(self.l / other, self.m / other, self.u / other)

        # For Fuzznum / Fuzznum, assuming (l/u_other, m/m_other, u/l_other) for TFN division
        if other.l == 0 or other.m == 0 or other.u == 0:
            raise ZeroDivisionError("division by zero with fuzzy number containing zero")
        return MockFuzznumForAHP(self.l / other.u, self.m / other.m, self.u / other.l)

    def __rtruediv__(self, other):  # For scalar / Fuzznum (e.g., 1 / Fuzznum)
        if isinstance(other, (int, float)):
            if self.l == 0 or self.m == 0 or self.u == 0:
                raise ZeroDivisionError("division by zero with fuzzy number containing zero")
            return MockFuzznumForAHP(other / self.u, other / self.m, other / self.l)
        raise NotImplementedError("Only scalar division for rdiv is implemented.")

    def score(self):  # Simple defuzzification (e.g., mean of bounds)
        return (self.l + self.m + self.u) / 3

    def __repr__(self):
        return f"MockFuzznumForAHP({self.l:.2f}, {self.m:.2f}, {self.u:.2f})"


# --- Test Fixtures ---

@pytest.fixture
def ahp_test_matrix_consistent():
    """Provides a 3x3 relatively consistent fuzzy AHP matrix."""
    # Based on a crisp matrix [1 3 5; 1/3 1 3; 1/5 1/3 1]
    # Weights should be approx [0.65, 0.23, 0.12]
    # CI for crisp is (3.00 - 3) / 2 = 0, CR = 0
    # Let's add slight fuzziness

    # C1 vs C2: About 3 times more important
    m12 = MockFuzznumForAHP(2.5, 3, 3.5)
    # C1 vs C3: About 5 times more important
    m13 = MockFuzznumForAHP(4.5, 5, 5.5)
    # C2 vs C3: About 3 times more important
    m23 = MockFuzznumForAHP(2.5, 3, 3.5)

    data = {
        'C1': [MockFuzznumForAHP(1, 1, 1), m12, m13],
        'C2': [1 / m12, MockFuzznumForAHP(1, 1, 1), m23],
        'C3': [1 / m13, 1 / m23, MockFuzznumForAHP(1, 1, 1)]
    }
    df = pd.DataFrame(data, index=['C1', 'C2', 'C3'])
    return df


@pytest.fixture
def ahp_test_matrix_inconsistent():
    """Provides a 3x3 inconsistent fuzzy AHP matrix."""
    # Classic inconsistent example: A > B, B > C, C > A
    # A > B (3), B > C (3), C > A (3) -> inconsistent
    m_ab = MockFuzznumForAHP(2.5, 3, 3.5)
    m_bc = MockFuzznumForAHP(2.5, 3, 3.5)
    m_ca = MockFuzznumForAHP(2.5, 3, 3.5)

    data = {
        'A': [MockFuzznumForAHP(1, 1, 1), m_ab, 1 / m_ca],
        'B': [1 / m_ab, MockFuzznumForAHP(1, 1, 1), m_bc],
        'C': [m_ca, 1 / m_bc, MockFuzznumForAHP(1, 1, 1)]
    }
    df = pd.DataFrame(data, index=['A', 'B', 'C'])
    return df


@pytest.fixture
def ahp_test_matrix_n1():
    """Provides a 1x1 AHP matrix (always consistent)."""
    data = {'C1': [MockFuzznumForAHP(1, 1, 1)]}
    df = pd.DataFrame(data, index=['C1'])
    return df


@pytest.fixture
def ahp_test_matrix_n2():
    """Provides a 2x2 AHP matrix (always consistent)."""
    m12 = MockFuzznumForAHP(2, 3, 4)
    data = {
        'C1': [MockFuzznumForAHP(1, 1, 1), m12],
        'C2': [1 / m12, MockFuzznumForAHP(1, 1, 1)]
    }
    df = pd.DataFrame(data, index=['C1', 'C2'])
    return df


# --- Test Cases ---

def test_ahp_tool_initialization():
    """Tests the constructor of AHPWeightTool."""
    tool = AHPWeightTool(consistency_threshold=0.05)
    assert tool.consistency_threshold == 0.05
    with pytest.raises(ValueError):
        AHPWeightTool(consistency_threshold=0)


def test_ahp_tool_run_consistent(ahp_test_matrix_consistent, capsys):
    """Tests the run method with a relatively consistent matrix."""
    matrix = ahp_test_matrix_consistent
    tool = AHPWeightTool(consistency_threshold=0.1)
    results = tool.run(pairwise_matrix=matrix)

    assert 'weights' in results
    assert 'consistency_ratio' in results

    weights = results['weights']
    cr = results['consistency_ratio']

    assert isinstance(weights, np.ndarray)
    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)

    assert isinstance(cr, float)
    assert cr <= tool.consistency_threshold  # Should be consistent

    # Check approximate weight values (based on manual calculation)
    # Score matrix: [[1, 3, 5], [0.333, 1, 3], [0.2, 0.333, 1]]
    # Principal eigenvalue for this crisp matrix is approx 3.003
    # CI = (3.003-3)/2 = 0.0015
    # CR = 0.0015 / 0.58 = 0.0025
    np.testing.assert_allclose(weights, [0.64, 0.23, 0.13], rtol=0.05,
                               atol=0.01)  # Allow for slight variations due to fuzziness
    np.testing.assert_allclose(cr, 0.0025, rtol=0.05, atol=0.001)

    # Check no warning issued if consistent
    captured = capsys.readouterr()
    assert "Warning" not in captured.out


def test_ahp_tool_run_inconsistent(ahp_test_matrix_inconsistent, capsys):
    """Tests the run method with an inconsistent matrix."""
    matrix = ahp_test_matrix_inconsistent
    tool = AHPWeightTool(consistency_threshold=0.05)  # Set a low threshold to ensure warning

    results = tool.run(pairwise_matrix=matrix)
    weights = results['weights']
    cr = results['consistency_ratio']

    assert isinstance(weights, np.ndarray)
    assert np.isclose(np.sum(weights), 1.0)

    assert isinstance(cr, float)
    assert cr > tool.consistency_threshold  # Should be inconsistent

    # Check that a warning is issued
    captured = capsys.readouterr()
    assert "Warning: Consistency Ratio" in captured.out
    assert "exceeds the threshold" in captured.out

    # For a 3x3 cyclic inconsistent matrix, CR is usually high, around 0.58
    np.testing.assert_allclose(cr, 0.58, rtol=0.05, atol=0.01)


def test_ahp_tool_run_n1(ahp_test_matrix_n1):
    """Tests AHP with a 1x1 matrix (CR should be 0)."""
    matrix = ahp_test_matrix_n1
    tool = AHPWeightTool()
    results = tool.run(pairwise_matrix=matrix)

    assert np.isclose(results['weights'][0], 1.0)
    assert np.isclose(results['consistency_ratio'], 0.0)


def test_ahp_tool_run_n2(ahp_test_matrix_n2):
    """Tests AHP with a 2x2 matrix (CR should be 0)."""
    matrix = ahp_test_matrix_n2
    tool = AHPWeightTool()
    results = tool.run(pairwise_matrix=matrix)

    assert np.isclose(np.sum(results['weights']), 1.0)
    assert np.isclose(results['consistency_ratio'], 0.0)


def test_ahp_tool_non_square_matrix():
    """Tests that AHP raises error for non-square matrix."""
    data = {
        'C1': [MockFuzznumForAHP(1, 1, 1), MockFuzznumForAHP(2, 3, 4)],
        'C2': [MockFuzznumForAHP(1, 1, 1), MockFuzznumForAHP(1, 1, 1)]
    }
    non_square_df = pd.DataFrame(data, index=['R1', 'R2', 'R3'])  # 3 rows, 2 cols
    tool = AHPWeightTool()
    with pytest.raises(ValueError, match="Input pairwise_matrix must be a square matrix."):
        tool.run(pairwise_matrix=non_square_df)


def test_ahp_tool_missing_fuzznum_methods():
    """Tests graceful failure if MockFuzznumForAHP lacks required methods."""

    class BadMockFuzznum:
        def __init__(self, val): self.val = val

        # Missing __mul__, __pow__, __add__, __truediv__, __rtruediv__, score
        def score(self): return self.val  # Only score is implemented for CR check

    data = {
        'C1': [BadMockFuzznum(1), BadMockFuzznum(2)],
        'C2': [BadMockFuzznum(0.5), BadMockFuzznum(1)]
    }
    df = pd.DataFrame(data, index=['C1', 'C2'])
    tool = AHPWeightTool()

    # Expecting TypeError or AttributeError due to missing arithmetic methods
    with pytest.raises((TypeError, AttributeError)):
        tool.run(pairwise_matrix=df)


if __name__ == "__main__":
    pytest.main([__file__])
