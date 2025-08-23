#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 14:43
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# tests/analysis/components/test_vikor.py

#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/26 10:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np

from axisfuzzy.analysis.components.vikor import VIKORTool


# --- Mock Objects for Testing ---
# These mocks simulate the behavior of core AxisFuzzy objects
# without needing the full library, allowing for focused component testing.

class MockFuzznum:
    """A mock fuzzy number with a simple value for distance calculation."""

    def __init__(self, value):
        self.value = value

    def distance(self, other):
        # Simulate distance as the absolute difference of crisp values
        return abs(self.value - other.value)

    def __repr__(self):
        return f"MockFuzznum({self.value})"


class MockFuzzarray:
    """A mock fuzzy array that holds a list of MockFuzznum."""

    def __init__(self, nums):
        self._nums = [MockFuzznum(n) for n in nums]

    def max(self):
        return max(self._nums, key=lambda fn: fn.value)

    def min(self):
        return min(self._nums, key=lambda fn: fn.value)

    def distance(self, other_fuzznum):
        # Simulate vectorized distance calculation
        return np.array([fn.distance(other_fuzznum) for fn in self._nums])


class MockFuzzyDataFrame:
    """A mock FuzzyDataFrame that behaves like the real one for the component."""

    def __init__(self, data_dict):
        self._data = {k: MockFuzzarray(v) for k, v in data_dict.items()}
        self.columns = list(data_dict.keys())
        # Assuming all columns have the same length
        num_rows = len(next(iter(data_dict.values())))
        self.shape = (num_rows, len(self.columns))

    def __getitem__(self, key):
        return self._data[key]

    @property
    def values(self):
        # A simplified representation for np.zeros_like
        return np.zeros(self.shape)


# --- Test Fixture ---

@pytest.fixture
def vikor_test_data():
    """Provides a standard set of data for VIKOR testing."""
    # 3 alternatives, 3 criteria
    data = {
        'C1': [7, 8, 9],  # Alt C is best
        'C2': [8, 9, 7],  # Alt B is best
        'C3': [9, 7, 8]  # Alt A is best
    }
    mock_df = MockFuzzyDataFrame(data)
    weights = np.array([0.4, 0.3, 0.3])
    names = ['Alt_A', 'Alt_B', 'Alt_C']
    return mock_df, weights, names


# --- Test Cases ---

def test_vikor_tool_initialization():
    """Tests the constructor of VIKORTool."""
    tool = VIKORTool(v=0.7)
    assert tool.v == 0.7
    with pytest.raises(ValueError):
        VIKORTool(v=1.1)
    with pytest.raises(ValueError):
        VIKORTool(v=-0.1)


def test_vikor_tool_run(vikor_test_data):
    """Tests the main run method of VIKORTool."""
    mock_df, weights, names = vikor_test_data

    tool = VIKORTool(v=0.5)
    result = tool.run(matrix=mock_df, weights=weights, alternative_names=names)

    # --- Assertions ---
    assert 'scores' in result
    assert 'ranking' in result

    scores = result['scores']
    ranking = result['ranking']

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)

    assert isinstance(ranking, list)
    assert len(ranking) == 3
    assert set(ranking) == set(names)

    # --- Manual calculation for verification ---
    # FPIS: C1=9, C2=9, C3=9
    # FNIS: C1=7, C2=7, C3=7
    # Span for all criteria: 9-7=2

    # Normalized Distances:
    # Alt_A (7, 8, 9): [0.4*(9-7)/2, 0.3*(9-8)/2, 0.3*(9-9)/2] = [0.4, 0.15, 0.0]
    # Alt_B (8, 9, 7): [0.4*(9-8)/2, 0.3*(9-9)/2, 0.3*(9-7)/2] = [0.2, 0.0, 0.3]
    # Alt_C (9, 7, 8): [0.4*(9-9)/2, 0.3*(9-7)/2, 0.3*(9-8)/2] = [0.0, 0.3, 0.15]

    # S scores:
    # S_A = 0.4 + 0.15 + 0.0 = 0.55
    # S_B = 0.2 + 0.0 + 0.3 = 0.50
    # S_C = 0.0 + 0.3 + 0.15 = 0.45
    # S* = 0.45, S- = 0.55

    # R scores:
    # R_A = max(0.4, 0.15, 0.0) = 0.4
    # R_B = max(0.2, 0.0, 0.3) = 0.3
    # R_C = max(0.0, 0.3, 0.15) = 0.3
    # R* = 0.3, R- = 0.4

    # Q scores (v=0.5):
    # Q_A = 0.5*(0.55-0.45)/(0.55-0.45) + 0.5*(0.4-0.3)/(0.4-0.3) = 0.5*1 + 0.5*1 = 1.0
    # Q_B = 0.5*(0.50-0.45)/(0.1) + 0.5*(0.3-0.3)/(0.1) = 0.5*0.5 + 0.5*0 = 0.25
    # Q_C = 0.5*(0.45-0.45)/(0.1) + 0.5*(0.3-0.3)/(0.1) = 0.5*0 + 0.5*0 = 0.0

    # Expected Q scores: [1.0, 0.25, 0.0]
    # Expected ranking: Alt_C, Alt_B, Alt_A

    np.testing.assert_allclose(scores, np.array([1.0, 0.25, 0.0]), atol=1e-6)
    assert ranking == ['Alt_C', 'Alt_B', 'Alt_A']


def test_vikor_with_different_v(vikor_test_data):
    """Tests that the 'v' parameter influences the result."""
    mock_df, weights, names = vikor_test_data

    tool_v0 = VIKORTool(v=0.0)  # Focus only on regret (R)
    result_v0 = tool_v0.run(matrix=mock_df, weights=weights, alternative_names=names)

    # R scores are [0.4, 0.3, 0.3]. Ascending order: B and C are tied, then A.
    # Q scores (v=0): Q_A=1, Q_B=0, Q_C=0.
    # Ranking: [Alt_B, Alt_C, Alt_A] or [Alt_C, Alt_B, Alt_A]
    # argsort is stable, so original order is preserved for ties. B is before C.
    assert result_v0['ranking'] == ['Alt_B', 'Alt_C', 'Alt_A']

    tool_v1 = VIKORTool(v=1.0)  # Focus only on group utility (S)
    result_v1 = tool_v1.run(matrix=mock_df, weights=weights, alternative_names=names)

    # S scores are [0.55, 0.50, 0.45]. Ascending order: C, B, A.
    assert result_v1['ranking'] == ['Alt_C', 'Alt_B', 'Alt_A']


if __name__ == "__main__":
    pytest.main()