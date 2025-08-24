#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 16:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np
import pandas as pd

from axisfuzzy.analysis._components.basic import (
    NormalizationTool, WeightNormalizationTool,
    StatisticsTool, SimpleAggregationTool
)


# --- Test Data Fixtures ---

@pytest.fixture
def sample_data():
    """Provides sample data for testing."""
    return pd.DataFrame({
        'Cost': [100, 200, 150],
        'Quality': [8, 9, 7],
        'Speed': [50, 60, 55]
    }, index=['A', 'B', 'C'])


@pytest.fixture
def sample_weights():
    """Provides sample weights for testing."""
    return np.array([3, 2, 5])  # Sum = 10


# --- NormalizationTool Tests ---

def test_normalization_tool_min_max(sample_data):
    """Tests min-max normalization."""
    tool = NormalizationTool(method='min_max', axis=0)  # Column-wise
    result = tool.run(data=sample_data)

    normalized = result['normalized_data']

    # Check that each column is normalized to [0, 1]
    for col in normalized.columns:
        assert normalized[col].min() == pytest.approx(0.0, abs=1e-6)
        assert normalized[col].max() == pytest.approx(1.0, abs=1e-6)


def test_normalization_tool_sum(sample_data):
    """Tests sum normalization."""
    tool = NormalizationTool(method='sum', axis=0)  # Column-wise
    result = tool.run(data=sample_data)

    normalized = result['normalized_data']

    # Check that each column sums to 1
    for col in normalized.columns:
        assert normalized[col].sum() == pytest.approx(1.0, abs=1e-6)


def test_normalization_tool_invalid_method():
    """Tests invalid method raises error."""
    with pytest.raises(ValueError, match="Method must be one of"):
        NormalizationTool(method='invalid')


# --- WeightNormalizationTool Tests ---

def test_weight_normalization_tool_array(sample_weights):
    """Tests weight normalization with numpy array."""
    tool = WeightNormalizationTool()
    result = tool.run(weights=sample_weights)

    normalized = result['normalized_weights']

    assert isinstance(normalized, np.ndarray)
    assert normalized.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(normalized, [0.3, 0.2, 0.5])


def test_weight_normalization_tool_series():
    """Tests weight normalization with pandas Series."""
    weights_series = pd.Series([3, 2, 5], index=['A', 'B', 'C'])
    tool = WeightNormalizationTool()
    result = tool.run(weights=weights_series)

    normalized = result['normalized_weights']

    assert isinstance(normalized, pd.Series)
    assert normalized.sum() == pytest.approx(1.0, abs=1e-6)
    assert list(normalized.index) == ['A', 'B', 'C']


def test_weight_normalization_tool_negative():
    """Tests weight normalization with negative values."""
    weights = np.array([3, -1, 2])
    tool = WeightNormalizationTool(ensure_positive=True)
    result = tool.run(weights=weights)

    normalized = result['normalized_weights']

    # Negative weight should be set to 0, then normalized
    expected = np.array([3, 0, 2]) / 5  # [0.6, 0.0, 0.4]
    np.testing.assert_allclose(normalized, expected)


# --- StatisticsTool Tests ---

def test_statistics_tool(sample_data):
    """Tests statistical calculation."""
    tool = StatisticsTool()
    result = tool.run(data=sample_data)

    stats = result['statistics']

    assert isinstance(stats, dict)
    required_keys = ['mean', 'std', 'min', 'max', 'median', 'count']
    assert all(key in stats for key in required_keys)
    assert stats['count'] == 9  # 3x3 matrix
    assert stats['min'] == 7.0
    assert stats['max'] == 200.0


# --- SimpleAggregationTool Tests ---

def test_simple_aggregation_tool_mean(sample_data):
    """Tests mean aggregation."""
    tool = SimpleAggregationTool(operation='mean', axis=1)  # Row-wise
    result = tool.run(data=sample_data)

    aggregated = result['aggregated_values']

    # Check mean calculation for each row
    expected_means = sample_data.mean(axis=1)
    pd.testing.assert_series_equal(aggregated, expected_means)


def test_simple_aggregation_tool_sum(sample_data):
    """Tests sum aggregation."""
    tool = SimpleAggregationTool(operation='sum', axis=0)  # Column-wise
    result = tool.run(data=sample_data)

    aggregated = result['aggregated_values']

    # Check sum calculation for each column
    expected_sums = sample_data.sum(axis=0)
    pd.testing.assert_series_equal(aggregated, expected_sums)


def test_simple_aggregation_tool_invalid_operation():
    """Tests invalid operation raises error."""
    with pytest.raises(ValueError, match="Operation must be one of"):
        SimpleAggregationTool(operation='invalid')


if __name__ == "__main__":
    pytest.main()
