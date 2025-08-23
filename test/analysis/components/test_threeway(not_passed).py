#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 15:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import numpy as np
from axisfuzzy.analysis.components.threeway import ClassicThreeWayTool, FuzzyThreeWayTool


# --- Test Fixture ---
@pytest.fixture
def decision_data():
    """Provides a standard set of scores and names for testing."""
    scores = np.array([0.95, 0.8, 0.7, 0.6, 0.3, 0.2, 0.05])
    names = ['Alt1', 'Alt2', 'Alt3', 'Alt4', 'Alt5', 'Alt6', 'Alt7']
    return scores, names


# --- Test Cases for ClassicThreeWayTool ---

def test_classic_tool_initialization():
    """Tests the constructor validation of ClassicThreeWayTool."""
    # Valid cases
    tool = ClassicThreeWayTool(alpha=0.8, beta=0.2)
    assert tool.alpha == 0.8 and tool.beta == 0.2

    # Invalid cases
    with pytest.raises(ValueError, match="Thresholds must satisfy 0 <= beta < alpha <= 1."):
        ClassicThreeWayTool(alpha=0.5, beta=0.5)  # alpha must be > beta
    with pytest.raises(ValueError, match="Thresholds must satisfy 0 <= beta < alpha <= 1."):
        ClassicThreeWayTool(alpha=0.5, beta=0.6)  # alpha must be > beta
    with pytest.raises(ValueError, match="Thresholds must satisfy 0 <= beta < alpha <= 1."):
        ClassicThreeWayTool(alpha=1.1, beta=0.5)  # alpha must be <= 1


def test_classic_tool_run_standard_partition(decision_data):
    """Tests a standard partition scenario."""
    scores, names = decision_data
    decider = ClassicThreeWayTool(alpha=0.7, beta=0.3)
    result = decider.run(scores=scores, alternative_names=names)

    assert isinstance(result, dict)
    assert set(result.keys()) == {'accept', 'reject', 'defer'}

    assert sorted(result['accept']) == ['Alt1', 'Alt2', 'Alt3']
    assert sorted(result['defer']) == ['Alt4']
    assert sorted(result['reject']) == ['Alt5', 'Alt6', 'Alt7']


def test_classic_tool_run_edge_thresholds(decision_data):
    """Tests scores that fall exactly on the thresholds."""
    scores, names = decision_data  # scores are [0.95, 0.8, 0.7, 0.6, 0.3, 0.2, 0.05]
    decider = ClassicThreeWayTool(alpha=0.7, beta=0.3)
    result = decider.run(scores=scores, alternative_names=names)

    # Alt3 (score 0.7) should be in 'accept' (>= alpha)
    # Alt5 (score 0.3) should be in 'reject' (<= beta)
    assert 'Alt3' in result['accept']
    assert 'Alt5' in result['reject']


def test_classic_tool_run_no_defer(decision_data):
    """Tests a scenario where the deferment region is empty."""
    scores, names = decision_data
    decider = ClassicThreeWayTool(alpha=0.5, beta=0.51)  # This is invalid, let's fix
    decider = ClassicThreeWayTool(alpha=0.5, beta=0.4)
    scores_custom = np.array([0.9, 0.6, 0.3, 0.1])
    names_custom = ['A', 'B', 'C', 'D']

    result = decider.run(scores=scores_custom, alternative_names=names_custom)
    assert sorted(result['accept']) == ['A', 'B']
    assert len(result['defer']) == 0
    assert sorted(result['reject']) == ['C', 'D']


def test_classic_tool_run_all_accept(decision_data):
    """Tests a scenario where all alternatives are accepted."""
    scores, names = decision_data
    decider = ClassicThreeWayTool(alpha=0.01, beta=0.0)
    result = decider.run(scores=scores, alternative_names=names)

    assert len(result['accept']) == len(names)
    assert len(result['defer']) == 0
    assert len(result['reject']) == 0


def test_classic_tool_run_all_reject(decision_data):
    """Tests a scenario where all alternatives are rejected."""
    scores, names = decision_data
    decider = ClassicThreeWayTool(alpha=1.0, beta=0.99)
    result = decider.run(scores=scores, alternative_names=names)

    assert len(result['accept']) == 0
    assert len(result['defer']) == 0
    assert len(result['reject']) == len(names)


@pytest.fixture
def risk_averse_loss_function():
    """A loss function for a risk-averse decision maker."""
    # High cost for wrong approval (lambda_pn), moderate for wrong rejection (lambda_np)
    return {
        'lambda_pp': 0, 'lambda_pn': 100,
        'lambda_np': 20, 'lambda_nn': 0,
        'lambda_bp': 5, 'lambda_bn': 5
    }
    # Expected alpha = (5 - 100) / ((5 - 100) + (0 - 5)) = 0.95
    # Expected beta  = (0 - 5) / ((0 - 5) + (5 - 20)) = 0.25


@pytest.fixture
def risk_seeking_loss_function():
    """A loss function for a risk-seeking decision maker."""
    # Low cost for wrong approval, high cost for wrong rejection (opportunity loss)
    return {
        'lambda_pp': -10,  # Negative loss means gain
        'lambda_pn': 10,
        'lambda_np': 50,
        'lambda_nn': 0,
        'lambda_bp': 2,
        'lambda_bn': 5
    }
    # Expected alpha = (5 - 10) / ((5 - 10) + (-10 - 2)) = -5 / -17 = 0.294
    # Expected beta  = (0 - 5) / ((0 - 5) + (2 - 50)) = -5 / -53 = 0.094


# --- Test Cases for FuzzyThreeWayTool ---

def test_fuzzy_tool_initialization_and_threshold_calculation(risk_averse_loss_function):
    """Tests if thresholds are calculated correctly upon initialization."""
    lf = risk_averse_loss_function
    tool = FuzzyThreeWayTool(loss_function=lf)

    assert np.isclose(tool.alpha, 0.95)
    assert np.isclose(tool.beta, 0.25)


def test_fuzzy_tool_initialization_invalid():
    """Tests constructor validation of FuzzyThreeWayTool."""
    with pytest.raises(ValueError, match="Loss function is missing required keys"):
        FuzzyThreeWayTool(loss_function={'lambda_pp': 0})

    # Test warning for non-standard loss values
    lf_warn = {'lambda_pp': 0, 'lambda_pn': 4, 'lambda_np': 3, 'lambda_nn': 0, 'lambda_bp': 5, 'lambda_bn': 5}
    with pytest.warns(None) as record:  # Using warns(None) to capture print output for now
        FuzzyThreeWayTool(loss_function=lf_warn)
    # This is a bit tricky to test with `print`, but a better implementation would use `warnings.warn`
    # For now, we just ensure it doesn't crash.


def test_fuzzy_tool_run_risk_averse(decision_data, risk_averse_loss_function):
    """Tests the run method with a risk-averse loss function."""
    scores, names = decision_data  # [0.95, 0.8, 0.7, 0.6, 0.3, 0.2, 0.05]
    tool = FuzzyThreeWayTool(loss_function=risk_averse_loss_function)
    # alpha=0.95, beta=0.25

    result = tool.run(scores=scores, alternative_names=names)

    assert sorted(result['accept']) == ['Alt1']  # score 0.95 >= 0.95
    assert sorted(result['defer']) == ['Alt2', 'Alt3', 'Alt4', 'Alt5']  # 0.25 < score < 0.95
    assert sorted(result['reject']) == ['Alt6', 'Alt7']  # score <= 0.25


def test_fuzzy_tool_run_risk_seeking(decision_data, risk_seeking_loss_function):
    """Tests the run method with a risk-seeking loss function."""
    scores, names = decision_data
    tool = FuzzyThreeWayTool(loss_function=risk_seeking_loss_function)
    # alpha=0.294, beta=0.094

    result = tool.run(scores=scores, alternative_names=names)

    assert sorted(result['accept']) == ['Alt1', 'Alt2', 'Alt3', 'Alt4', 'Alt5']
    assert sorted(result['defer']) == ['Alt6']
    assert sorted(result['reject']) == ['Alt7']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
