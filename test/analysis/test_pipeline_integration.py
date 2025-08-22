#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 02:14
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# file: tests/analysis/test_pipeline_integration.py

import pytest
import pandas as pd
import numpy as np

# 导入所有必要的组件
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.registry.base import register_tool
from axisfuzzy.analysis.dataframe import FuzzyDataFrame
from axisfuzzy.fuzzifier import Fuzzifier

# 确保 Accessor 被注册
import axisfuzzy.analysis.accessor


# --- Test Fixtures and Mock Tools ---
# 我们将在这里定义所有测试用例都会用到的工具和数据

@pytest.fixture(scope="module")
def setup_tools():
    """
    A pytest fixture to register a suite of mock tools for testing.
    This runs only once per test module.
    """

    @register_tool(inputs="CrispTable", outputs="FuzzyTable")
    def fuzzify_tool(data: pd.DataFrame, fuzzifier: Fuzzifier) -> FuzzyDataFrame:
        """Converts a crisp DataFrame to a FuzzyDataFrame."""
        return FuzzyDataFrame.from_pandas(data, fuzzifier=fuzzifier)

    @register_tool(inputs="FuzzyTable", outputs="WeightVector")
    def uniform_weight_tool(matrix: FuzzyDataFrame) -> np.ndarray:
        """Calculates uniform weights based on the number of columns."""
        return np.full(matrix.shape[1], 1 / matrix.shape[1])

    @register_tool(inputs="FuzzyTable", outputs="WeightVector")
    def constant_weight_tool(matrix: FuzzyDataFrame, const_weights: list) -> np.ndarray:
        """Returns a predefined constant weight vector."""
        return np.array(const_weights)

    @register_tool(
        inputs={"matrix": "FuzzyTable", "weights": "WeightVector"},
        outputs="ScoreVector"
    )
    def aggregate_tool(matrix: FuzzyDataFrame, weights: np.ndarray) -> np.ndarray:
        """A mock aggregation tool."""
        # Simplified aggregation for testing purposes
        crisp_matrix = np.array(
            [[f.mean() if hasattr(f, 'mean') else 0.5 for f in matrix[col]] for col in matrix.columns]).T
        return np.dot(crisp_matrix, weights)

    @register_tool(
        inputs="ScoreVector",
        outputs={"ranking": "RankingResult", "top_performer": "str"}
    )
    def decision_tool(scores: np.ndarray, alternative_names: list) -> dict:
        """A mock multi-output decision tool."""
        ranking_indices = np.argsort(scores)[::-1]  # Sort descending
        return {
            "ranking": [alternative_names[i] for i in ranking_indices],
            "top_performer": alternative_names[ranking_indices[0]]
        }

    # This yield is important for fixtures. It runs the tests after this point.
    yield
    # Teardown can happen here if needed, but our registry is global, so no teardown.


@pytest.fixture
def crisp_data() -> pd.DataFrame:
    """Provides a standard crisp DataFrame for tests."""
    return pd.DataFrame({
        'cost': [0.5, 0.7, 0.3],
        'safety': [0.2, 0.9, 0.4],
        'comfort': [0.8, 0.3, 0.1]
    }, index=['CarA', 'CarB', 'CarC'])


@pytest.fixture
def fuzzifier_instance() -> Fuzzifier:
    """Provides a standard Fuzzifier instance for tests."""
    return Fuzzifier(mtype='qrofn', mf='GaussianMF', mf_params=[{"sigma": 0.2, "c": 0.5}])


# --- Test Cases ---

def test_linear_pipeline_single_output(setup_tools, crisp_data, fuzzifier_instance):
    """
    Tests a simple, linear pipeline with a single final output.
    Verifies that `run()` returns the final value directly.
    """
    p = FuzzyPipeline()

    # Define a linear workflow
    crisp_input = p.input(contract="CrispTable")
    fuzz_table = p.tool("fuzzify_tool")(data=crisp_input, fuzzifier=fuzzifier_instance)
    weights = p.tool("uniform_weight_tool")(matrix=fuzz_table)

    # Execute via Accessor
    final_weights = crisp_data.fuzzy.run(p)

    # Assertions
    assert isinstance(final_weights, np.ndarray)
    assert final_weights.shape == (3,)
    assert np.allclose(final_weights, [1 / 3, 1 / 3, 1 / 3])
    print("\n✅ Test linear pipeline (single output): PASSED")


def test_dag_pipeline_multi_source_and_multi_output(setup_tools, crisp_data, fuzzifier_instance):
    """
    Tests a non-linear (DAG) pipeline with multiple branches and a multi-output terminal node.
    Verifies that `run()` returns a dictionary of results.
    """
    p = FuzzyPipeline()

    # Define a non-linear workflow
    crisp_input = p.input("init_data", contract="CrispTable")

    # Branch A: Fuzzification
    fuzz_table = p.tool("fuzzify_tool")(data=crisp_input, fuzzifier=fuzzifier_instance)

    # Branch B: Predefined weights (no data dependency)
    # Note: This tool has no data inputs, only a parameter. We need to adjust our registration logic slightly
    # For now, let's make it depend on the fuzz_table to fit the current model.
    # A better approach would be a tool with no 'inputs' contract.
    const_weights = p.tool("constant_weight_tool")(matrix=fuzz_table, const_weights=[0.6, 0.3, 0.1])

    # Merge branches
    scores = p.tool("aggregate_tool")(matrix=fuzz_table, weights=const_weights)

    # Final multi-output step
    decision = p.tool("decision_tool")(scores=scores, alternative_names=list(crisp_data.index))

    # Execute directly via FuzzyPipeline.run()
    # Note: The 'decision_tool' is the terminal node.
    final_results = p.run(crisp_data)

    # Assertions
    assert isinstance(final_results, dict)
    assert "ranking" in final_results
    assert "top_performer" in final_results
    assert final_results["ranking"] == ['CarA', 'CarB', 'CarC']  # Based on mock aggregation
    assert final_results["top_performer"] == 'CarA'
    print("\n✅ Test DAG pipeline (multi-output): PASSED")


def test_pipeline_starting_with_fuzzy_data(setup_tools, crisp_data, fuzzifier_instance):
    """
    Tests a pipeline that starts directly with a FuzzyDataFrame, skipping fuzzification.
    """
    # 1. Prepare a pre-fuzzified DataFrame
    fuzzy_df = FuzzyDataFrame.from_pandas(crisp_data, fuzzifier=fuzzifier_instance)

    p = FuzzyPipeline()

    # 2. Define a pipeline that expects a FuzzyTable as its primary input
    fuzzy_input = p.input("my_fuzzy_data", contract="FuzzyTable")
    weights = p.tool("uniform_weight_tool")(matrix=fuzzy_input)

    # 3. Execute with the FuzzyDataFrame
    final_weights = p.run(initial_data={"my_fuzzy_data": fuzzy_df})

    # Assertions
    assert isinstance(final_weights, np.ndarray)
    assert np.allclose(final_weights, [1 / 3, 1 / 3, 1 / 3])
    print("\n✅ Test pipeline starting with FuzzyDataFrame: PASSED")


def test_cycle_detection_in_dag(setup_tools):
    """
    Tests that the pipeline correctly detects and raises an error for cyclical dependencies.
    """
    p = FuzzyPipeline()

    input_a = p.input("a", contract="any")

    # Create a cycle: B depends on C, and C depends on B
    output_b = p.tool("uniform_weight_tool")(matrix=input_a)  # Placeholder tool
    output_c = p.tool("uniform_weight_tool")(matrix=output_b)

    # Now, try to make B depend on C, creating a cycle
    with pytest.raises(TypeError, match="expects data inputs"):  # _add_step will catch this
        p.tool("uniform_weight_tool")(matrix=output_c)

    # A more subtle cycle test
    p2 = FuzzyPipeline()
    node1 = p2.input("start", contract="FuzzyTable")
    node2 = p2.tool("uniform_weight_tool")(matrix=node1)
    node3 = p2.tool("aggregate_tool")(matrix=node1, weights=node2)

    # Create the cycle: make node2 depend on node3
    # We need to manually manipulate the internal state for this test,
    # as the Fluent API makes it hard to create cycles accidentally.
    node2_step_id = node2.step_id
    p2._steps[node2_step_id]['dependencies']['extra_dep'] = node3
    p2._steps[node2_step_id]['inputs']['extra_dep'] = node3

    with pytest.raises(ValueError, match="A cycle was detected in the pipeline graph"):
        p2.run(FuzzyDataFrame())  # Pass mock data

    print("\n✅ Test cycle detection: PASSED")


def test_runtime_contract_validation_failure(setup_tools, crisp_data):
    """
    Tests that a runtime contract mismatch raises a TypeError.
    """
    p = FuzzyPipeline()

    # This pipeline is valid at graph-building time because of the 'any' contract.
    any_input = p.input()  # Default contract is 'any'
    weights = p.tool("uniform_weight_tool")(matrix=any_input)  # This tool expects a FuzzyTable

    # Execute with incorrect data type (a crisp DataFrame)
    with pytest.raises(TypeError, match="Runtime contract validation failed"):
        p.run(crisp_data)

    print("\n✅ Test runtime contract validation failure: PASSED")
