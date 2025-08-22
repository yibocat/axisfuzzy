#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 19:33
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# file: test_fluent_api.py (Updated)

import pytest
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.registry.base import register_tool


# --- Mock Tool Registration for Testing (unchanged) ---
@register_tool(
    name="fuzzify_correct",
    inputs={"matrix": "CrispTable"},
    outputs={"fuzz_result": "FuzzyTable"})
def mock_fuzzify(matrix):
    pass


@register_tool(
    name="get_weights",
    inputs={"table": "FuzzyTable"},
    outputs={"weights": "WeightVector"})
def mock_get_weights(table):
    pass


@register_tool(
    name="get_scores",
    inputs={"scores": "ScoreVector"},
    outputs={"final_scores": "ScoreVector"})
def mock_get_scores(scores):
    pass


# --- Test Cases (Updated) ---

def test_pipeline_creation_success():
    """Tests successful graph construction with matching contracts."""
    p = FuzzyPipeline()

    # MODIFIED: Declare the contract of the input node
    init_data = p.input("crisp_df", contract="CrispTable")

    # Now, the contract check should pass
    fuzz_table = p.tool("fuzzify_correct")(matrix=init_data)
    weights = p.tool("get_weights")(table=fuzz_table)

    assert len(p.steps) == 3
    assert weights.pipeline is p
    print("\n✅ Successful pipeline graph:")
    print(p)
    print(f"  Input: {init_data}")
    print(f"  Step 1: {fuzz_table}")
    print(f"  Step 2: {weights}")


def test_pipeline_creation_contract_mismatch():
    """Tests that a contract mismatch raises TypeError during graph construction."""
    p = FuzzyPipeline()

    # MODIFIED: Declare the input with a specific, but WRONG, contract
    init_data = p.input("crisp_df", contract="WeightVector")  # This input promises a WeightVector

    # This should fail: fuzzify_correct expects CrispTable, but gets a promise for WeightVector
    with pytest.raises(TypeError, match="Contract mismatch for tool 'fuzzify_correct' on input 'matrix'"):
        p.tool("fuzzify_correct")(matrix=init_data)

    print("\n✅ Correctly caught contract mismatch at graph-building time.")


def test_pipeline_creation_with_any_contract_input():
    """Tests that an input with the default 'any' contract passes static checks."""
    p = FuzzyPipeline()

    # Create an input without specifying a contract (defaults to 'any')
    init_data = p.input("crisp_df")

    # This should now PASS the static check because 'any' is a wildcard promise.
    # The actual data type will be checked at runtime.
    fuzz_table = p.tool("fuzzify_correct")(matrix=init_data)

    assert len(p.steps) == 2
    print("\n✅ Correctly allowed 'any' contract to pass static validation.")


# --- Run the tests ---
if __name__ == "__main__":
    test_pipeline_creation_success()
    test_pipeline_creation_contract_mismatch()
    test_pipeline_creation_with_any_contract_input()
