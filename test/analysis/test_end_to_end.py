#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 20:25
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import List, Dict

import pytest
import pandas as pd
import numpy as np

from axisfuzzy.fuzzifier import Fuzzifier
# Import necessary components from our new analysis module
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.registry.base import register_tool
from axisfuzzy.analysis.dataframe import FuzzyDataFrame

# Make sure the accessor is registered by importing the module
import axisfuzzy.analysis.accessor


# --- Define and Register Mock Tools for the Test ---

@register_tool(
    inputs='CrispTable',
    outputs='FuzzyTable',
)
def fuzzify_from_dataframe(
        data: pd.DataFrame,
        f: Fuzzifier) -> FuzzyDataFrame:
    """
    A fuzzifier tool that converts a DataFrame to a FuzzyDataFrame
    using configurable fuzzification parameters.
    """
    return FuzzyDataFrame.from_pandas(data, fuzzifier=f)


@register_tool(
    inputs='FuzzyTable',
    outputs='WeightVector'
)
def calculate_uniform_weights(matrix: FuzzyDataFrame) -> np.ndarray:
    """A mock weighting tool that calculates uniform weights."""
    n_cols = matrix.shape[1]
    return np.full(n_cols, 1 / n_cols)


# --- The End-to-End Test ---

def test_full_pipeline_execution_via_accessor():
    """
    Tests the complete workflow from a pandas DataFrame to a final result
    through the FuzzyAccessor and FuzzyPipeline execution engine.
    """
    # 1. Prepare initial crisp data in a pandas DataFrame
    crisp_df = pd.DataFrame({
        'cost': [0.5, 0.7, 0.3],
        'safety': [0.2, 0.9, 0.4],
        'comfort': [0.8, 0.3, 0.1]
    }, index=['CarA', 'CarB', 'CarC'])

    fuzzifier = Fuzzifier(mtype='qrofn', mf='GaussianMF', mf_params=[{"sigma": 0.2, "c": 0.5}])

    # 2. Define the analysis workflow using the Fluent API
    p = FuzzyPipeline()

    # Declare the input node, promising a 'CrispTable'
    init_data = p.input("init_data", contract="CrispTable")

    # Define the steps
    fuzz_table = p.tool("fuzzify_from_dataframe")(
        data=init_data,
        f=fuzzifier)

    weights = p.tool("calculate_uniform_weights")(matrix=fuzz_table)

    # 3. Execute the pipeline via the FuzzyAccessor
    # The `crisp_df` DataFrame is automatically passed as 'init_data'
    final_results = crisp_df.fuzzy.run(pipeline=p)

    # 4. Assert the results
    assert isinstance(final_results, dict)

    # The final result should contain the output of the terminal node ('calculate_uniform_weights')
    assert "calculate_uniform_weights" in final_results

    final_weights = final_results["calculate_uniform_weights"]
    assert isinstance(final_weights, np.ndarray)
    assert final_weights.shape == (3,)
    assert np.allclose(final_weights, [1 / 3, 1 / 3, 1 / 3])

    print("\n✅ End-to-end test passed successfully!")
    print(f"Final results dictionary: {final_results}")


# To run this test file directly
if __name__ == "__main__":
    test_full_pipeline_execution_via_accessor()
