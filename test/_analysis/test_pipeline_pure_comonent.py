#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 11:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Unit and integration tests for the purified, component-based FuzzyPipeline.

This test file specifically validates the core functionality after the Phase 1
refactoring (v2.4), ensuring that the fundamental workflow of creating and
running a simple, single-step pipeline with an AnalysisComponent works as expected.
"""

"""
Unit and integration tests for the purified, component-based FuzzyPipeline.

This test file specifically validates the core functionality after the Phase 1
refactoring (v2.4), ensuring that the fundamental workflow of creating and
running a simple, single-step pipeline with an AnalysisComponent works as expected.
"""

import pytest
import pandas as pd
import numpy as np

# Core _components to be tested
from axisfuzzy.analysis._pipeline import FuzzyPipeline
from axisfuzzy.analysis.contracts import contract
from axisfuzzy.analysis._components.base import AnalysisComponent
from axisfuzzy.analysis.dataframe import FuzzyDataFrame

# Import accessor to ensure it's registered with pandas
import axisfuzzy.analysis.accessor


# --- Test Fixtures and Mock Components ---

@pytest.fixture
def crisp_input_data() -> pd.DataFrame:
    """Provides a simple, standard crisp DataFrame for pipeline input."""
    return pd.DataFrame({'attr1': [10, 20], 'attr2': [30, 40]}, index=['Alt_A', 'Alt_B'])


class SimpleMultiplierComponent(AnalysisComponent):
    """
    A minimal analysis component for testing purposes.
    It takes a DataFrame, multiplies all its values by a factor,
    and returns a new DataFrame.
    """

    def __init__(self, factor: float):
        """
        Initializes the component with a multiplication factor.

        Parameters
        ----------
        factor : float
            The number to multiply the data by.
        """
        self.factor = factor

    @contract(
        inputs={'data': 'CrispTable'},
        outputs={'result': 'CrispTable'}
    )
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        The contract-decorated method that performs the work.

        Parameters
        ----------
        data : pd.DataFrame
            The input crisp table.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with all values multiplied by the component's factor.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        return data * self.factor


# --- Test Cases ---

def test_component_instantiation_and_contract_metadata():
    """
    Tests that the component can be instantiated and its method has the
    correct contract metadata attached.
    """
    # 1. Instantiate the component
    multiplier = SimpleMultiplierComponent(factor=2.0)
    assert multiplier.factor == 2.0

    # 2. Verify that the @contract decorator worked
    method_to_test = multiplier.run
    assert hasattr(method_to_test, '_is_contract_method')
    assert getattr(method_to_test, '_is_contract_method') is True

    expected_inputs = {'data': 'CrispTable'}
    expected_outputs = {'result': 'CrispTable'}
    assert hasattr(method_to_test, '_contract_inputs')
    assert getattr(method_to_test, '_contract_inputs') == expected_inputs
    assert hasattr(method_to_test, '_contract_outputs')
    assert getattr(method_to_test, '_contract_outputs') == expected_outputs

    print("\n✅ [Phase 1] Test: Component contract metadata is correctly attached.")


def test_single_step_pipeline_direct_run(crisp_input_data):
    """
    Tests building and running a single-step pipeline directly via FuzzyPipeline.run().
    """
    # 1. Instantiate the component
    multiplier = SimpleMultiplierComponent(factor=3.0)

    # 2. Build the pipeline with a single step
    p = FuzzyPipeline()
    init_data_node = p.input("my_data", contract="CrispTable")
    result_node = p.add(multiplier.run, data=init_data_node)

    # 3. Check the pipeline structure (optional but good for debugging)
    assert len(p._steps) == 2  # One input node, one task node
    assert len(p._input_nodes) == 1
    assert result_node.pipeline is p

    # 4. Execute the pipeline
    final_result = p.run({'my_data': crisp_input_data})

    # 5. Validate the result
    assert isinstance(final_result, pd.DataFrame)

    # [FIXED] Change integer literals to float literals to match the expected dtype.
    expected_df = pd.DataFrame(
        {'attr1': [30.0, 60.0], 'attr2': [90.0, 120.0]},
        index=['Alt_A', 'Alt_B']
    )

    pd.testing.assert_frame_equal(final_result, expected_df)

    print("\n✅ [Phase 1] Test: Single-step pipeline runs correctly via direct call.")


def test_single_step_pipeline_via_accessor(crisp_input_data):
    """
    Tests the full user-facing workflow using the pandas accessor df.fuzzy.run().
    """
    # 1. Instantiate the component
    multiplier = SimpleMultiplierComponent(factor=0.5)

    # 2. Build the pipeline
    p = FuzzyPipeline()
    # When using the accessor with a single-input pipeline, the input name
    # doesn't matter, but providing one is good practice.
    init_data_node = p.input("init_data", contract="CrispTable")
    p.add(multiplier.run, data=init_data_node)

    # 3. Execute via the accessor on the DataFrame
    final_result = crisp_input_data.fuzzy.run(p)

    # 4. Validate the result
    assert isinstance(final_result, pd.DataFrame)
    expected_df = pd.DataFrame({'attr1': [5.0, 10.0], 'attr2': [15.0, 20.0]}, index=['Alt_A', 'Alt_B'])
    pd.testing.assert_frame_equal(final_result, expected_df)

    print("\n✅ [Phase 1] Test: Single-step pipeline runs correctly via pandas accessor.")


def test_add_non_contract_method_raises_error():
    """
    Ensures that FuzzyPipeline.add() raises a TypeError if given a method
    that is not decorated with @contract.
    """

    class BadComponent(AnalysisComponent):
        def run(self, data):
            return data

    p = FuzzyPipeline()
    init_data = p.input()
    bad_component_instance = BadComponent()

    with pytest.raises(TypeError, match="must be a callable method decorated with @contract"):
        p.add(bad_component_instance.run, data=init_data)

    print("\n✅ [Phase 1] Test: Adding a non-contract method correctly raises TypeError.")


if __name__ == "__main__":
    # Run tests directly for quick feedback during development
    test_component_instantiation_and_contract_metadata()
    crisp_data = pd.DataFrame({'attr1': [10, 20], 'attr2': [30, 40]}, index=['Alt_A', 'Alt_B'])
    test_single_step_pipeline_direct_run(crisp_data)
    test_single_step_pipeline_via_accessor(crisp_data)
    test_add_non_contract_method_raises_error()
