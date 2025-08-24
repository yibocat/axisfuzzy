#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 20:42
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pytest
import pandas as pd
import numpy as np

from axisfuzzy.analysis._pipeline import FuzzyPipeline
from axisfuzzy.analysis._components.basic import (
    NormalizationTool, StatisticsTool, SimpleAggregationTool
)


# --- Test Fixtures ---

@pytest.fixture
def sample_dataframe():
    """Provides sample data for pipeline testing."""
    return pd.DataFrame({
        'A': [10, 20, 30],
        'B': [1, 2, 3]
    }, index=['R1', 'R2', 'R3'])


@pytest.fixture
def multi_branch_pipeline():
    """Creates a pipeline with multiple branches and a final aggregation step."""
    p = FuzzyPipeline()
    data_input = p.input("data")

    # Branch 1
    norm1 = p.add(NormalizationTool(method='min_max', axis=0).run, data=data_input)

    # Branch 2
    agg1 = p.add(SimpleAggregationTool(operation='sum', axis=1).run, data=data_input)

    # Merge step (depends on Branch 1 and 2)
    # Let's create a simple component for merging for the test
    from axisfuzzy.analysis._components.base import AnalysisComponent
    from axisfuzzy.analysis.contracts import contract, CrispTable, WeightVector

    class MergeTool(AnalysisComponent):
        @contract(inputs={'table': 'CrispTable', 'vector': 'WeightVector'}, outputs={'result': 'CrispTable'})
        def run(self, table: pd.DataFrame, vector: pd.Series):
            # A simple, deterministic operation for testing
            return {'result': table.mul(vector, axis=0)}

    merge_tool = MergeTool()
    merged = p.add(merge_tool.run, table=norm1, vector=agg1)

    return p


# --- Core Test Function ---

def test_execution_modes_consistency(multi_branch_pipeline, sample_dataframe):
    """
    Tests that all three execution modes (run, step_by_step, start_execution)
    produce identical final results.
    """
    p = multi_branch_pipeline
    data = {"data": sample_dataframe}

    # --- Mode 1: Direct run() ---
    final_result_run, intermediate_run = p.run(data, return_intermediate=True)

    # --- Mode 2: step_by_step() iterator ---
    iterator = p.step_by_step(data)
    for _ in iterator:  # Exhaust the iterator
        pass
    intermediate_iterator = iterator.get_current_state_dict()
    final_state_iterator = iterator.current_state
    final_result_iterator = p._format_final_output(final_state_iterator)

    # --- Mode 3: start_execution() chainable state ---
    state = p.start_execution(data)
    final_state_chain = state.run_all()
    intermediate_chain = final_state_chain.step_results
    final_result_chain = p._format_final_output(final_state_chain)

    # --- Consistency Checks ---
    assert intermediate_run.keys() == intermediate_iterator.keys() == intermediate_chain.keys()

    for step_id in intermediate_run:
        res_run = intermediate_run[step_id]
        res_iter = intermediate_iterator[step_id]
        res_chain = intermediate_chain[step_id]

        # FIX: Check if the result is a dictionary (from a component)
        if isinstance(res_run, dict):
            assert res_run.keys() == res_iter.keys() == res_chain.keys()
            for key in res_run:
                val_run = res_run[key]
                val_iter = res_iter[key]
                val_chain = res_chain[key]

                # FIX: Use the correct asserter based on the value's type
                if isinstance(val_run, pd.DataFrame):
                    pd.testing.assert_frame_equal(val_run, val_iter)
                    pd.testing.assert_frame_equal(val_run, val_chain)
                elif isinstance(val_run, pd.Series):
                    pd.testing.assert_series_equal(val_run, val_iter)
                    pd.testing.assert_series_equal(val_run, val_chain)
                elif isinstance(val_run, np.ndarray):
                    np.testing.assert_allclose(val_run, val_iter)
                    np.testing.assert_allclose(val_run, val_chain)
                else:
                    assert val_run == val_iter == val_chain
        # Handle cases where the result is not a dict (e.g., input data)
        elif isinstance(res_run, pd.DataFrame):
            pd.testing.assert_frame_equal(res_run, res_iter)
            pd.testing.assert_frame_equal(res_run, res_chain)
        else:
             assert res_run == res_iter == res_chain

    # Check final results are identical
    pd.testing.assert_frame_equal(final_result_run['result'], final_result_iterator['result'])
    pd.testing.assert_frame_equal(final_result_run['result'], final_result_chain['result'])


def test_manual_next_on_iterator(multi_branch_pipeline, sample_dataframe):
    """
    Tests manual `next()` calls on the FuzzyPipelineIterator.
    """
    p = multi_branch_pipeline
    data = {"data": sample_dataframe}

    iterator = p.step_by_step(data)

    # Step 1
    step1 = next(iterator)
    assert 'NormalizationTool' in step1['step_name']
    assert not iterator.is_complete()

    # Step 2
    step2 = next(iterator)
    assert 'SimpleAggregationTool' in step2['step_name']
    assert not iterator.is_complete()

    # Step 3
    step3 = next(iterator)
    assert 'MergeTool' in step3['step_name']
    assert iterator.is_complete()

    # Should raise StopIteration
    with pytest.raises(StopIteration):
        next(iterator)
