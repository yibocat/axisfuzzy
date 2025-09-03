#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 20:59
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Pytest tests for FuzzyPipeline's I/O capabilities.
This file validates the scenarios demonstrated in demo_pipeline_io.py.
"""
from typing import Dict

import pytest
import numpy as np
import pandas as pd

from axisfuzzy.analysis.component import AnalysisComponent
# Assuming the demo file is in the same directory or accessible via PYTHONPATH
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.component.basic import ToolNormalization
from axisfuzzy.analysis.contracts import contract
from axisfuzzy.analysis.build_in import (
    ContractCrispTable,
    ContractWeightVector, ContractScoreVector
)


class ToolSplitter(AnalysisComponent):
    """
    A test component that takes one table and splits it into two vectors.
    Demonstrates: Single Input -> Multiple Outputs.
    """

    def get_config(self) -> dict:
        return {}

    @contract
    def run(self, data: ContractCrispTable) -> Dict[str, ContractWeightVector]:
        """
        Splits a DataFrame into the first row and the first column.

        Parameters
        ----------
        data : ContractCrispTable
            Input DataFrame.

        Returns
        -------
        Dict[str, ContractWeightVector]
            A dictionary containing two outputs: 'first_row' and 'first_col'.
        """
        return {
            'first_row': data.iloc[0, :],
            'first_col': data.iloc[:, 0]
        }


class ToolCombiner(AnalysisComponent):
    """
    A test component that combines two vectors into a single score vector.
    Demonstrates: Multiple Inputs -> Single Output.
    """

    def get_config(self) -> dict:
        return {}

    @contract
    def run(self, vector_a: ContractWeightVector, vector_b: ContractWeightVector) -> ContractScoreVector:
        """
        Combines two vectors by element-wise addition.

        Parameters
        ----------
        vector_a : ContractWeightVector
            The first input vector.
        vector_b : ContractWeightVector
            The second input vector.

        Returns
        -------
        ContractScoreVector
            The resulting combined vector.
        """
        # Ensure they are numpy arrays for robust addition
        vec_a = pd.Series(vector_a).values
        vec_b = pd.Series(vector_b).values
        return vec_a + vec_b


@pytest.fixture
def sample_data():
    """Provides common data for tests."""
    df1 = pd.DataFrame(
        np.arange(9).reshape(3, 3),
        columns=['A', 'B', 'C'],
        index=['X', 'Y', 'Z']
    )
    df2 = pd.DataFrame(
        np.random.rand(3, 3) * 10,
        columns=['A', 'B', 'C'],
        index=['X', 'Y', 'Z']
    )
    weights = pd.Series([0.1, 0.5, 0.4], index=['A', 'B', 'C'])
    return {"df1": df1, "df2": df2, "weights": weights}


def test_single_input_single_output(sample_data):
    """Tests Scenario A: 1 input -> 1 output."""
    p = FuzzyPipeline()
    norm_tool = ToolNormalization(method='sum', axis=1)
    input_node = p.input(contract=ContractCrispTable)
    p.add(norm_tool.run, data=input_node)

    result = p.run(sample_data['df1'])

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    # Check if rows sum to 1 (or are close, due to float precision)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_single_input_multi_output(sample_data):
    """Tests Scenario B: 1 input -> 2 outputs."""
    p = FuzzyPipeline()
    splitter_tool = ToolSplitter()
    input_node = p.input(contract=ContractCrispTable)
    p.add(splitter_tool.run, data=input_node)

    result = p.run(sample_data['df1'])

    assert isinstance(result, dict)
    assert 'first_row' in result
    assert 'first_col' in result
    assert isinstance(result['first_row'], pd.Series)
    assert isinstance(result['first_col'], pd.Series)
    assert len(result['first_row']) == 3
    pd.testing.assert_series_equal(result['first_row'], sample_data['df1'].iloc[0, :])


def test_multi_input_single_output(sample_data):
    """Tests Scenario C: 2 inputs -> 1 output."""
    p = FuzzyPipeline()
    combiner_tool = ToolCombiner()
    input1 = p.input(name="vec1", contract=ContractWeightVector)
    input2 = p.input(name="vec2", contract=ContractWeightVector)
    p.add(combiner_tool.run, vector_a=input1, vector_b=input2)

    df1_row0 = sample_data['df1'].iloc[0, :]
    weights = sample_data['weights']
    result = p.run({"vec1": df1_row0, "vec2": weights})

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = df1_row0.values + weights.values
    assert np.allclose(result, expected)


def test_multi_input_multi_output(sample_data):
    """Tests Scenario D: 2 inputs -> 2 outputs."""
    p = FuzzyPipeline()
    norm_tool = ToolNormalization(method='max', axis=0)
    splitter_tool = ToolSplitter()

    input1 = p.input(name="main_data", contract=ContractCrispTable)
    input2 = p.input(name="secondary_data", contract=ContractCrispTable)

    # These two `add` calls create two terminal nodes
    p.add(norm_tool.run, data=input1)
    p.add(splitter_tool.run, data=input2)

    result = p.run({
        "main_data": sample_data['df1'],
        "secondary_data": sample_data['df2']
    })

    assert isinstance(result, dict)
    # The keys are derived from the step's display name
    assert len(result.keys()) == 2
    key1 = 'ToolNormalization.run'
    key2 = 'ToolSplitter.run'
    # Find keys that start with the expected names
    result_keys = list(result.keys())
    assert any(k.startswith(key1) for k in result_keys)
    assert any(k.startswith(key2) for k in result_keys)

    norm_result_key = next(k for k in result_keys if k.startswith(key1))
    split_result_key = next(k for k in result_keys if k.startswith(key2))

    assert isinstance(result[norm_result_key], pd.DataFrame)
    assert isinstance(result[split_result_key], dict)
    assert 'first_row' in result[split_result_key]
