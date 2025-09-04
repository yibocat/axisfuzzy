#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared fixtures and configuration for axisfuzzy.analysis tests.

This module provides common test fixtures, utilities, and configuration
that are used across multiple test modules in the analysis test suite.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import analysis components for fixtures
try:
    from axisfuzzy.fuzzifier import Fuzzifier
    from axisfuzzy.analysis.component.basic import (
        ToolNormalization,
        ToolFuzzification,
        ToolSimpleAggregation,
        ToolWeightNormalization
    )
    from axisfuzzy.analysis.pipeline import FuzzyPipeline
    from axisfuzzy.analysis.app.model import Model
    from axisfuzzy.analysis.build_in import (
        ContractCrispTable,
        ContractWeightVector,
        ContractFuzzyTable
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Skip all tests if analysis module is not available
pytestmark = pytest.mark.skipif(
    not ANALYSIS_AVAILABLE,
    reason=f"Analysis module not available: {IMPORT_ERROR if not ANALYSIS_AVAILABLE else ''}"
)


# ========================= Data Fixtures =========================

@pytest.fixture
def sample_dataframe():
    """
    Standard test DataFrame with numerical data.
    
    Returns
    -------
    pd.DataFrame
        A 5x3 DataFrame with random numerical values.
    """
    np.random.seed(42)  # For reproducible tests
    return pd.DataFrame(
        np.random.rand(5, 3),
        columns=['Feature_A', 'Feature_B', 'Feature_C'],
        index=[f'Sample_{i}' for i in range(5)]
    )


@pytest.fixture
def sample_weights():
    """
    Standard test weight vector.
    
    Returns
    -------
    np.ndarray
        A weight vector with 3 elements.
    """
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def large_dataframe():
    """
    Larger test DataFrame for performance testing.
    
    Returns
    -------
    pd.DataFrame
        A 100x10 DataFrame with random numerical values.
    """
    np.random.seed(123)
    return pd.DataFrame(
        np.random.rand(100, 10),
        columns=[f'Feature_{i}' for i in range(10)]
    )


@pytest.fixture
def mixed_dataframe():
    """
    DataFrame with mixed data types for error testing.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing both numerical and non-numerical data.
    """
    return pd.DataFrame({
        'numeric_col': [1.0, 2.0, 3.0],
        'string_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })


# ========================= Component Fixtures =========================

@pytest.fixture
def sample_fuzzifier():
    """
    Standard test fuzzifier.
    
    Returns
    -------
    Fuzzifier
        A configured fuzzifier using Gaussian membership function.
    """
    return Fuzzifier(
        mf='gaussmf',
        mtype='qrofn',
        pi=0.2,
        mf_params=[{'sigma': 0.15, 'c': 0.5}]
    )


@pytest.fixture
def normalizer_component():
    """
    Standard normalization component.
    
    Returns
    -------
    ToolNormalization
        A min-max normalization component.
    """
    return ToolNormalization(method='min_max', axis=0)


@pytest.fixture
def fuzzification_component(sample_fuzzifier):
    """
    Standard fuzzification component.
    
    Returns
    -------
    ToolFuzzification
        A fuzzification component with sample fuzzifier.
    """
    return ToolFuzzification(fuzzifier=sample_fuzzifier)


@pytest.fixture
def aggregation_component():
    """
    Standard aggregation component.
    
    Returns
    -------
    ToolSimpleAggregation
        A mean aggregation component.
    """
    return ToolSimpleAggregation(operation='mean', axis=1)


@pytest.fixture
def weight_normalizer_component():
    """
    Standard weight normalization component.
    
    Returns
    -------
    ToolWeightNormalization
        A weight normalization component.
    """
    return ToolWeightNormalization()


# ========================= Pipeline Fixtures =========================

@pytest.fixture
def empty_pipeline():
    """
    Empty pipeline for testing.
    
    Returns
    -------
    FuzzyPipeline
        An empty pipeline instance.
    """
    return FuzzyPipeline(name="TestPipeline")


@pytest.fixture
def basic_pipeline(normalizer_component):
    """
    Basic pipeline with one component.
    
    Returns
    -------
    FuzzyPipeline
        A pipeline with a single normalization step.
    """
    pipeline = FuzzyPipeline(name="BasicPipeline")
    data_input = pipeline.input("data", contract=ContractCrispTable)
    pipeline.add(normalizer_component.run, data=data_input)
    return pipeline


@pytest.fixture
def complex_pipeline(normalizer_component, fuzzification_component, aggregation_component):
    """
    Complex pipeline with multiple components.
    
    Returns
    -------
    FuzzyPipeline
        A pipeline with normalization, fuzzification, and aggregation.
    """
    pipeline = FuzzyPipeline(name="ComplexPipeline")
    data_input = pipeline.input("data", contract=ContractCrispTable)
    
    norm_output = pipeline.add(normalizer_component.run, data=data_input)
    fuzz_output = pipeline.add(fuzzification_component.run, data=norm_output)
    pipeline.add(aggregation_component.run, data=fuzz_output)
    
    return pipeline


# ========================= Model Fixtures =========================

class SimpleTestModel(Model):
    """
    Simple test model for testing purposes.
    """
    
    def __init__(self, fuzzifier):
        super().__init__()
        self.normalizer = ToolNormalization(method='min_max')
        self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)
    
    def get_config(self):
        return {"fuzzifier": self.fuzzifier.get_config()}
    
    def forward(self, data: ContractCrispTable) -> ContractFuzzyTable:
        norm_data = self.normalizer(data)
        fuzzy_data = self.fuzzifier(norm_data)
        return fuzzy_data


class ComplexTestModel(Model):
    """
    Complex test model with multiple inputs and outputs.
    """
    
    def __init__(self, fuzzifier):
        super().__init__()
        self.data_normalizer = ToolNormalization(method='min_max')
        self.weight_normalizer = ToolWeightNormalization()
        self.fuzzifier = ToolFuzzification(fuzzifier=fuzzifier)
        self.aggregator = ToolSimpleAggregation(operation='mean', axis=1)
    
    def get_config(self):
        return {"fuzzifier": self.fuzzifier.get_config()}
    
    def forward(self, data: ContractCrispTable, weights: ContractWeightVector):
        norm_data = self.data_normalizer(data)
        fuzzy_data = self.fuzzifier(norm_data)
        norm_weights = self.weight_normalizer(weights)
        scores = self.aggregator(fuzzy_data)
        
        return {
            'scores': scores,
            'fuzzy_data': fuzzy_data,
            'normalized_weights': norm_weights
        }


@pytest.fixture
def simple_model(sample_fuzzifier):
    """
    Simple test model instance.
    
    Returns
    -------
    SimpleTestModel
        A simple model for testing.
    """
    return SimpleTestModel(sample_fuzzifier)


@pytest.fixture
def complex_model(sample_fuzzifier):
    """
    Complex test model instance.
    
    Returns
    -------
    ComplexTestModel
        A complex model for testing.
    """
    return ComplexTestModel(sample_fuzzifier)


# ========================= Utility Fixtures =========================

@pytest.fixture
def temp_model_file(tmp_path):
    """
    Temporary file path for model serialization testing.
    
    Returns
    -------
    Path
        A temporary file path for saving models.
    """
    return tmp_path / "test_model.json"


@pytest.fixture
def execution_results():
    """
    Sample execution results for testing.
    
    Returns
    -------
    dict
        Sample execution results dictionary.
    """
    return {
        'step_1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        'step_2': np.array([0.5, 0.3, 0.2]),
        'final': {'score': 0.75, 'confidence': 0.9}
    }


# ========================= Test Utilities =========================

def assert_dataframe_equal(df1, df2, check_dtype=True, check_index=True):
    """
    Enhanced DataFrame comparison with better error messages.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to compare.
    check_dtype : bool
        Whether to check data types.
    check_index : bool
        Whether to check index equality.
    """
    try:
        pd.testing.assert_frame_equal(
            df1, df2, 
            check_dtype=check_dtype, 
            check_index=check_index
        )
    except AssertionError as e:
        # Add more context to the error
        raise AssertionError(
            f"DataFrames are not equal:\n"
            f"Shape 1: {df1.shape}, Shape 2: {df2.shape}\n"
            f"Columns 1: {list(df1.columns)}, Columns 2: {list(df2.columns)}\n"
            f"Original error: {str(e)}"
        )


def assert_contract_valid(contract, data):
    """
    Assert that data satisfies a contract.
    
    Parameters
    ----------
    contract : Contract
        The contract to validate against.
    data : Any
        The data to validate.
    """
    assert contract.validate(data), (
        f"Data does not satisfy contract '{contract.name}'. "
        f"Data type: {type(data)}, Data: {data}"
    )


# ========================= Markers =========================

# Custom pytest markers for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_optional_deps = pytest.mark.requires_optional_deps