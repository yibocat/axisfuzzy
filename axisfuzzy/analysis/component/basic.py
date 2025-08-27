#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 20:18
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Basic utility components for data preprocessing and transformation.

This module provides simple, reusable components for common data operations
that are frequently needed in fuzzy analysis pipelines.
This version is refactored to use the new type-annotation-driven contract system.
"""
from typing import Union

import numpy as np
import pandas as pd

from axisfuzzy.fuzzifier import Fuzzifier

from ..contracts import contract
from ..build_in import (
    ContractCrispTable,
    ContractWeightVector,
    ContractStatisticsDict,
    ContractNormalizedWeights,
    ContractFuzzyTable
)

from .base import AnalysisComponent


class ToolNormalization(AnalysisComponent):
    """
    A component for normalizing numerical data using various methods.

    Parameters
    ----------
    method : str, default 'min_max'
        The normalization method to use. Options are:
        - 'min_max': Scale to [0, 1] range.
        - 'z_score': Standardize using (x - mean) / std.
        - 'sum': Normalize by dividing by the sum.
        - 'max': Normalize by dividing by the maximum.
    axis : int, default 1
        The axis along which to normalize (0 for columns, 1 for rows).
    """
    def __init__(self, method: str = 'min_max', axis: int = 1):
        valid_methods = ['min_max', 'z_score', 'sum', 'max']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.method = method
        self.axis = axis

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'method': self.method, 'axis': self.axis}

    @contract
    def run(self, data: ContractCrispTable) -> ContractCrispTable:
        """
        Normalizes the input data using the specified method.

        The contract is now defined directly in the function signature.

        Parameters
        ----------
        data : ContractCrispTable
            Input DataFrame with numerical data to normalize.

        Returns
        -------
        ContractCrispTable
            The normalized DataFrame.
        """
        # The implementation logic remains the same.
        normalized_data = data.copy().astype(float)

        if self.method == 'min_max':
            # ... (implementation is unchanged)
            if self.axis == 1:
                for idx in normalized_data.index:
                    row = normalized_data.loc[idx]
                    min_val, max_val = row.min(), row.max()
                    if max_val != min_val:
                        normalized_data.loc[idx] = (row - min_val) / (max_val - min_val)
                    else:
                        normalized_data.loc[idx] = 0.5
            else:
                for col in normalized_data.columns:
                    col_data = normalized_data[col]
                    min_val, max_val = col_data.min(), col_data.max()
                    if max_val != min_val:
                        normalized_data[col] = (col_data - min_val) / (max_val - min_val)
                    else:
                        normalized_data[col] = 0.5
        # ... (other methods like z_score, sum, max are unchanged)
        elif self.method == 'z_score':
            if self.axis == 1:
                for idx in normalized_data.index:
                    row = normalized_data.loc[idx]
                    mean_val, std_val = row.mean(), row.std()
                    if std_val != 0:
                        normalized_data.loc[idx] = (row - mean_val) / std_val
                    else:
                        normalized_data.loc[idx] = 0
            else:
                normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()
        elif self.method == 'sum':
            normalized_data = normalized_data.div(normalized_data.sum(axis=self.axis), axis=1-self.axis)
        elif self.method == 'max':
            normalized_data = normalized_data.div(normalized_data.max(axis=self.axis), axis=1-self.axis)

        # The pipeline will handle single-output unpacking.
        return normalized_data


class ToolWeightNormalization(AnalysisComponent):
    """
    A component for normalizing weight vectors to sum to 1.0.

    Parameters
    ----------
    ensure_positive : bool, default True
        If True, negative weights are set to zero before normalization.
    """
    def __init__(self, ensure_positive: bool = True):
        self.ensure_positive = ensure_positive

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'ensure_positive': self.ensure_positive}

    @contract
    def run(self, weights: ContractWeightVector) -> ContractNormalizedWeights:
        """
        Normalizes weights to sum to 1.0.

        Parameters
        ----------
        weights : ContractWeightVector
            Input weights to normalize.

        Returns
        -------
        ContractNormalizedWeights
            The normalized weights, guaranteed to sum to 1.0.
        """
        if isinstance(weights, pd.Series):
            weights_array = weights.values
            return_series = True
            index = weights.index
        else:
            weights_array = np.array(weights)
            return_series = False
            index = None

        if self.ensure_positive:
            weights_array = np.maximum(weights_array, 0)

        total = np.sum(weights_array)
        if total == 0:
            normalized = np.full_like(weights_array, 1.0 / len(weights_array), dtype=float)
        else:
            normalized = weights_array / total

        if return_series:
            return pd.Series(normalized, index=index)
        else:
            return normalized


class ToolStatistics(AnalysisComponent):
    """
    A component for calculating basic statistical summaries.

    Parameters
    ----------
    axis : int, default 1
        The axis along which to calculate statistics (0 for columns, 1 for rows).
        Note: Current implementation flattens the data, so axis is for future use.
    """
    def __init__(self, axis: int = 1):
        self.axis = axis

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'axis': self.axis}

    @contract
    def run(self, data: ContractCrispTable) -> ContractStatisticsDict:
        """
        Calculates statistical summary of the data.

        Parameters
        ----------
        data : ContractCrispTable
            Input DataFrame for statistical analysis.

        Returns
        -------
        ContractStatisticsDict
            A dictionary containing statistical summaries.
        """
        values = data.values.flatten()
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }
        return stats


class ToolSimpleAggregation(AnalysisComponent):
    """
    A component for simple aggregation operations.

    Parameters
    ----------
    operation : str, default 'mean'
        The aggregation operation. Options: 'mean', 'sum', 'max', 'min', 'std'.
    axis : int, default 1
        The axis along which to aggregate (0 for columns, 1 for rows).
    """
    def __init__(self, operation: str = 'mean', axis: int = 1):
        valid_ops = ['mean', 'sum', 'max', 'min', 'std']
        if operation not in valid_ops:
            raise ValueError(f"Operation must be one of {valid_ops}")
        self.operation = operation
        self.axis = axis

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'operation': self.operation, 'axis': self.axis}

    @contract
    def run(self, data: ContractCrispTable) -> ContractWeightVector:
        """
        Aggregates data using the specified operation.

        Parameters
        ----------
        data : ContractCrispTable
            Input DataFrame to aggregate.

        Returns
        -------
        ContractWeightVector
            A pandas Series or numpy array of aggregated values.
        """
        agg_func = getattr(data, self.operation)
        result = agg_func(axis=self.axis)
        return result


class ToolFuzzification(AnalysisComponent):
    """
    An analysis component for converting crisp data into fuzzy data.

    This component wraps the core `axisfuzzy.fuzzifier.Fuzzifier` engine,
    allowing it to be seamlessly integrated into an analysis pipeline. It is
    configured during instantiation with all necessary fuzzification parameters.

    Parameters
    ----------
    fuzzifier : Fuzzifier
        A pre-configured instance of the `Fuzzifier` class. This approach
        promotes separation of concerns, where the fuzzification logic is
        defined once and then passed to this pipeline component.
    """
    def __init__(self, fuzzifier: Union[Fuzzifier, dict]):

        if isinstance(fuzzifier, Fuzzifier):
            self.fuzzifier = fuzzifier
        elif isinstance(fuzzifier, dict):
            self.fuzzifier = Fuzzifier.from_config(fuzzifier)
        else:
            raise TypeError(
                f"fuzzifier must be an instance of 'Fuzzifier' or a config dict. "
                f"Got '{type(fuzzifier)}'."
            )

    def get_config(self):
        """
        Returns the component's configuration by requesting it from the inner Fuzzifier.
        """
        # 职责清晰：ToolFuzzification 只负责向 Fuzzifier 请求其配置
        return {'fuzzifier': self.fuzzifier.get_config()}

    @contract
    def run(self, data: ContractCrispTable) -> ContractFuzzyTable:
        """
        Executes the fuzzification process on the input data.

        Parameters
        ----------
        data : ContractCrispTable
            A pandas DataFrame with crisp, numerical data.

        Returns
        -------
        ContractFuzzyTable
            A FuzzyDataFrame containing the fuzzy representation of the data.
        """
        from ..dataframe import FuzzyDataFrame
        return FuzzyDataFrame.from_pandas(data, self.fuzzifier)
