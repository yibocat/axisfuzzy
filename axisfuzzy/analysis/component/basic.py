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
from __future__ import annotations
from typing import Union

# Lazy import strategy: Avoid directly importing optional dependencies at the module level
# This ensures that the core package installation won't result in import errors due to missing optional dependencies.
try:
    import numpy as np
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    np = None
    pd = None
    _PANDAS_AVAILABLE = False

from axisfuzzy.fuzzifier import Fuzzifier

from ..contracts import contract
from ..build_in import (
    ContractCrispTable,
    ContractWeightVector,
    ContractStatisticsDict,
    ContractNormalizedWeights,
    ContractFuzzyTable,
    ContractThreeWayResult
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
        if not _PANDAS_AVAILABLE or pd is None:
            raise ImportError(
                "pandas is not installed. ToolNormalization requires pandas. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
        
        # The implementation logic remains the same.
        normalized_data = data.copy().astype(float)

        if self.method == 'min_max':
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
        if not _PANDAS_AVAILABLE or np is None:
            raise ImportError(
                "numpy is not installed. ToolWeightNormalization requires numpy. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
        
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
        if not _PANDAS_AVAILABLE or pd is None:
            raise ImportError(
                "pandas is not installed. ToolStatistics requires pandas. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
        
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
        if not _PANDAS_AVAILABLE or pd is None:
            raise ImportError(
                "pandas is not installed. ToolSimpleAggregation requires pandas. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
        
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


class ToolThreeWayDecision(AnalysisComponent):
    """
    A component for three-way decision analysis based on crisp numerical input.
    
    This component implements the classical three-way decision theory, which
    divides data into three regions: Accept, Boundary (Defer), and Reject.
    
    Parameters
    ----------
    alpha : float, default 0.7
        The acceptance threshold. Data above this value is classified into the accept region.
    beta : float, default 0.3  
        The rejection threshold. Data below this value is classified into the reject region.
        Data between beta and alpha is classified into the boundary region.
    decision_column : str, optional
        The column name to use for decision making. If None, uses the mean of all columns.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, decision_column: str = None):
        if alpha <= beta:
            raise ValueError("Alpha threshold must be greater than beta threshold")
        if not (0 <= beta < alpha <= 1):
            raise ValueError("Thresholds must be between 0 and 1, with beta < alpha")
            
        self.alpha = alpha
        self.beta = beta
        self.decision_column = decision_column

    def get_config(self) -> dict:
        """Return the component's configuration."""
        return {
            'alpha': self.alpha,
            'beta': self.beta, 
            'decision_column': self.decision_column
        }

    @contract
    def run(self, data: ContractCrispTable) -> ContractThreeWayResult:
        """
        Execute three-way decision analysis.
        
        Parameters
        ----------
        data : ContractCrispTable
            Input crisp numerical data table.
            
        Returns
        -------
        ContractThreeWayResult
            Dictionary containing three-way decision results with 'accept', 'reject', 'defer' keys.
            Each key maps to a list of row indices classified into that region.
        """
        if not _PANDAS_AVAILABLE or pd is None:
            raise ImportError(
                "pandas and numpy are not installed. ToolThreeWayDecision requires these dependencies. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
            
        # Ensure data is numeric type
        data_numeric = data.astype(float)
        
        # Calculate decision values
        if self.decision_column is not None:
            if self.decision_column not in data_numeric.columns:
                raise ValueError(f"Decision column '{self.decision_column}' not found in data")
            decision_values = data_numeric[self.decision_column]
        else:
            # Use mean of all columns as decision basis
            decision_values = data_numeric.mean(axis=1)
        
        # Initialize lists for three regions
        accept_indices = []
        reject_indices = []
        defer_indices = []
        
        # Classify based on thresholds
        for idx in data_numeric.index:
            value = decision_values.loc[idx] if hasattr(decision_values, 'loc') else decision_values[idx]
            if value >= self.alpha:
                accept_indices.append(idx)
            elif value <= self.beta:
                reject_indices.append(idx)
            else:
                defer_indices.append(idx)
        
        # Return three-way decision results
        return {
            'accept': accept_indices,
            'reject': reject_indices, 
            'defer': defer_indices
        }


class ToolAdvancedThreeWayDecision(AnalysisComponent):
    """
    An advanced three-way decision component supporting multi-criteria decision making and weight allocation.
    
    This component extends basic three-way decision to support multi-criteria weighted evaluation,
    allowing different weights to be assigned to different criteria for more flexible decision making.
    
    Parameters
    ----------
    alpha : float, default 0.7
        The acceptance threshold.
    beta : float, default 0.3
        The rejection threshold.  
    weights : array-like, optional
        Weights for each criterion. If None, equal weights will be used.
    aggregation_method : str, default 'weighted_average'
        Aggregation method: 'weighted_average', 'weighted_geometric_mean', or 'topsis_like'.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, 
                 weights: Union[list, np.ndarray] = None,
                 aggregation_method: str = 'weighted_average'):
        
        if alpha <= beta:
            raise ValueError("Alpha threshold must be greater than beta threshold")
        if not (0 <= beta < alpha <= 1):
            raise ValueError("Thresholds must be between 0 and 1, with beta < alpha")
            
        valid_methods = ['weighted_average', 'weighted_geometric_mean', 'topsis_like']
        if aggregation_method not in valid_methods:
            raise ValueError(f"Aggregation method must be one of {valid_methods}")
            
        self.alpha = alpha
        self.beta = beta
        self.weights = weights
        self.aggregation_method = aggregation_method

    def get_config(self) -> dict:
        """Return the component's configuration."""
        weights_config = None
        if self.weights is not None:
            if hasattr(self.weights, 'tolist'):
                weights_config = self.weights.tolist()
            else:
                weights_config = list(self.weights)
        
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'weights': weights_config,
            'aggregation_method': self.aggregation_method
        }

    @contract
    def run(self, data: ContractCrispTable, weights: ContractWeightVector = None) -> ContractThreeWayResult:
        """
        Execute advanced three-way decision analysis.
        
        Parameters
        ----------
        data : ContractCrispTable
            Input crisp numerical data table.
        weights : ContractWeightVector, optional
            Criterion weight vector. If provided, overrides the weights set during initialization.
            
        Returns
        -------
        ContractThreeWayResult
            Dictionary containing three-way decision results.
        """
        if not _PANDAS_AVAILABLE or pd is None:
            raise ImportError(
                "pandas and numpy are not installed. ToolAdvancedThreeWayDecision requires these dependencies. "
                "Please install with: pip install 'axisfuzzy[analysis]'"
            )
        
        # Ensure data is numeric type
        data_numeric = data.astype(float)
        
        # Determine weights to use
        if weights is not None:
            used_weights = np.array(weights)
        elif self.weights is not None:
            used_weights = np.array(self.weights)
        else:
            # Use equal weights
            used_weights = np.ones(data_numeric.shape[1]) / data_numeric.shape[1]
        
        # Validate weight dimensions
        if len(used_weights) != data_numeric.shape[1]:
            raise ValueError(f"Weight vector length ({len(used_weights)}) must match number of data columns ({data_numeric.shape[1]})")
        
        # Normalize weights
        used_weights = used_weights / np.sum(used_weights)
        
        # Calculate weighted decision values
        if self.aggregation_method == 'weighted_average':
            decision_values = (data_numeric * used_weights).sum(axis=1)
        elif self.aggregation_method == 'weighted_geometric_mean':
            # Geometric weighted mean
            decision_values = np.power(np.prod(data_numeric ** used_weights, axis=1), 1.0 / data_numeric.shape[1])
        elif self.aggregation_method == 'topsis_like':
            # TOPSIS-style evaluation
            # Calculate distances to positive and negative ideal solutions
            positive_ideal = data_numeric.max(axis=0)
            negative_ideal = data_numeric.min(axis=0)
            
            # Calculate distance to positive ideal solution
            pos_distance = np.sqrt(((data_numeric - positive_ideal) ** 2 * used_weights).sum(axis=1))
            # Calculate distance to negative ideal solution  
            neg_distance = np.sqrt(((data_numeric - negative_ideal) ** 2 * used_weights).sum(axis=1))
            
            # Relative closeness
            decision_values = neg_distance / (pos_distance + neg_distance + 1e-10)
        
        # Execute three-way decision classification
        accept_indices = []
        reject_indices = []
        defer_indices = []
        
        for idx in data_numeric.index:
            value = decision_values.loc[idx] if hasattr(decision_values, 'loc') else decision_values[idx]
            if value >= self.alpha:
                accept_indices.append(idx)
            elif value <= self.beta:
                reject_indices.append(idx)
            else:
                defer_indices.append(idx)
        
        return {
            'accept': accept_indices,
            'reject': reject_indices,
            'defer': defer_indices
        }