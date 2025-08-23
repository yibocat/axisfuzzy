#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 16:47
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Basic utility components for data preprocessing and transformation.

This module provides simple, reusable components for common data operations
that are frequently needed in fuzzy analysis pipelines.
"""

from typing import Dict, Union, List
import numpy as np
import pandas as pd

from .base import AnalysisComponent
from ..contracts import contract, CrispTable, Matrix, WeightVector, NormalizedWeights, StatisticsDict


class NormalizationTool(AnalysisComponent):
    """
    A component for normalizing numerical data using various methods.

    This tool provides common data normalization techniques including min-max
    scaling, z-score normalization, and sum normalization.

    Parameters
    ----------
    method : str, default 'min_max'
        The normalization method to use. Options are:
        - 'min_max': Scale to [0, 1] range using (x - min) / (max - min)
        - 'z_score': Standardize using (x - mean) / std
        - 'sum': Normalize by dividing each value by the sum of all values
        - 'max': Normalize by dividing each value by the maximum value
    axis : int, default 1
        The axis along which to normalize (0 for columns, 1 for rows).
    """
    def __init__(self, method: str = 'min_max', axis: int = 1):
        valid_methods = ['min_max', 'z_score', 'sum', 'max']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.method = method
        self.axis = axis

    @contract(
        inputs={'data': 'CrispTable'},
        outputs={'normalized_data': 'CrispTable'}
    )
    def run(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Normalizes the input data using the specified method.

        Parameters
        ----------
        data : CrispTable
            Input DataFrame with numerical data to normalize.

        Returns
        -------
        dict[str, CrispTable]
            A dictionary with 'normalized_data' containing the normalized DataFrame.

        Examples
        --------
        .. code-block:: python

            import pandas as pd
            from axisfuzzy.analysis.components.basic import NormalizationTool

            data = pd.DataFrame({
                'Cost': [100, 200, 150],
                'Quality': [8, 9, 7],
                'Speed': [50, 60, 55]
            })

            normalizer = NormalizationTool(method='min_max')
            result = normalizer.run(data=data)
            print(result['normalized_data'])
        """
        normalized_data = data.copy().astype(float)

        if self.method == 'min_max':
            if self.axis == 1:  # Row-wise normalization
                for idx in normalized_data.index:
                    row = normalized_data.loc[idx]
                    min_val, max_val = row.min(), row.max()
                    if max_val != min_val:
                        normalized_data.loc[idx] = (row - min_val) / (max_val - min_val)
                    else:
                        normalized_data.loc[idx] = 0.5  # All values are the same
            else:  # Column-wise normalization
                for col in normalized_data.columns:
                    col_data = normalized_data[col]
                    min_val, max_val = col_data.min(), col_data.max()
                    if max_val != min_val:
                        normalized_data[col] = (col_data - min_val) / (max_val - min_val)
                    else:
                        normalized_data[col] = 0.5

        elif self.method == 'z_score':
            if self.axis == 1:  # Row-wise
                for idx in normalized_data.index:
                    row = normalized_data.loc[idx]
                    mean_val, std_val = row.mean(), row.std()
                    if std_val != 0:
                        normalized_data.loc[idx] = (row - mean_val) / std_val
                    else:
                        normalized_data.loc[idx] = 0
            else:  # Column-wise
                normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()

        elif self.method == 'sum':
            if self.axis == 1:  # Row-wise
                row_sums = normalized_data.sum(axis=1)
                normalized_data = normalized_data.div(row_sums, axis=0)
            else:  # Column-wise
                col_sums = normalized_data.sum(axis=0)
                normalized_data = normalized_data / col_sums

        elif self.method == 'max':
            if self.axis == 1:  # Row-wise
                row_max = normalized_data.max(axis=1)
                normalized_data = normalized_data.div(row_max, axis=0)
            else:  # Column-wise
                col_max = normalized_data.max(axis=0)
                normalized_data = normalized_data / col_max

        return {'normalized_data': normalized_data}


class WeightNormalizationTool(AnalysisComponent):
    """
    A component for normalizing weight vectors to sum to 1.0.

    Parameters
    ----------
    ensure_positive : bool, default True
        If True, negative weights are set to zero before normalization.
    """
    def __init__(self, ensure_positive: bool = True):
        self.ensure_positive = ensure_positive

    @contract(
        inputs={'weights': 'WeightVector'},
        outputs={'normalized_weights': 'NormalizedWeights'}
    )
    def run(self, weights: WeightVector) -> Dict[str, NormalizedWeights]:
        """
        Normalizes weights to sum to 1.0.

        Parameters
        ----------
        weights : WeightVector
            Input weights to normalize.

        Returns
        -------
        dict[str, NormalizedWeights]
            A dictionary with 'normalized_weights'.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            from axisfuzzy.analysis.components.basic import WeightNormalizationTool

            weights = np.array([3, 2, 5])  # Sum = 10
            normalizer = WeightNormalizationTool()
            result = normalizer.run(weights=weights)
            print(result['normalized_weights'])  # [0.3, 0.2, 0.5]
        """
        if isinstance(weights, pd.Series):
            weights_array = weights.values
            return_series = True
            index = weights.index
        else:
            weights_array = np.array(weights)
            return_series = False
            index = None

        # Ensure positive if requested
        if self.ensure_positive:
            weights_array = np.maximum(weights_array, 0)

        # Normalize
        total = np.sum(weights_array)
        if total == 0:
            # All weights are zero, return equal weights
            normalized = np.full_like(weights_array, 1.0 / len(weights_array))
        else:
            normalized = weights_array / total

        if return_series:
            return {'normalized_weights': pd.Series(normalized, index=index)}
        else:
            return {'normalized_weights': normalized}


class StatisticsTool(AnalysisComponent):
    """
    A component for calculating basic statistical summaries.

    Parameters
    ----------
    axis : int, default 1
        The axis along which to calculate statistics (0 for columns, 1 for rows).
    """
    def __init__(self, axis: int = 1):
        self.axis = axis

    @contract(
        inputs={'data': 'CrispTable'},
        outputs={'statistics': 'StatisticsDict'}
    )
    def run(self, data: pd.DataFrame) -> Dict[str, StatisticsDict]:
        """
        Calculates statistical summary of the data.

        Parameters
        ----------
        data : CrispTable
            Input DataFrame for statistical analysis.

        Returns
        -------
        dict[str, StatisticsDict]
            A dictionary containing statistical summaries.

        Examples
        --------
        .. code-block:: python

            import pandas as pd
            from axisfuzzy.analysis.components.basic import StatisticsTool

            data = pd.DataFrame({
                'A': [1, 2, 3, 4, 5],
                'B': [2, 4, 6, 8, 10]
            })

            stats_tool = StatisticsTool()
            result = stats_tool.run(data=data)
            print(result['statistics'])
        """
        if self.axis == 1:
            # Row-wise statistics (overall statistics across all values)
            values = data.values.flatten()
        else:
            # Column-wise statistics (will be implemented as overall stats for simplicity)
            values = data.values.flatten()

        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }

        return {'statistics': stats}


class SimpleAggregationTool(AnalysisComponent):
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

    @contract(
        inputs={'data': 'CrispTable'},
        outputs={'aggregated_values': 'WeightVector'}
    )
    def run(self, data: pd.DataFrame) -> Dict[str, WeightVector]:
        """
        Aggregates data using the specified operation.

        Parameters
        ----------
        data : CrispTable
            Input DataFrame to aggregate.

        Returns
        -------
        dict[str, WeightVector]
            A dictionary with aggregated values.
        """
        if self.axis == 1:
            # Row-wise aggregation
            if self.operation == 'mean':
                result = data.mean(axis=1)
            elif self.operation == 'sum':
                result = data.sum(axis=1)
            elif self.operation == 'max':
                result = data.max(axis=1)
            elif self.operation == 'min':
                result = data.min(axis=1)
            elif self.operation == 'std':
                result = data.std(axis=1)
        else:
            # Column-wise aggregation
            if self.operation == 'mean':
                result = data.mean(axis=0)
            elif self.operation == 'sum':
                result = data.sum(axis=0)
            elif self.operation == 'max':
                result = data.max(axis=0)
            elif self.operation == 'min':
                result = data.min(axis=0)
            elif self.operation == 'std':
                result = data.std(axis=0)

        return {'aggregated_values': result}
