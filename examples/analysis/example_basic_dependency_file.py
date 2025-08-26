#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/25 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Dict

import pandas as pd
import numpy as np

from axisfuzzy.analysis.build_in import ContractWeightVector, ContractCrispTable, ContractStatisticsDict, \
    ContractNormalizedWeights
from axisfuzzy.analysis.component import AnalysisComponent
from axisfuzzy.analysis.contracts import contract
from axisfuzzy.analysis.contracts.base import Contract

# === Demo Contracts ===

# Additional demo contracts
ContractAccuracyVector = Contract(
    'ContractAccuracyVector',
    lambda obj: isinstance(obj, (pd.Series, np.ndarray)) and len(obj) > 0,
    parent=ContractWeightVector
)

ContractOrderResult = Contract(
    'ContractOrderResult',
    lambda obj: isinstance(obj, pd.Series) and obj.dtype in ['int64', 'float64']
)

ContractMultiMetrics = Contract(
    'ContractMultiMetrics',
    lambda obj: isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values())
)


# === Demo Components ===

class DemoDataGenerator(AnalysisComponent):
    """
    Generates sample data for testing pipeline modes.

    Parameters
    ----------
    rows : int, default 10
        Number of rows to generate.
    cols : int, default 3
        Number of columns to generate.
    """

    def __init__(self, rows: int = 10, cols: int = 3):
        self.rows = rows
        self.cols = cols

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'rows': self.rows, 'cols': self.cols}

    @contract
    def run(self) -> ContractCrispTable:
        """
        Generates random crisp data.

        Returns
        -------
        ContractCrispTable
            A DataFrame with random numerical data.
        """
        data = np.random.rand(self.rows, self.cols) * 100
        return pd.DataFrame(data, columns=[f'C{i + 1}' for i in range(self.cols)])


class DemoScoreCalculator(AnalysisComponent):
    """
    Calculates composite scores from data and weights.

    Parameters
    ----------
    score_method : str, default 'weighted_sum'
        Method for score calculation.
    """

    def __init__(self, score_method: str = 'weighted_sum'):
        valid_methods = ['weighted_sum', 'geometric_mean']
        if score_method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.score_method = score_method

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'score_method': self.score_method}

    @contract
    def run(self, data: ContractCrispTable, weights: ContractNormalizedWeights) -> ContractAccuracyVector:
        """
        Calculates composite scores.

        Parameters
        ----------
        data : ContractCrispTable
            Input data matrix.
        weights : ContractNormalizedWeights
            Normalized weights for criteria.

        Returns
        -------
        ContractScoreVector
            Composite scores for each alternative.
        """
        if self.score_method == 'weighted_sum':
            scores = (data * weights).sum(axis=1)
        else:  # geometric_mean
            scores = np.power(np.prod(data ** weights, axis=1), 1 / len(weights))

        return pd.Series(scores, name='composite_score', index=data.index)


class DemoMultiOutputAnalyzer(AnalysisComponent):
    """
    Performs multiple analyses simultaneously (multi-output demo).
    """

    def __init__(self):
        pass

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {}

    @contract
    def run(self, data: ContractCrispTable) -> Dict[str, ContractStatisticsDict]:
        """
        Performs multiple statistical analyses.

        Parameters
        ----------
        data : ContractCrispTable
            Input data for analysis.

        Returns
        -------
        Dict[str, ContractStatisticsDict]
            Multiple statistical summaries.
        """
        # Row-wise statistics
        row_stats = {
            'mean': float(data.mean(axis=1).mean()),
            'std': float(data.mean(axis=1).std()),
            'min': float(data.mean(axis=1).min()),
            'max': float(data.mean(axis=1).max())
        }

        # Column-wise statistics
        col_stats = {
            'mean': float(data.mean(axis=0).mean()),
            'std': float(data.mean(axis=0).std()),
            'min': float(data.mean(axis=0).min()),
            'max': float(data.mean(axis=0).max())
        }

        return {
            'row_statistics': row_stats,
            'column_statistics': col_stats
        }


class DemoRanker(AnalysisComponent):
    """
    Ranks alternatives based on scores.

    Parameters
    ----------
    ascending : bool, default False
        Whether to rank in ascending order.
    """

    def __init__(self, ascending: bool = False):
        self.ascending = ascending

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'ascending': self.ascending}

    @contract
    def run(self, scores: ContractAccuracyVector) -> ContractOrderResult:
        """
        Ranks alternatives based on scores.

        Parameters
        ----------
        scores : ContractScoreVector
            Input scores to rank.

        Returns
        -------
        ContractRankingResult
            Ranking results (1 = best).
        """
        return scores.rank(ascending=self.ascending, method='min').astype(int)


class DemoDataAggregator(AnalysisComponent):
    """
    Aggregates multiple data sources (multi-input demo).

    Parameters
    ----------
    aggregation_method : str, default 'average'
        Method for aggregating multiple inputs.
    """

    def __init__(self, aggregation_method: str = 'average'):
        valid_methods = ['average', 'weighted_average']
        if aggregation_method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.aggregation_method = aggregation_method

    def get_config(self) -> dict:
        """Returns the component's configuration."""
        return {'aggregation_method': self.aggregation_method}

    @contract
    def run(self,
            data1: ContractCrispTable,
            data2: ContractCrispTable,
            weights: ContractNormalizedWeights = None) -> ContractCrispTable:
        """
        Aggregates multiple data sources.

        Parameters
        ----------
        data1 : ContractCrispTable
            First data source.
        data2 : ContractCrispTable
            Second data source.
        weights : ContractNormalizedWeights, optional
            Weights for weighted aggregation.

        Returns
        -------
        ContractCrispTable
            Aggregated data.
        """
        if self.aggregation_method == 'average':
            return (data1 + data2) / 2
        else:  # weighted_average
            if weights is None or len(weights) != 2:
                weights = np.array([0.5, 0.5])
            return data1 * weights[0] + data2 * weights[1]
