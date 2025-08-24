#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:44
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Any

# --- Optional Dependencies ---
try:
    import pandas as pd
    import numpy as np

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    _PANDAS_AVAILABLE = False

# --- Local Imports ---
from .contracts import Contract

from axisfuzzy.core import Fuzznum, Fuzzarray
from axisfuzzy.analysis.dataframe import FuzzyDataFrame


# --- Helper Validators ---
def _is_pandas_df(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame)


def _is_pandas_series(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and isinstance(obj, pd.Series)


def _is_numpy_array(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and isinstance(obj, np.ndarray)


# --- Base Contracts ---
ContractAny = Contract('Any', lambda obj: True)

ContractCrispTable = Contract(
    'ContractCrispTable',
    lambda obj: _is_pandas_df(obj) and all(pd.api.types.is_numeric_dtype(dtype) for dtype in obj.dtypes)
)

ContractFuzzyTable = Contract(
    'ContractFuzzyTable',
    lambda obj: isinstance(obj, FuzzyDataFrame))

ContractWeightVector = Contract(
    'ContractWeightVector',
    lambda obj: (_is_numpy_array(obj) and obj.ndim == 1) or _is_pandas_series(obj)
)

ContractMatrix = Contract(
    'ContractMatrix',
    lambda obj: (_is_numpy_array(obj) and obj.ndim == 2) or _is_pandas_df(obj)
)

ContractFuzzyNumber = Contract(
    'ContractFuzzyNumber',
    lambda obj: isinstance(obj, Fuzznum))

ContractFuzzyArray = Contract(
    'ContractFuzzyArray',
    lambda obj: isinstance(obj, Fuzzarray))

ContractNumericValue = Contract(
    'ContractNumericValue',
    lambda obj: isinstance(obj, (int, float)) and not isinstance(obj, bool))

ContractStringList = Contract(
    'ContractStringList',
    lambda obj: isinstance(obj, list) and all(isinstance(i, str) for i in obj))

# --- Derived Contracts ---
ContractScoreVector = Contract(
    'ContractScoreVector',
    ContractWeightVector.validate,
    parent=ContractWeightVector)

ContractNormalizedWeights = Contract(
    'ContractNormalizedWeights',
    lambda obj: ContractWeightVector.validate(obj) and len(obj) > 0 and np.isclose(np.sum(obj), 1.0),
    parent=ContractWeightVector
)

ContractPairwiseMatrix = Contract(
    'ContractPairwiseMatrix',
    lambda obj: _is_pandas_df(obj) and obj.shape[0] == obj.shape[1],
    parent=ContractMatrix
)
ContractCriteriaList = Contract(
    'ContractCriteriaList',
    ContractStringList.validate,
    parent=ContractStringList)

ContractAlternativeList = Contract(
    'ContractAlternativeList',
    ContractStringList.validate,
    parent=ContractStringList)

ContractRankingResult = Contract(
    'ContractRankingResult',
    lambda obj: _is_pandas_series(obj) or (isinstance(obj, list) and all(isinstance(i, (str, int)) for i in obj))
)
ContractThreeWayResult = Contract(
    'ContractThreeWayResult',
    lambda obj: isinstance(obj, dict) and all(k in obj for k in ['accept', 'reject', 'defer'])
)
ContractStatisticsDict = Contract(
    'ContractStatisticsDict',
    lambda obj: isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values())
)

