#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:39
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/24 16:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import TypeAlias, Dict, Any, Union, List

import numpy as np
import pandas as pd

from axisfuzzy.core import Fuzznum, Fuzzarray
from .base import ContractValidator
from .registry import register_contract, get_registry_contracts
from ..dataframe import FuzzyDataFrame


CrispTable: TypeAlias = "pd.DataFrame"
"""
Type alias for crisp (non-fuzzy) tabular data.

Represents traditional numerical data in tabular format, typically used as
input for fuzzification processes.

Examples
--------
.. code-block:: python

    import pandas as pd
    from axisfuzzy.analysis.contracts import CrispTable

    # A crisp decision matrix
    crisp_data: CrispTable = pd.DataFrame({
        'Price': [100, 200, 300],
        'Quality': [8, 9, 7],
        'Service': [6, 8, 9]
    }, index=['Alt_A', 'Alt_B', 'Alt_C'])
"""

FuzzyTable: TypeAlias = "FuzzyDataFrame"
"""
Type alias for fuzzy tabular data.

Represents tabular data where columns contain Fuzzarray objects instead of
scalar values. This is the primary data structure for fuzzy decision analysis.

Notes
-----
- Each column should contain a single Fuzzarray representing fuzzy values
  for all alternatives under that criterion
- Index represents alternatives, columns represent criteria/attributes
- Compatible with pandas DataFrame operations while maintaining fuzzy semantics

Examples
--------
.. code-block:: python

    # A fuzzy decision matrix
    fuzzy_data: FuzzyTable = pd.DataFrame({
        'Price': price_fuzzarray,      # Fuzzarray with fuzzy price values
        'Quality': quality_fuzzarray,  # Fuzzarray with fuzzy quality values  
        'Service': service_fuzzarray   # Fuzzarray with fuzzy service values
    })
"""

WeightVector: TypeAlias = Union["np.ndarray", "pd.Series"]
"""
Type alias for criterion weights.

Represents the relative importance of different criteria in multi-criteria
decision analysis. Values should typically sum to 1.0.

Examples
--------
.. code-block:: python

    import numpy as np
    import pandas as pd
    
    # As numpy array
    weights: WeightVector = np.array([0.3, 0.4, 0.3])
    
    # As pandas Series with labels
    weights: WeightVector = pd.Series(
        [0.3, 0.4, 0.3], 
        index=['Price', 'Quality', 'Service'])
"""

ScoreVector: TypeAlias = Union["np.ndarray", "pd.Series"]
"""
Type alias for alternative scores.

Represents computed scores for each alternative in decision analysis,
typically used for ranking or further decision processing.

Examples
--------
.. code-block:: python

    # Scores for three alternatives
    scores: ScoreVector = np.array([0.65, 0.78, 0.55])
    
    # With alternative names
    scores: ScoreVector = pd.Series(
        [0.65, 0.78, 0.55],
        index=['Car_A', 'Car_B', 'Car_C'])
"""

FuzzyNumber: TypeAlias = "Fuzznum"
"""
Type alias for a single fuzzy number.

Represents an individual fuzzy number instance, typically used for
scalar fuzzy operations or as building blocks for fuzzy arrays.
"""

FuzzyArray: TypeAlias = "Fuzzarray"
"""
Type alias for fuzzy array data.

Represents high-performance arrays of fuzzy numbers, used for vectorized
fuzzy computations.
"""

RankingResult: TypeAlias = Union["pd.Series", List[str], List[int]]
"""
Type alias for ranking results.

Represents the ranked order of alternatives, which can be expressed as
ordered indices, names, or pandas Series with ranking information.

Examples
--------
.. code-block:: python

    # As list of alternative names (best to worst)
    ranking: RankingResult = ['Car_B', 'Car_A', 'Car_C']
    
    # As pandas Series with ranking scores
    ranking: RankingResult = pd.Series(
        [2, 1, 3], 
        index=['Car_A', 'Car_B', 'Car_C'])
"""

ThreeWayResult: TypeAlias = Dict[str, List[str]]
"""
Type alias for three-way decision results.

Represents the outcome of three-way decision analysis, partitioning
alternatives into accept, defer, and reject categories.

Examples
--------
.. code-block:: python

    result: ThreeWayResult = {
        'accept': ['Car_B', 'Car_A'],
        'defer': ['Car_C'],
        'reject': []
    }
"""

PairwiseMatrix: TypeAlias = "pd.DataFrame"
"""
Type alias for pairwise comparison matrices.

Used in methods like AHP (Analytic Hierarchy Process) where criteria
or alternatives are compared pairwise.

Examples
--------
.. code-block:: python

    # AHP pairwise comparison matrix
    pairwise: PairwiseMatrix = pd.DataFrame([
        [1.0, 3.0, 0.5],
        [1/3, 1.0, 2.0],
        [2.0, 0.5, 1.0]
    ], index=['Price', 'Quality', 'Service'],
    columns=['Price', 'Quality', 'Service'])
"""

NumericValue: TypeAlias = Union[int, float]
"""
Type alias for a single numeric value.

Examples
--------
.. code-block:: python

    threshold: NumericValue = 0.5
    count: NumericValue = 10
"""

CriteriaList: TypeAlias = List[str]
"""
Type alias for criterion names.

Examples
--------
.. code-block:: python

    criteria: CriteriaList = ['Cost', 'Quality', 'Delivery_Time']
"""

AlternativeList: TypeAlias = List[str]
"""
Type alias for alternative names.

Examples
--------
.. code-block:: python

    alternatives: AlternativeList = ['Supplier_A', 'Supplier_B', 'Supplier_C']
"""

NormalizedWeights: TypeAlias = Union["np.ndarray", "pd.Series"]
"""
Type alias for normalized criterion weights (sum to 1.0).

Examples
--------
.. code-block:: python

    weights: NormalizedWeights = np.array([0.3, 0.4, 0.3])  # sum = 1.0
"""

Matrix: TypeAlias = Union["np.ndarray", "pd.DataFrame"]
"""
Type alias for general numeric matrices.

Examples
--------
.. code-block:: python

    correlation_matrix: Matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
"""

StatisticsDict: TypeAlias = Dict[str, Union[float, int]]
"""
Type alias for statistical summaries.

Examples
--------
.. code-block:: python

    stats: StatisticsDict = {
        'mean': 0.75, 'std': 0.12, 'min': 0.45, 'max': 0.95, 'count': 100
    }
"""


CrispTable: TypeAlias = "pd.DataFrame"
"""
Type alias for crisp (non-fuzzy) tabular data.

Represents traditional numerical data in tabular format, typically used as
input for fuzzification processes.

Examples
--------
.. code-block:: python

    import pandas as pd
    from axisfuzzy.analysis.contracts import CrispTable

    # A crisp decision matrix
    crisp_data: CrispTable = pd.DataFrame({
        'Price': [100, 200, 300],
        'Quality': [8, 9, 7],
        'Service': [6, 8, 9]
    }, index=['Alt_A', 'Alt_B', 'Alt_C'])
"""


@register_contract('CrispTable')
class _CrispTableValidator(ContractValidator):
    """Validates that an object is a numeric pandas DataFrame."""

    def validate(self, obj: Any) -> bool:
        if not isinstance(obj, pd.DataFrame):
            return False
        if obj.empty:
            return True
        return all(pd.api.types.is_numeric_dtype(dtype) for dtype in obj.dtypes)


@register_contract('FuzzyTable')
class _FuzzyTableValidator(ContractValidator):
    """Validates that an object is a pandas DataFrame of Fuzznum objects."""

    def validate(self, obj: Any) -> bool:
        # Lazy import to avoid circular dependencies at the module level.
        from ..dataframe import FuzzyDataFrame
        # The sole and correct validation criterion: it must be an instance of FuzzyDataFrame.
        return isinstance(obj, FuzzyDataFrame)


@register_contract('WeightVector')
class _WeightVectorValidator(ContractValidator):
    """Validates that an object is a 1D numpy array or a pandas Series."""

    def validate(self, obj: Any) -> bool:
        return (isinstance(obj, np.ndarray) and obj.ndim == 1) or \
            isinstance(obj, pd.Series)


# Register ScoreVector using the same validator as ContractWeightVector for DRY principle
get_registry_contracts().register('ScoreVector', _WeightVectorValidator())


@register_contract('FuzzyNumber')
def _is_fuzznum(obj: Any) -> bool:
    """Validates if an object is a Fuzznum instance."""
    from axisfuzzy.core.fuzznums import Fuzznum
    return isinstance(obj, Fuzznum)


@register_contract('FuzzyArray')
def _is_fuzzarray(obj: Any) -> bool:
    """Validates if an object is a Fuzzarray instance."""
    from axisfuzzy.core.fuzzarray import Fuzzarray
    return isinstance(obj, Fuzzarray)


@register_contract('RankingResult')
class _RankingResultValidator(ContractValidator):
    """Validates that an object represents a ranking result."""

    def validate(self, obj: Any) -> bool:
        if isinstance(obj, pd.Series):
            return True
        if isinstance(obj, list):
            # An empty list is a valid ranking
            if not obj:
                return True
            # Check if all elements are of the same simple type (int or str)
            first_type = type(obj[0])
            if first_type not in [int, str]:
                return False
            return all(isinstance(x, first_type) for x in obj)
        return False


@register_contract('ThreeWayResult')
def _is_threeway_result(obj: Any) -> bool:
    """Validates if an object is a three-way decision result dictionary."""
    return (isinstance(obj, dict) and
            all(k in obj for k in ['accept', 'reject', 'defer']))


@register_contract('PairwiseMatrix')
class _PairwiseMatrixValidator(ContractValidator):
    """Validates that an object is a square pandas DataFrame."""

    def validate(self, obj: Any) -> bool:
        if not isinstance(obj, pd.DataFrame):
            return False
        # Must be a square matrix
        return obj.shape[0] == obj.shape[1]


@register_contract('NumericValue')
def _is_numeric_value(obj: Any) -> bool:
    """Validates if an object is a single numeric value."""
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)


@register_contract('CriteriaList')
def _is_criteria_list(obj: Any) -> bool:
    """Validates if an object is a list of criterion names."""
    return (isinstance(obj, list) and
            all(isinstance(item, str) for item in obj))


@register_contract('AlternativeList')
def _is_alternative_list(obj: Any) -> bool:
    """Validates if an object is a list of alternative names."""
    return (isinstance(obj, list) and
            all(isinstance(item, str) for item in obj))


@register_contract('NormalizedWeights')
class _NormalizedWeightsValidator(ContractValidator):
    """Validates normalized weights (sum to 1.0 within tolerance)."""

    def validate(self, obj: Any) -> bool:
        if not ((isinstance(obj, np.ndarray) and obj.ndim == 1) or isinstance(obj, pd.Series)):
            return False

        if len(obj) == 0:
            return False

        # Check if all values are non-negative
        if np.any(obj < 0):
            return False

        # Check if sum is approximately 1.0 (within tolerance)
        return np.isclose(np.sum(obj), 1.0, atol=1e-6)


@register_contract('Matrix')
class _MatrixValidator(ContractValidator):
    """Validates that an object is a 2D numeric matrix."""

    def validate(self, obj: Any) -> bool:
        if isinstance(obj, np.ndarray):
            return obj.ndim == 2 and np.issubdtype(obj.dtype, np.number)
        elif isinstance(obj, pd.DataFrame):
            if obj.empty:
                return True
            return all(pd.api.types.is_numeric_dtype(dtype) for dtype in obj.dtypes)
        return False


@register_contract('StatisticsDict')
def _is_statistics_dict(obj: Any) -> bool:
    """Validates if an object is a statistics dictionary."""
    if not isinstance(obj, dict):
        return False
    return all(isinstance(v, (int, float)) for v in obj.values())
