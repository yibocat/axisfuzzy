#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/22 11:45
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Data contracts and type definitions for fuzzy data analysis.

This module defines the core data contracts used throughout the fuzzy analysis
pipeline system. These type aliases serve as the foundation for tool input/output
validation and enable flexible composition of analysis components.

Overview
--------
Data contracts provide a standardized way to declare what types of data each
analysis tool expects as input and produces as output. This enables the pipeline
engine to perform type checking and ensures compatibility between different
analysis components.

This module establishes a formal system for defining and validating the types
of data structures that flow through an analysis pipeline. It provides:

1.  **Type Aliases**: For static analysis and improved code readability.
2.  **A Validation System**: A runtime mechanism to ensure data conforms to
    pre-defined contracts, driven by a central registry.
3.  **An Extensible API**: A decorator-based system (`@register_contract`)
    that allows developers to easily define and register new custom contracts.
"""

from typing import TypeAlias, Union, Dict, Any, List, Type, Callable
from abc import ABC, abstractmethod

try:
    import pandas as pd
    import numpy as np

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    # Create placeholder types when pandas is not available

    class _MockDataFrame:
        pass

    class _MockSeries:
        pass


    pd = type('MockPandas', (), {'DataFrame': _MockDataFrame, 'Series': _MockSeries})()
    np = type('MockNumpy', (), {'ndarray': object})()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axisfuzzy.core import Fuzznum, Fuzzarray


# ===================================================================
# 1. Type Aliases for Static Analysis & Code Hinting
# ===================================================================

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


FuzzyTable: TypeAlias = "pd.DataFrame"
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


# ===================================================================
# 2. Validator Interface Definition
# ===================================================================

class ContractValidator(ABC):
    """
    Abstract base class for all data contract validators.

    This class defines a standard interface, ``validate``, which all concrete
    validator implementations must provide. This ensures that the registry
    and pipeline engine can handle any registered contract polymorphically.
    """
    @abstractmethod
    def validate(self, obj: Any) -> bool:
        """
        Validate if an object conforms to the specific contract.

        Parameters
        ----------
        obj : Any
            The object to be validated.

        Returns
        -------
        bool
            ``True`` if the object conforms to the contract, ``False`` otherwise.
        """
        pass


# ===================================================================
# 3. Contract Registry (Singleton)
# ===================================================================


class _ContractRegistry:
    """
    Manages all data contracts and their validators in a central registry.

    This is a singleton class ensuring that there is only one instance of the
    registry throughout the application's lifecycle. It maps contract names
    (strings) to their corresponding validator instances. This class is not
    intended for direct use; interact with it via the public API functions
    ``register_contract`` and ``validate``.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_ContractRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # The __init__ method is guarded to run only once.
        if self._initialized:
            return
        self._validators: Dict[str, ContractValidator] = {}
        self._initialized = True

    def register(self, name: str, validator: ContractValidator):
        if not isinstance(validator, ContractValidator):
            raise TypeError("Validator must be an instance of ContractValidator.")
        self._validators[name] = validator

    def get_validator(self, name: str) -> ContractValidator:
        if name not in self._validators:
            available = ", ".join(self._validators.keys())
            raise KeyError(
                f"Contract '{name}' is not registered. "
                f"Available contracts: [{available}]"
            )
        return self._validators[name]


_contract_registry = _ContractRegistry()


# ===================================================================
# 4. Public API: Decorator and Validation Function
# ===================================================================

def get_registry_contracts() -> _ContractRegistry:
    """
    Get the singleton instance of the contract registry.

    This function provides access to the central contract registry, allowing
    inspection of registered contracts and their validators.

    Returns
    -------
    _ContractRegistry
        The singleton instance of the contract registry.
    """
    return _contract_registry


def register_contract(name: str) -> Callable:
    """
    A decorator to register a data contract validator.

    This decorator can be applied to a class that inherits from
    `ContractValidator` or to a simple function that returns a boolean.
    It provides a declarative and elegant way to extend the system with
    new, custom data contracts.

    Parameters
    ----------
    name : str
        The unique name for the contract being registered.

    Returns
    -------
    Callable
        A decorator that registers the decorated class or function.

    Examples
    --------
    Decorating a class:

    .. code-block:: python

        from axisfuzzy.analysis.contracts import register_contract, ContractValidator

        @register_contract("MyCustomContract")
        class MyValidator(ContractValidator):
            def validate(self, obj):
                return hasattr(obj, 'my_special_attribute')

    Decorating a function:

    .. code-block:: python

        @register_contract("IsPositiveNumber")
        def is_positive(obj):
            return isinstance(obj, (int, float)) and obj > 0
    """
    def decorator(validator_cls_or_func: Union[Type[ContractValidator], Callable[[Any], bool]]):
        if isinstance(validator_cls_or_func, type) and issubclass(validator_cls_or_func, ContractValidator):
            validator_instance = validator_cls_or_func()
        elif callable(validator_cls_or_func):
            class FunctionalValidator(ContractValidator):
                def validate(self, obj: Any) -> bool:
                    return validator_cls_or_func(obj)
            validator_instance = FunctionalValidator()
        else:
            raise TypeError(
                f"@{register_contract.__name__} can only decorate a "
                f"ContractValidator subclass or a callable function."
            )

        _contract_registry.register(name, validator_instance)
        return validator_cls_or_func
    return decorator


def validate(contract_name: str, obj: Any) -> bool:
    """
    Validate an object against a named contract at runtime.

    This is the primary function used by the pipeline engine to perform
    data contract checks before executing a tool.

    Parameters
    ----------
    contract_name : str
        The name of the contract to validate against.
    obj : Any
        The object to be validated.

    Returns
    -------
    bool
        ``True`` if the object conforms to the contract, ``False`` otherwise.
        Returns ``False`` if the contract name is not registered.
    """
    try:
        validator = _contract_registry.get_validator(contract_name)
        return validator.validate(obj)
    except KeyError:
        return False


# ===================================================================
# 5. Built-in Contract Implementations
# ===================================================================


@register_contract('CrispTable')
class _CrispTableValidator(ContractValidator):
    """Validates that an object is a numeric pandas DataFrame."""
    def validate(self, obj: Any) -> bool:
        if not _PANDAS_AVAILABLE:
            return False
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
        from .dataframe import FuzzyDataFrame
        # The sole and correct validation criterion: it must be an instance of FuzzyDataFrame.
        return isinstance(obj, FuzzyDataFrame)


@register_contract('WeightVector')
class _WeightVectorValidator(ContractValidator):
    """Validates that an object is a 1D numpy array or a pandas Series."""
    def validate(self, obj: Any) -> bool:
        return (isinstance(obj, np.ndarray) and obj.ndim == 1) or \
               isinstance(obj, pd.Series)


# Register ScoreVector using the same validator as WeightVector for DRY principle
_contract_registry.register('ScoreVector', _WeightVectorValidator())


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
