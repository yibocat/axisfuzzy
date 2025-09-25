#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
This module defines various operations for Fuzzy Sets (FS) based on Zadeh's fuzzy set theory.
It includes arithmetic, comparison, and set-theoretic operations, all implemented
as subclasses of `OperationMixin` and registered with the global operation registry.

Each operation class specifies:
- The operation name (e.g., 'add', 'mul', 'gt').
- The fuzzy number types it supports (currently 'fs').
- The core logic for executing the operation, often leveraging `OperationTNorm`
  for t-norm and t-conorm calculations.

Fuzzy Set Operations
--------------------
For Fuzzy Sets (FS), operations are simplified compared to Q-Rung Orthopair Fuzzy Numbers
since FS only has a membership degree (md) component. The operations follow classical
Zadeh fuzzy set theory:

- **Arithmetic Operations**: Based on extension principle and t-norm/t-conorm operations
- **Set Operations**: Union (∪), Intersection (∩), Complement (~), etc.
- **Comparison Operations**: Based on membership degree comparison

Mathematical Foundations
------------------------
For two fuzzy sets A and B with membership degrees μ_A(x) and μ_B(x):

- Union: μ_{A∪B}(x) = S(μ_A(x), μ_B(x)) where S is a t-conorm
- Intersection: μ_{A∩B}(x) = T(μ_A(x), μ_B(x)) where T is a t-norm  
- Complement: μ_{~A}(x) = 1 - μ_A(x)
- Addition: μ_{A+B}(x) = S(μ_A(x), μ_B(x))
- Multiplication: μ_{A×B}(x) = T(μ_A(x), μ_B(x))
"""

import warnings
from typing import List, Any, Dict, Union, Optional

import numpy as np

from ...config import get_config

from ...core import (
    Fuzznum,
    Fuzzarray,
    get_registry_fuzztype,
    OperationTNorm,
    OperationMixin,
    register_operation
)

from ...utils import experimental


def _prepare_fs_operands(
        fuzzarray_1: Fuzzarray,
        other: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to prepare operands for 'FS' operations.
    
    Parameters
    ----------
    fuzzarray_1 : Fuzzarray
        The first fuzzy array operand
    other : Any
        The second operand (Fuzzarray, Fuzznum, or scalar)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Broadcasted membership degree arrays (mds1, mds2)
        
    Raises
    ------
    ValueError
        If operands have incompatible types or shapes
    TypeError
        If operand type is not supported
    """
    # Get membership degrees from first operand
    mds1, = fuzzarray_1.backend.get_component_arrays()

    if isinstance(other, Fuzzarray):
        if other.mtype != fuzzarray_1.mtype:
            raise ValueError(f"Cannot operate on Fuzzarrays with different mtypes: "
                             f"{fuzzarray_1.mtype} and {other.mtype}")

        mds2, = other.backend.get_component_arrays()
        # Let NumPy handle broadcasting errors
        try:
            # Unpack broadcast_arrays result to ensure correct return type
            mds1_broadcast, mds2_broadcast = np.broadcast_arrays(mds1, mds2)
            return mds1_broadcast, mds2_broadcast
        except ValueError:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {fuzzarray_1.shape} and {other.shape}")

    elif isinstance(other, Fuzznum):
        if other.mtype != fuzzarray_1.mtype:
            raise TypeError(f"Mtype mismatch for operation: "
                            f"Fuzzarray('{fuzzarray_1.mtype}') and Fuzznum('{other.mtype}')")

        # Create arrays from Fuzznum and let NumPy broadcast them
        mds2 = np.full((1,), other.md, dtype=mds1.dtype)
        try:
            # Unpack broadcast_arrays result to ensure correct return type
            mds1_broadcast, mds2_broadcast = np.broadcast_arrays(mds1, mds2)
            return mds1_broadcast, mds2_broadcast
        except ValueError:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {fuzzarray_1.shape} and {mds2.shape}")

    elif isinstance(other, (int, float, np.number)):
        # Handle scalar operations - treat scalar as membership degree
        if not (0 <= other <= 1):
            warnings.warn(f"Scalar value {other} is outside [0,1] range for fuzzy operations",
                          UserWarning)
        mds2 = np.full_like(mds1, float(other))
        return mds1, mds2
    else:
        raise TypeError(f"Unsupported operand type for vectorized operation: {type(other)}")


# --- FS Arithmetic Operations ---

@register_operation
class FSAddition(OperationMixin):
    """
    Implements the addition operation for Fuzzy Sets (FS).

    The addition of two fuzzy sets A and B is defined as:
    μ_{A+B}(x) = S(μ_A(x), μ_B(x))
    where S is a t-conorm (fuzzy OR operation).
    
    This follows the algebraic sum interpretation in fuzzy set theory.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'add'.
        """
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary addition operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # For FS addition, use t-conorm (fuzzy OR)
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized addition for FS fuzzy arrays.

        Formula:
        - md_result = S(md1, md2) = t-conorm of membership degrees
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = S(md1, md2) - fuzzy union
        md_res = tnorm.t_conorm(mds1, mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSSubtraction(OperationMixin):
    """
    Implements the subtraction operation for Fuzzy Sets (FS).

    The subtraction of two fuzzy sets A and B is defined as:
    μ_{A-B}(x) = T(μ_A(x), 1 - μ_B(x))
    where T is a t-norm and represents A ∩ ~B (A intersect complement of B).
    
    This is equivalent to the fuzzy difference operation.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'sub'.
        """
        return 'sub'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary subtraction operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # For FS subtraction: A - B = A ∩ ~B = T(μ_A, 1 - μ_B)
        complement_md2 = 1.0 - strategy_2.md
        md = tnorm.t_norm(strategy_1.md, complement_md2)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized subtraction for FS fuzzy arrays.

        Formula:
        - md_result = T(md1, 1 - md2) = t-norm with complement
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = T(md1, 1 - md2) - fuzzy difference
        complement_mds2 = 1.0 - mds2
        md_res = tnorm.t_norm(mds1, complement_mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSMultiplication(OperationMixin):
    """
    Implements the multiplication operation for Fuzzy Sets (FS).

    The multiplication of two fuzzy sets A and B is defined as:
    μ_{A×B}(x) = T(μ_A(x), μ_B(x))
    where T is a t-norm (fuzzy AND operation).
    
    This represents the fuzzy intersection of the two sets.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'mul'.
        """
        return 'mul'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary multiplication operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # For FS multiplication, use t-norm (fuzzy AND)
        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized multiplication for FS fuzzy arrays.

        Formula:
        - md_result = T(md1, md2) = t-norm of membership degrees
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = T(md1, md2) - fuzzy intersection
        md_res = tnorm.t_norm(mds1, mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSDivision(OperationMixin):
    """
    Implements the division operation for Fuzzy Sets (FS).

    The division of two fuzzy sets A and B is defined as:
    μ_{A/B}(x) = S(1 - μ_A(x), μ_B(x))
    where S is a t-conorm, representing the fuzzy implication ~A ∪ B.
    
    This follows the material implication interpretation in fuzzy logic.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'truediv'.
        """
        return 'div'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary division operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # For FS division: A / B = ~A ∪ B = S(1 - μ_A, μ_B)
        # Handle division by zero case
        if strategy_2.md == 0.0:
            warnings.warn("Division by fuzzy set with zero membership degree", UserWarning)
            md = 1.0  # Implication is true when antecedent is false
        else:
            complement_md1 = 1.0 - strategy_1.md
            md = tnorm.t_conorm(complement_md1, strategy_2.md)

        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized division for FS fuzzy arrays.

        Formula:
        - md_result = S(1 - md1, md2) = t-conorm with complement
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Handle division by zero case
        zero_mask = (mds2 == 0.0)
        if np.any(zero_mask):
            warnings.warn("Division by fuzzy set with zero membership degrees detected", UserWarning)

        # Formula: md = S(1 - md1, md2) - fuzzy implication
        complement_mds1 = 1.0 - mds1
        md_res = tnorm.t_conorm(complement_mds1, mds2)

        # Set result to 1.0 where divisor is 0
        md_res = np.where(zero_mask, 1.0, md_res)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSPower(OperationMixin):
    """
    Implements the power operation for Fuzzy Sets (FS).

    The power operation for a fuzzy set A with exponent n is defined as:
    μ_{A^n}(x) = (μ_A(x))^n
    
    This represents the concentration (n > 1) or dilation (0 < n < 1) of the fuzzy set.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'pow'.
        """
        return 'pow'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the power operation on an FS strategy with a numeric operand.

        Parameters
        ----------
        strategy : Any
            The FS strategy instance.
        operand : Union[int, float]
            The power exponent.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for power operation).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        if operand < 0:
            warnings.warn(f"Negative exponent {operand} may produce unexpected results", UserWarning)

        # Power operation: μ^n
        md = np.power(strategy.md, operand)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized power operation for FS fuzzy arrays.

        Formula:
        - md_result = md^operand
        """
        if operand < 0:
            warnings.warn(f"Negative exponent {operand} may produce unexpected results", UserWarning)

        mds, = fuzzarray.backend.get_component_arrays()

        # Formula: md = md^operand
        md_res = np.power(mds, operand)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSTimes(OperationMixin):
    """
    Implements the scalar multiplication operation for Fuzzy Sets (FS).

    The scalar multiplication of a fuzzy set A with scalar λ is defined as:
    μ_{λA}(x) = (μ_A(x))^(1/λ) for λ > 0
    
    This represents the linguistic hedge operation in fuzzy logic.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'times'.
        """
        return 'tim'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the scalar multiplication operation on an FS strategy.

        Parameters
        ----------
        strategy : Any
            The FS strategy instance.
        operand : Union[int, float]
            The scalar multiplier.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for times operation).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        if operand <= 0:
            raise ValueError(f"Scalar multiplier must be positive, got {operand}")

        # Scalar multiplication: μ^(1/λ)
        md = np.power(strategy.md, 1.0 / operand)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized scalar multiplication for FS fuzzy arrays.

        Formula:
        - md_result = md^(1/operand)
        """
        if operand <= 0:
            raise ValueError(f"Scalar multiplier must be positive, got {operand}")

        mds, = fuzzarray.backend.get_component_arrays()

        # Formula: md = md^(1/operand)
        md_res = np.power(mds, 1.0 / operand)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


# --- FS Comparison Operations ---

@register_operation
class FSGreaterThan(OperationMixin):
    """
    Implements the greater-than comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A > B is determined by comparing their membership degrees:
    A > B if μ_A > μ_B
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'gt'.
        """
        return 'gt'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater-than comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        tolerance = get_config().DEFAULT_EPSILON 
        result = strategy_1.md > strategy_2.md + tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized greater-than comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return mds1 > mds2 + tolerance


@register_operation
class FSLessThan(OperationMixin):
    """
    Implements the less-than comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A < B is determined by comparing their membership degrees:
    A < B if μ_A < μ_B
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'lt'.
        """
        return 'lt'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less-than comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        tolerance = get_config().DEFAULT_EPSILON
        result = strategy_1.md < strategy_2.md - tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized less-than comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return mds1 < mds2 - tolerance


@register_operation
class FSEquals(OperationMixin):
    """
    Implements the equality comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A == B is determined by comparing their membership degrees:
    A == B if μ_A == μ_B (within numerical tolerance)
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'eq'.
        """
        return 'eq'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the equality comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        # Use numerical tolerance for floating point comparison
        tolerance = get_config().DEFAULT_EPSILON
        result = abs(strategy_1.md - strategy_2.md) < tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized equality comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return np.abs(mds1 - mds2) < tolerance


@register_operation
class FSGreaterEquals(OperationMixin):
    """
    Implements the greater-than-or-equal comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A >= B is determined by comparing their membership degrees:
    A >= B if μ_A >= μ_B
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'ge'.
        """
        return 'ge'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater-than-or-equal comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        tolerance = get_config().DEFAULT_EPSILON
        result = strategy_1.md >= strategy_2.md + tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized greater-than-or-equal comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return mds1 >= mds2 + tolerance


@register_operation
class FSLessEquals(OperationMixin):
    """
    Implements the less-than-or-equal comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A <= B is determined by comparing their membership degrees:
    A <= B if μ_A <= μ_B
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'le'.
        """
        return 'le'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less-than-or-equal comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        tolerance = get_config().DEFAULT_EPSILON
        result = strategy_1.md <= strategy_2.md - tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized less-than-or-equal comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return mds1 <= mds2 - tolerance


@register_operation
class FSNotEquals(OperationMixin):
    """
    Implements the not-equal comparison for Fuzzy Sets (FS).

    For fuzzy sets A and B, A != B is determined by comparing their membership degrees:
    A != B if μ_A != μ_B (beyond numerical tolerance)
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'ne'.
        """
        return 'ne'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the not-equal comparison between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for comparison).

        Returns
        -------
        Dict[str, bool]
            A dictionary containing the comparison result.
        """
        # Use numerical tolerance for floating point comparison
        tolerance = get_config().DEFAULT_EPSILON
        result = abs(strategy_1.md - strategy_2.md) >= tolerance
        return {'value': result}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        High-performance vectorized not-equal comparison for FS fuzzy arrays.
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, fuzzarray_2)
        tolerance = get_config().DEFAULT_EPSILON
        return np.abs(mds1 - mds2) >= tolerance


# --- FS Set Operations ---

@register_operation
class FSIntersection(OperationMixin):
    """
    Implements the intersection operation for Fuzzy Sets (FS).

    The intersection of two fuzzy sets A and B is defined as:
    μ_{A∩B}(x) = T(μ_A(x), μ_B(x))
    where T is a t-norm (fuzzy AND operation).
    
    This is equivalent to the multiplication operation for FS.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'intersection'.
        """
        return 'intersection'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the intersection operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Intersection uses t-norm (fuzzy AND)
        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized intersection for FS fuzzy arrays.

        Formula:
        - md_result = T(md1, md2) = t-norm of membership degrees
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = T(md1, md2) - fuzzy intersection
        md_res = tnorm.t_norm(mds1, mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSUnion(OperationMixin):
    """
    Implements the union operation for Fuzzy Sets (FS).

    The union of two fuzzy sets A and B is defined as:
    μ_{A∪B}(x) = S(μ_A(x), μ_B(x))
    where S is a t-conorm (fuzzy OR operation).
    
    This is equivalent to the addition operation for FS.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'union'.
        """
        return 'union'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the union operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Union uses t-conorm (fuzzy OR)
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized union for FS fuzzy arrays.

        Formula:
        - md_result = S(md1, md2) = t-conorm of membership degrees
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = S(md1, md2) - fuzzy union
        md_res = tnorm.t_conorm(mds1, mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSComplement(OperationMixin):
    """
    Implements the complement operation for Fuzzy Sets (FS).

    The complement of a fuzzy set A is defined as:
    μ_{~A}(x) = 1 - μ_A(x)
    
    This represents the standard fuzzy negation operation.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'complement'.
        """
        return 'complement'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the complement operation on an FS strategy.

        Parameters
        ----------
        strategy : Any
            The FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm (not used for complement).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Complement: μ_{~A} = 1 - μ_A
        md = 1.0 - strategy.md
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized complement for FS fuzzy arrays.

        Formula:
        - md_result = 1 - md
        """
        mds, = fuzzarray_1.backend.get_component_arrays()

        # Formula: md = 1 - md - fuzzy complement
        md_res = 1.0 - mds

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSDifference(OperationMixin):
    """
    Implements the difference operation for Fuzzy Sets (FS).

    The difference of two fuzzy sets A and B is defined as:
    μ_{A-B}(x) = T(μ_A(x), 1 - μ_B(x))
    where T is a t-norm, representing A ∩ ~B.
    
    This is equivalent to the subtraction operation for FS.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'difference'.
        """
        return 'difference'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the difference operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Difference: A - B = A ∩ ~B = T(μ_A, 1 - μ_B)
        complement_md2 = 1.0 - strategy_2.md
        md = tnorm.t_norm(strategy_1.md, complement_md2)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized difference for FS fuzzy arrays.

        Formula:
        - md_result = T(md1, 1 - md2) = t-norm with complement
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = T(md1, 1 - md2) - fuzzy difference
        complement_mds2 = 1.0 - mds2
        md_res = tnorm.t_norm(mds1, complement_mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSSymmetricDifference(OperationMixin):
    """
    Implements the symmetric difference operation for Fuzzy Sets (FS).

    The symmetric difference of two fuzzy sets A and B is defined as:
    μ_{A⊕B}(x) = S(T(μ_A(x), 1 - μ_B(x)), T(1 - μ_A(x), μ_B(x)))
    where S is a t-conorm and T is a t-norm, representing (A ∩ ~B) ∪ (~A ∩ B).
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'symdiff'.
        """
        return 'symdiff'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the symmetric difference operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Symmetric difference: (A ∩ ~B) ∪ (~A ∩ B)
        # = S(T(μ_A, 1 - μ_B), T(1 - μ_A, μ_B))
        complement_md1 = 1.0 - strategy_1.md
        complement_md2 = 1.0 - strategy_2.md

        term1 = tnorm.t_norm(strategy_1.md, complement_md2)  # A ∩ ~B
        term2 = tnorm.t_norm(complement_md1, strategy_2.md)  # ~A ∩ B

        md = tnorm.t_conorm(term1, term2)  # (A ∩ ~B) ∪ (~A ∩ B)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized symmetric difference for FS fuzzy arrays.

        Formula:
        - md_result = S(T(md1, 1 - md2), T(1 - md1, md2))
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = S(T(md1, 1 - md2), T(1 - md1, md2))
        complement_mds1 = 1.0 - mds1
        complement_mds2 = 1.0 - mds2

        term1 = tnorm.t_norm(mds1, complement_mds2)  # A ∩ ~B
        term2 = tnorm.t_norm(complement_mds1, mds2)  # ~A ∩ B

        md_res = tnorm.t_conorm(term1, term2)  # (A ∩ ~B) ∪ (~A ∩ B)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSImplication(OperationMixin):
    """
    Implements the implication operation for Fuzzy Sets (FS).

    The implication A → B is defined as:
    μ_{A→B}(x) = S(1 - μ_A(x), μ_B(x))
    where S is a t-conorm, representing ~A ∪ B.
    
    This is equivalent to the division operation for FS.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'implication'.
        """
        return 'implication'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the implication operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance (antecedent).
        strategy_2 : Any
            The second FS strategy instance (consequent).
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Implication: A → B = ~A ∪ B = S(1 - μ_A, μ_B)
        complement_md1 = 1.0 - strategy_1.md
        md = tnorm.t_conorm(complement_md1, strategy_2.md)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized implication for FS fuzzy arrays.

        Formula:
        - md_result = S(1 - md1, md2) = t-conorm with complement
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = S(1 - md1, md2) - fuzzy implication
        complement_mds1 = 1.0 - mds1
        md_res = tnorm.t_conorm(complement_mds1, mds2)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


@register_operation
class FSEquivalence(OperationMixin):
    """
    Implements the equivalence operation for Fuzzy Sets (FS).

    The equivalence A ↔ B is defined as:
    μ_{A↔B}(x) = T(S(1 - μ_A(x), μ_B(x)), S(1 - μ_B(x), μ_A(x)))
    where S is a t-conorm and T is a t-norm, representing (A → B) ∩ (B → A).
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns
        -------
        str
            The string 'equivalence'.
        """
        return 'equivalence'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns
        -------
        List[str]
            A list containing 'fs'.
        """
        return ['fs']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the equivalence operation between two FS strategies.

        Parameters
        ----------
        strategy_1 : Any
            The first FS strategy instance.
        strategy_2 : Any
            The second FS strategy instance.
        tnorm : OperationTNorm
            An instance of OperationTNorm to perform t-norm and t-conorm calculations.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'md' of the resulting FS.
        """
        # Equivalence: (A → B) ∩ (B → A)
        # = T(S(1 - μ_A, μ_B), S(1 - μ_B, μ_A))
        complement_md1 = 1.0 - strategy_1.md
        complement_md2 = 1.0 - strategy_2.md

        implication_ab = tnorm.t_conorm(complement_md1, strategy_2.md)  # A → B
        implication_ba = tnorm.t_conorm(complement_md2, strategy_1.md)  # B → A

        md = tnorm.t_norm(implication_ab, implication_ba)  # (A → B) ∩ (B → A)
        return {'md': md}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized equivalence for FS fuzzy arrays.

        Formula:
        - md_result = T(S(1 - md1, md2), S(1 - md2, md1))
        """
        mds1, mds2 = _prepare_fs_operands(fuzzarray_1, other)

        # Formula: md = T(S(1 - md1, md2), S(1 - md2, md1))
        complement_mds1 = 1.0 - mds1
        complement_mds2 = 1.0 - mds2

        implication_ab = tnorm.t_conorm(complement_mds1, mds2)  # A → B
        implication_ba = tnorm.t_conorm(complement_mds2, mds1)  # B → A

        md_res = tnorm.t_norm(implication_ab, implication_ba)  # (A → B) ∩ (B → A)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(md_res)
        return Fuzzarray(backend=new_backend)


# --- Advanced FS Operations ---

@register_operation
class FSMatmul(OperationMixin):
    """
    Implements matrix multiplication for Fuzzy Set arrays.
    
    This operation performs fuzzy matrix multiplication where:
    - Element-wise multiplication uses t-norm (intersection)
    - Summation uses t-conorm reduction (union)
    """

    def get_operation_name(self) -> str:
        return 'matmul'

    def get_supported_mtypes(self) -> List[str]:
        return ['fs']

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance fuzzy matrix multiplication for FS arrays.
        """
        if not isinstance(other, Fuzzarray):
            raise TypeError("Matrix multiplication requires two Fuzzarray operands")

        if other.mtype != fuzzarray_1.mtype:
            raise ValueError(f"Cannot perform matmul on different mtypes: "
                             f"{fuzzarray_1.mtype} and {other.mtype}")

        # Get membership degree arrays
        mds1, = fuzzarray_1.backend.get_component_arrays()
        mds2, = other.backend.get_component_arrays()

        # Validate shapes for matrix multiplication
        if mds1.ndim != 2 or mds2.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays")

        if mds1.shape[1] != mds2.shape[0]:
            raise ValueError(f"Cannot multiply matrices with shapes {mds1.shape} and {mds2.shape}")

        # Perform fuzzy matrix multiplication using vectorized operations
        # For each element (i,j) in result: S_k(T(A_ik, B_kj))
        # Using broadcasting to compute all t-norm operations at once
        # Shape: (i, k, j) where k is the common dimension
        md_products = tnorm.t_norm(mds1[:, :, np.newaxis], mds2[np.newaxis, :, :])
        
        # Aggregate along the common dimension k using t-conorm
        result_md = tnorm.t_conorm_reduce(md_products, axis=1)

        backend_cls = get_registry_fuzztype().get_backend('fs')
        new_backend = backend_cls.from_arrays(result_md)
        return Fuzzarray(backend=new_backend)


def register_fs_operations():
    """
    Register all FS operations with the global operation registry.
    
    This function is called automatically when the module is imported
    to ensure all FS operations are available for use.
    """
    # Operations are automatically registered via @register_operation decorator
    # This function serves as a placeholder for any additional registration logic
    pass


# Automatically register operations when module is imported
register_fs_operations()
