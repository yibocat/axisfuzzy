#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/9/26 
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
This module defines various operations for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).
It includes arithmetic, comparison, and set-theoretic operations, all implemented
as subclasses of `OperationMixin` and registered with the global operation registry.

Each operation class specifies:
- The operation name (e.g., 'add', 'mul', 'gt').
- The fuzzy number types it supports (currently 'ivqrofn').
- The core logic for executing the operation, often leveraging `OperationTNorm`
  for t-norm and t-conorm calculations.

IVQROFNs have the mathematical structure:
- md: interval [md_lower, md_upper] as np.ndarray shape (..., 2)
- nmd: interval [nmd_lower, nmd_upper] as np.ndarray shape (..., 2)
- Constraint: md_upper^q + nmd_upper^q ≤ 1
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


def _prepare_operands(
        fuzzarray_1: Fuzzarray,
        other: Any) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Helper to get component arrays from operands for IVQROFN operations.
    
    Args:
        fuzzarray_1: Primary IVQROFN Fuzzarray operand
        other: Secondary operand (Fuzzarray, Fuzznum, or scalar)
        
    Returns:
        Tuple of broadcasted arrays: (mds1, nmds1, mds2, nmds2)
        Each array has shape (..., 2) for interval values
        
    Raises:
        ValueError: If operands have incompatible types or shapes
        TypeError: If operand types are not supported
    """
    mds1, nmds1 = fuzzarray_1.backend.get_component_arrays()

    if isinstance(other, Fuzzarray):
        if other.mtype != fuzzarray_1.mtype:
            raise ValueError(f"Cannot operate on Fuzzarrays with different mtypes: "
                             f"{fuzzarray_1.mtype} and {other.mtype}")
        if other.q != fuzzarray_1.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{fuzzarray_1.q} and {other.q}")

        mds2, nmds2 = other.backend.get_component_arrays()
        # Let NumPy handle broadcasting errors
        try:
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        except ValueError:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {fuzzarray_1.shape} and {other.shape}")

    elif isinstance(other, Fuzznum):
        if other.mtype != fuzzarray_1.mtype:
            raise TypeError(f"Mtype mismatch for operation: "
                            f"Fuzzarray('{fuzzarray_1.mtype}') and Fuzznum('{other.mtype}')")
        if other.q != fuzzarray_1.q:
            raise ValueError(f"Q-rung mismatch for operation: "
                             f"Fuzzarray(q={fuzzarray_1.q}) and Fuzznum(q={other.q})")

        # Create interval arrays from Fuzznum and let NumPy broadcast them
        mds2 = np.full((1, 2), other.md, dtype=mds1.dtype)
        nmds2 = np.full((1, 2), other.nmd, dtype=nmds1.dtype)
        try:
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        except ValueError:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {fuzzarray_1.shape} and (1, 2)")
    else:
        raise TypeError(f"Unsupported operand type for vectorized operation: {type(other)}")


# --- IVQROFN Arithmetic Operations ---

@register_operation
class IVQROFNAddition(OperationMixin):
    """
    Implements the addition operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The addition of two IVQROFNs, A = ([md_A_l, md_A_u], [nmd_A_l, nmd_A_u]) and 
    B = ([md_B_l, md_B_u], [nmd_B_l, nmd_B_u]), is defined as:
    md_result = [S(md_A_l, md_B_l), S(md_A_u, md_B_u)]
    nmd_result = [T(nmd_A_l, nmd_B_l), T(nmd_A_u, nmd_B_u)]
    where S is a t-conorm and T is a t-norm.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'add'.
        """
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary addition operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply t-conorm to membership degree intervals directly
        # md_result = [S(md1_lower, md2_lower), S(md1_upper, md2_upper)]
        md_result = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        
        # Apply t-norm to non-membership degree intervals directly
        # nmd_result = [T(nmd1_lower, nmd2_lower), T(nmd1_upper, nmd2_upper)]
        nmd_result = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input IVQROFNs.
        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized addition for IVQROFN fuzzy arrays.

        Formula:
        - md_result = [S(md1_l, md2_l), S(md1_u, md2_u)] = t-conorm of membership intervals
        - nmd_result = [T(nmd1_l, nmd2_l), T(nmd1_u, nmd2_u)] = t-norm of non-membership intervals
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Apply t-conorm to membership intervals element-wise
        # Shape: (..., 2) -> (..., 2)
        md_res = tnorm.t_conorm(mds1, mds2)
        
        # Apply t-norm to non-membership intervals element-wise
        # Shape: (..., 2) -> (..., 2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNSubtraction(OperationMixin):
    """
    Implements the subtraction operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The subtraction of two IVQROFNs is defined based on specific conditions to ensure valid IVQROFN results.
    This operation extends the QROFN subtraction to interval-valued cases.
    If conditions are not met, the result defaults to ([0, 0], [1, 1]).
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'sub'.
        """
        return 'sub'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary subtraction operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance (minuend).
            strategy_2 (Any): The second IVQROFN strategy instance (subtrahend).
            tnorm (OperationTNorm): An instance of OperationTNorm (not directly used in this specific formula,
                                    but required by the OperationMixin interface).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.

        Notes:
            This operation is currently only applicable under the 'algebraic' t-norm;
            other norms are not supported for now.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The subtraction operation of 'ivqrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Subtraction operations of 'ivqrofn' based on other t-norms "
                          f"are temporarily performed using the 'algebraic' t-norm.")

        q = strategy_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Extract interval bounds
        md1_l, md1_u = strategy_1.md[0], strategy_1.md[1]
        nmd1_l, nmd1_u = strategy_1.nmd[0], strategy_1.nmd[1]
        md2_l, md2_u = strategy_2.md[0], strategy_2.md[1]
        nmd2_l, nmd2_u = strategy_2.nmd[0], strategy_2.nmd[1]

        # For interval subtraction, we need to check conditions for both bounds
        # Use the more restrictive bounds for validation
        condition_1_l = nmd1_l / nmd2_l if nmd2_l > epsilon else float('inf')
        condition_1_u = nmd1_u / nmd2_u if nmd2_u > epsilon else float('inf')
        
        condition_2_l = ((1 - md1_l ** q) / (1 - md2_l ** q)) ** (1 / q) if md2_l < 1 - epsilon else float('inf')
        condition_2_u = ((1 - md1_u ** q) / (1 - md2_u ** q)) ** (1 / q) if md2_u < 1 - epsilon else float('inf')

        # Check if all conditions for valid subtraction are met
        valid_lower = (epsilon <= condition_1_l <= 1 - epsilon and
                      epsilon <= condition_2_l <= 1 - epsilon and
                      condition_1_l <= condition_2_l)
        
        valid_upper = (epsilon <= condition_1_u <= 1 - epsilon and
                      epsilon <= condition_2_u <= 1 - epsilon and
                      condition_1_u <= condition_2_u)

        if valid_lower and valid_upper:
            # Calculate subtraction for both bounds
            md_l = ((md1_l ** q - md2_l ** q) / (1 - md2_l ** q)) ** (1 / q)
            md_u = ((md1_u ** q - md2_u ** q) / (1 - md2_u ** q)) ** (1 / q)
            nmd_l = nmd1_l / nmd2_l
            nmd_u = nmd1_u / nmd2_u
            
            md_result = np.array([md_l, md_u])
            nmd_result = np.array([nmd_l, nmd_u])
        else:
            # If conditions are not met, return default "empty" IVQROFN
            md_result = np.array([0.0, 0.0])
            nmd_result = np.array([1.0, 1.0])

        return {'md': md_result, 'nmd': nmd_result, 'q': q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized subtraction for IVQROFN fuzzy arrays.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The subtraction operation of 'ivqrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Vectorized subtraction will be performed using the 'algebraic' t-norm logic.")

        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Extract bounds for all elements
        # Shape: (..., 2) -> (...,) for each bound
        md1_l, md1_u = mds1[..., 0], mds1[..., 1]
        nmd1_l, nmd1_u = nmds1[..., 0], nmds1[..., 1]
        md2_l, md2_u = mds2[..., 0], mds2[..., 1]
        nmd2_l, nmd2_u = nmds2[..., 0], nmds2[..., 1]

        # Vectorized condition checking
        with np.errstate(divide='ignore', invalid='ignore'):
            condition_1_l = np.divide(nmd1_l, nmd2_l)
            condition_1_u = np.divide(nmd1_u, nmd2_u)
            condition_2_l = ((1 - md1_l ** q) / (1 - md2_l ** q)) ** (1 / q)
            condition_2_u = ((1 - md1_u ** q) / (1 - md2_u ** q)) ** (1 / q)

        # Create boolean masks for valid operations
        valid_mask_l = (
            (condition_1_l >= epsilon) & (condition_1_l <= 1 - epsilon) &
            (condition_2_l >= epsilon) & (condition_2_l <= 1 - epsilon) &
            (condition_1_l <= condition_2_l)
        )
        
        valid_mask_u = (
            (condition_1_u >= epsilon) & (condition_1_u <= 1 - epsilon) &
            (condition_2_u >= epsilon) & (condition_2_u <= 1 - epsilon) &
            (condition_1_u <= condition_2_u)
        )
        
        # Both bounds must be valid
        valid_mask = valid_mask_l & valid_mask_u

        # Calculate results using vectorized formulas
        with np.errstate(divide='ignore', invalid='ignore'):
            md_l_res = ((md1_l ** q - md2_l ** q) / (1 - md2_l ** q)) ** (1 / q)
            md_u_res = ((md1_u ** q - md2_u ** q) / (1 - md2_u ** q)) ** (1 / q)
            nmd_l_res = np.divide(nmd1_l, nmd2_l)
            nmd_u_res = np.divide(nmd1_u, nmd2_u)

        # Initialize result arrays with default values
        md_res = np.zeros_like(mds1)
        nmd_res = np.ones_like(nmds1)

        # Apply results only where valid
        md_res[..., 0] = np.where(valid_mask, md_l_res, 0.0)
        md_res[..., 1] = np.where(valid_mask, md_u_res, 0.0)
        nmd_res[..., 0] = np.where(valid_mask, nmd_l_res, 1.0)
        nmd_res[..., 1] = np.where(valid_mask, nmd_u_res, 1.0)

        # Handle NaNs
        np.nan_to_num(md_res, copy=False, nan=0.0)
        np.nan_to_num(nmd_res, copy=False, nan=1.0)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNMultiplication(OperationMixin):
    """
    Implements the multiplication operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The multiplication of two IVQROFNs, A = ([md_A_l, md_A_u], [nmd_A_l, nmd_A_u]) and 
    B = ([md_B_l, md_B_u], [nmd_B_l, nmd_B_u]), is defined as:
    md_result = [T(md_A_l, md_B_l), T(md_A_u, md_B_u)]
    nmd_result = [S(nmd_A_l, nmd_B_l), S(nmd_A_u, nmd_B_u)]
    where T is a t-norm and S is a t-conorm.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'mul'.
        """
        return 'mul'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary multiplication operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply t-norm to membership degree intervals directly
        # md_result = [T(md1_lower, md2_lower), T(md1_upper, md2_upper)]
        md_result = tnorm.t_norm(strategy_1.md, strategy_2.md)
        
        # Apply t-conorm to non-membership degree intervals directly
        # nmd_result = [S(nmd1_lower, nmd2_lower), S(nmd1_upper, nmd2_upper)]
        nmd_result = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input IVQROFNs.
        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized multiplication for IVQROFN fuzzy arrays.

        Formula:
        - md_result = [T(md1_l, md2_l), T(md1_u, md2_u)] = t-norm of membership intervals
        - nmd_result = [S(nmd1_l, nmd2_l), S(nmd1_u, nmd2_u)] = t-conorm of non-membership intervals
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Apply t-norm to membership intervals element-wise
        md_res = tnorm.t_norm(mds1, mds2)
        
        # Apply t-conorm to non-membership intervals element-wise
        nmd_res = tnorm.t_conorm(nmds1, nmds2)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNDivision(OperationMixin):
    """
    Implements the division operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The division of two IVQROFNs is defined based on specific conditions to ensure valid IVQROFN results.
    This operation extends the QROFN division to interval-valued cases.
    If conditions are not met, the result defaults to ([1, 1], [0, 0]).
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'div'.
        """
        return 'div'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary division operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance (dividend).
            strategy_2 (Any): The second IVQROFN strategy instance (divisor).
            tnorm (OperationTNorm): An instance of OperationTNorm (not directly used in this specific formula,
                                    but required by the OperationMixin interface).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The division operation of 'ivqrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Division operations of 'ivqrofn' based on other t-norms "
                          f"are temporarily performed using the 'algebraic' t-norm.")

        q = strategy_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Extract interval bounds
        md1_l, md1_u = strategy_1.md[0], strategy_1.md[1]
        nmd1_l, nmd1_u = strategy_1.nmd[0], strategy_1.nmd[1]
        md2_l, md2_u = strategy_2.md[0], strategy_2.md[1]
        nmd2_l, nmd2_u = strategy_2.nmd[0], strategy_2.nmd[1]

        # For interval division, check conditions for both bounds
        condition_1_l = md1_l / md2_l if md2_l > epsilon else float('inf')
        condition_1_u = md1_u / md2_u if md2_u > epsilon else float('inf')
        
        condition_2_l = ((1 - nmd1_l ** q) / (1 - nmd2_l ** q)) ** (1 / q) if nmd2_l < 1 - epsilon else float('inf')
        condition_2_u = ((1 - nmd1_u ** q) / (1 - nmd2_u ** q)) ** (1 / q) if nmd2_u < 1 - epsilon else float('inf')

        # Check if all conditions for valid division are met
        valid_lower = (epsilon <= condition_1_l <= 1 - epsilon and
                      epsilon <= condition_2_l <= 1 - epsilon and
                      condition_1_l <= condition_2_l)
        
        valid_upper = (epsilon <= condition_1_u <= 1 - epsilon and
                      epsilon <= condition_2_u <= 1 - epsilon and
                      condition_1_u <= condition_2_u)

        if valid_lower and valid_upper:
            # Calculate division for both bounds
            md_l = md1_l / md2_l
            md_u = md1_u / md2_u
            nmd_l = ((nmd1_l ** q - nmd2_l ** q) / (1 - nmd2_l ** q)) ** (1 / q)
            nmd_u = ((nmd1_u ** q - nmd2_u ** q) / (1 - nmd2_u ** q)) ** (1 / q)
            
            md_result = np.array([md_l, md_u])
            nmd_result = np.array([nmd_l, nmd_u])
        else:
            # If conditions are not met, return default "full" IVQROFN
            md_result = np.array([1.0, 1.0])
            nmd_result = np.array([0.0, 0.0])

        return {'md': md_result, 'nmd': nmd_result, 'q': q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized division for IVQROFN fuzzy arrays.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The division operation of 'ivqrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Vectorized division will be performed using the 'algebraic' t-norm logic.")

        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Extract bounds for all elements
        md1_l, md1_u = mds1[..., 0], mds1[..., 1]
        nmd1_l, nmd1_u = nmds1[..., 0], nmds1[..., 1]
        md2_l, md2_u = mds2[..., 0], mds2[..., 1]
        nmd2_l, nmd2_u = nmds2[..., 0], nmds2[..., 1]

        # Vectorized condition checking
        with np.errstate(divide='ignore', invalid='ignore'):
            condition_1_l = np.divide(md1_l, md2_l)
            condition_1_u = np.divide(md1_u, md2_u)
            condition_2_l = ((1 - nmd1_l ** q) / (1 - nmd2_l ** q)) ** (1 / q)
            condition_2_u = ((1 - nmd1_u ** q) / (1 - nmd2_u ** q)) ** (1 / q)

        # Create boolean masks for valid operations
        valid_mask_l = (
            (condition_1_l >= epsilon) & (condition_1_l <= 1 - epsilon) &
            (condition_2_l >= epsilon) & (condition_2_l <= 1 - epsilon) &
            (condition_1_l <= condition_2_l)
        )
        
        valid_mask_u = (
            (condition_1_u >= epsilon) & (condition_1_u <= 1 - epsilon) &
            (condition_2_u >= epsilon) & (condition_2_u <= 1 - epsilon) &
            (condition_1_u <= condition_2_u)
        )
        
        # Both bounds must be valid
        valid_mask = valid_mask_l & valid_mask_u

        # Calculate results using vectorized formulas
        with np.errstate(divide='ignore', invalid='ignore'):
            md_l_res = np.divide(md1_l, md2_l)
            md_u_res = np.divide(md1_u, md2_u)
            nmd_l_res = ((nmd1_l ** q - nmd2_l ** q) / (1 - nmd2_l ** q)) ** (1 / q)
            nmd_u_res = ((nmd1_u ** q - nmd2_u ** q) / (1 - nmd2_u ** q)) ** (1 / q)

        # Initialize result arrays with default values
        md_res = np.ones_like(mds1)
        nmd_res = np.zeros_like(nmds1)

        # Apply results only where valid
        md_res[..., 0] = np.where(valid_mask, md_l_res, 1.0)
        md_res[..., 1] = np.where(valid_mask, md_u_res, 1.0)
        nmd_res[..., 0] = np.where(valid_mask, nmd_l_res, 0.0)
        nmd_res[..., 1] = np.where(valid_mask, nmd_u_res, 0.0)

        # Handle NaNs
        np.nan_to_num(md_res, copy=False, nan=1.0)
        np.nan_to_num(nmd_res, copy=False, nan=0.0)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNPower(OperationMixin):
    """
    Implements the power operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    This operation calculates A^operand, where A is an IVQROFN and operand is a scalar.
    It leverages the generator and pseudo-inverse functions of the underlying t-norm.
    The operation is applied element-wise to both bounds of the intervals.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'pow'.
        """
        return 'pow'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary power operation on an IVQROFN strategy with a scalar operand.

        Args:
            strategy (Any): The IVQROFN strategy instance (base).
            operand (Union[int, float]): The scalar exponent.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply power operation to both bounds of membership degree interval
        md_result = np.array([
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.md[0])),  # lower bound
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.md[1]))   # upper bound
        ])
        
        # Apply power operation to both bounds of non-membership degree interval
        nmd_result = np.array([
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd[0])),  # lower bound
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd[1]))   # upper bound
        ])

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized power operation for IVQROFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        # Apply power operation to all interval bounds element-wise
        md_res = tnorm.g_inv_func(operand * tnorm.g_func(mds))
        nmd_res = tnorm.f_inv_func(operand * tnorm.f_func(nmds))

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNTimes(OperationMixin):
    """
    Implements scalar multiplication for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    This operation calculates operand * A, where A is an IVQROFN and operand is a scalar.
    It uses the T-norm generator framework applied to interval bounds.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'tim' (for times).
        """
        return 'tim'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary scalar multiplication operation on an IVQROFN strategy with a scalar operand.

        Args:
            strategy (Any): The IVQROFN strategy instance.
            operand (Union[int, float]): The scalar multiplier.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply times operation to both bounds of membership degree interval
        md_result = np.array([
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.md[0])),  # lower bound
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.md[1]))   # upper bound
        ])
        
        # Apply times operation to both bounds of non-membership degree interval
        nmd_result = np.array([
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd[0])),  # lower bound
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd[1]))   # upper bound
        ])

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized scalar multiplication for IVQROFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        # Apply times operation to all interval bounds element-wise
        md_res = tnorm.f_inv_func(operand * tnorm.f_func(mds))
        nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(nmds))

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class IVQROFNExponential(OperationMixin):
    """
    Implements the exponential operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    This operation calculates exp(operand * A), where A is an IVQROFN and operand is a scalar.
    Note: The implementation is currently a placeholder and may have limitations.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'exp'.
        """
        return 'exp'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary exponential operation on an IVQROFN strategy with a scalar operand.

        Args:
            strategy (Any): The IVQROFN strategy instance.
            operand (Union[int, float]): The scalar multiplier for the exponent.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply exponential operation to both bounds of membership degree interval
        md_result = np.array([
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.md[0])),  # lower bound
            tnorm.f_inv_func(operand * tnorm.f_func(strategy.md[1]))   # upper bound
        ])
        
        # Apply exponential operation to both bounds of non-membership degree interval
        nmd_result = np.array([
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd[0])),  # lower bound
            tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd[1]))   # upper bound
        ])

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy.q}

    @experimental
    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized exponential operation for IVQROFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.f_inv_func(operand * tnorm.f_func(mds))
        nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(nmds))

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class IVQROFNLogarithmic(OperationMixin):
    """
    Implements the logarithmic operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    This operation calculates log_operand(A), where A is an IVQROFN and operand is the base.
    Note: The implementation is currently a placeholder and may have limitations.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'log'.
        """
        return 'log'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary logarithmic operation on an IVQROFN strategy with a scalar operand (base).

        Args:
            strategy (Any): The IVQROFN strategy instance.
            operand (Union[int, float]): The base of the logarithm.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply logarithmic operation to both bounds of membership degree interval
        md_result = np.array([
            tnorm.f_inv_func(tnorm.f_func(strategy.md[0]) / operand),  # lower bound
            tnorm.f_inv_func(tnorm.f_func(strategy.md[1]) / operand)   # upper bound
        ])
        
        # Apply logarithmic operation to both bounds of non-membership degree interval
        nmd_result = np.array([
            tnorm.g_inv_func(tnorm.g_func(strategy.nmd[0]) / operand),  # lower bound
            tnorm.g_inv_func(tnorm.g_func(strategy.nmd[1]) / operand)   # upper bound
        ])

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy.q}

    @experimental
    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized logarithmic operation for IVQROFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.f_inv_func(tnorm.f_func(mds) / operand)
        nmd_res = tnorm.g_inv_func(tnorm.g_func(nmds) / operand)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# --- IVQROFN Comparison Operations ---

@register_operation
class IVQROFNGreaterThan(OperationMixin):
    """
    Implements the greater than (>) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    An IVQROFN A is considered greater than B based on their score functions.
    For interval-valued fuzzy numbers, we use the mean of interval bounds for comparison:
    Score(A) = (md_mean^q - nmd_mean^q)
    where md_mean = (md_lower + md_upper) / 2, nmd_mean = (nmd_lower + nmd_upper) / 2
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'gt'.
        """
        return 'gt'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater than comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is strictly greater than strategy_2.
        """
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison: Score = md_mean^q - nmd_mean^q
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': score1 > score2}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized greater than comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return score1 > score2


@register_operation
class IVQROFNLessThan(OperationMixin):
    """
    Implements the less than (<) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    An IVQROFN A is considered less than B based on their score functions.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'lt'.
        """
        return 'lt'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less than comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is strictly less than strategy_2.
        """
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': score1 < score2}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized less than comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return score1 < score2


@register_operation
class IVQROFNEquals(OperationMixin):
    """
    Implements the equality (==) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    Two IVQROFNs are considered equal if their scores are approximately equal,
    considering floating-point precision.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'eq'.
        """
        return 'eq'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the equality comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is approximately equal to strategy_2.
        """
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': abs(score1 - score2) < epsilon}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized equality comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return np.abs(score1 - score2) < epsilon


@register_operation
class IVQROFNGreaterEquals(OperationMixin):
    """
    Implements the greater than or equal to (>=) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    An IVQROFN A is considered greater than or equal to B if it is strictly greater than B
    or approximately equal to B.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'ge'.
        """
        return 'ge'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater than or equal to comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is greater than or equal to strategy_2.
        """
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': score1 > score2 or abs(score1 - score2) < epsilon}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized greater than or equal comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return score1 >= score2


@register_operation
class IVQROFNLessEquals(OperationMixin):
    """
    Implements the less than or equal to (<=) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    An IVQROFN A is considered less than or equal to B if it is strictly less than B
    or approximately equal to B.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'le'.
        """
        return 'le'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less than or equal to comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is less than or equal to strategy_2.
        """
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': score1 < score2 or abs(score1 - score2) < epsilon}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized less than or equal comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return score1 <= score2


@register_operation
class IVQROFNNotEquals(OperationMixin):
    """
    Implements the not equal (!=) comparison for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    Two IVQROFNs are considered not equal if their scores are not approximately equal.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'ne'.
        """
        return 'ne'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the not equal comparison between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is not equal to strategy_2.
        """
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for both intervals
        md1_mean = (strategy_1.md[0] + strategy_1.md[1]) / 2
        nmd1_mean = (strategy_1.nmd[0] + strategy_1.nmd[1]) / 2
        md2_mean = (strategy_2.md[0] + strategy_2.md[1]) / 2
        nmd2_mean = (strategy_2.nmd[0] + strategy_2.nmd[1]) / 2
        
        # Use score function for comparison
        q = strategy_1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return {'value': abs(score1 - score2) >= epsilon}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized not equal comparison for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON
        
        # Calculate mean values for all intervals
        md1_mean = (mds1[..., 0] + mds1[..., 1]) / 2
        nmd1_mean = (nmds1[..., 0] + nmds1[..., 1]) / 2
        md2_mean = (mds2[..., 0] + mds2[..., 1]) / 2
        nmd2_mean = (nmds2[..., 0] + nmds2[..., 1]) / 2
        
        # Calculate scores and compare
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q
        
        return np.abs(score1 - score2) >= epsilon


# --- IVQROFN Set Operations ---

@register_operation
class IVQROFNIntersection(OperationMixin):
    """
    Implements the intersection operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The intersection of two IVQROFNs, A = ([md_A_l, md_A_u], [nmd_A_l, nmd_A_u]) and 
    B = ([md_B_l, md_B_u], [nmd_B_l, nmd_B_u]), is defined as:
    md_result = [T(md_A_l, md_B_l), T(md_A_u, md_B_u)]
    nmd_result = [S(nmd_A_l, nmd_B_l), S(nmd_A_u, nmd_B_u)]
    where T is a t-norm and S is a t-conorm.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'intersection'.
        """
        return 'intersection'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary intersection operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply t-norm to membership degree intervals (minimum) directly
        md_result = tnorm.t_norm(strategy_1.md, strategy_2.md)
        
        # Apply t-conorm to non-membership degree intervals (maximum) directly
        nmd_result = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized intersection for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Apply t-norm to membership intervals, t-conorm to non-membership intervals
        md_res = tnorm.t_norm(mds1, mds2)
        nmd_res = tnorm.t_conorm(nmds1, nmds2)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNUnion(OperationMixin):
    """
    Implements the union operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The union of two IVQROFNs, A = ([md_A_l, md_A_u], [nmd_A_l, nmd_A_u]) and 
    B = ([md_B_l, md_B_u], [nmd_B_l, nmd_B_u]), is defined as:
    md_result = [S(md_A_l, md_B_l), S(md_A_u, md_B_u)]
    nmd_result = [T(nmd_A_l, nmd_B_l), T(nmd_A_u, nmd_B_u)]
    where S is a t-conorm and T is a t-norm.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'union'.
        """
        return 'union'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary union operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Apply t-conorm to membership degree intervals (maximum) directly
        md_result = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        
        # Apply t-norm to non-membership degree intervals (minimum) directly
        nmd_result = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized union for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Apply t-conorm to membership intervals, t-norm to non-membership intervals
        md_res = tnorm.t_conorm(mds1, mds2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNComplement(OperationMixin):
    """
    Implements the complement operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The complement of an IVQROFN A = ([md_l, md_u], [nmd_l, nmd_u]) is defined as:
    complement(A) = ([nmd_l, nmd_u], [md_l, md_u])
    This swaps the membership and non-membership degree intervals.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'complement'.
        """
        return 'complement'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary complement operation on an IVQROFN strategy.

        Args:
            strategy (Any): The IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in complement).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Swap membership and non-membership intervals
        return {'md': strategy.nmd.copy(), 'nmd': strategy.md.copy(), 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized complement for IVQROFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        # Swap membership and non-membership arrays
        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(nmds, mds, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNDifference(OperationMixin):
    """
    Implements the difference operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The difference of two IVQROFNs A and B is defined as A ∩ ¬B,
    where ¬B is the complement of B.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'difference'.
        """
        return 'difference'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary difference operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance (A).
            strategy_2 (Any): The second IVQROFN strategy instance (B).
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # A ∩ ¬B = A ∩ (nmd_B, md_B) - direct computation
        # md_result = T(md_A, nmd_B) where nmd_B is the complement of B's membership
        md_result = tnorm.t_norm(strategy_1.md, strategy_2.nmd)
        
        # nmd_result = S(nmd_A, md_B) where md_B becomes the non-membership in ¬B
        nmd_result = tnorm.t_conorm(strategy_1.nmd, strategy_2.md)

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized difference for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # A ∩ ¬B: use B's complement (nmds2, mds2)
        md_res = tnorm.t_norm(mds1, nmds2)  # T(md_A, nmd_B)
        nmd_res = tnorm.t_conorm(nmds1, mds2)  # S(nmd_A, md_B)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class IVQROFNSymmetricDifference(OperationMixin):
    """
    Implements the symmetric difference operation for Interval-Valued Q-Rung Orthopair Fuzzy Numbers (IVQROFNs).

    The symmetric difference of two IVQROFNs A and B is defined as (A ∪ B) ∩ ¬(A ∩ B),
    which represents elements that belong to either A or B but not both.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'symdiff'.
        """
        return 'symdiff'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'ivqrofn'.
        """
        return ['ivqrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary symmetric difference operation between two IVQROFN strategies.

        Args:
            strategy_1 (Any): The first IVQROFN strategy instance.
            strategy_2 (Any): The second IVQROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting IVQROFN.
        """
        # Step 1: Calculate A ∪ B directly
        union_md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        union_nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)
        
        # Step 2: Calculate A ∩ B directly
        intersection_md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        intersection_nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)
        
        # Step 3: Calculate (A ∪ B) ∩ ¬(A ∩ B) directly
        # ¬(A ∩ B) swaps md and nmd of the intersection
        md_result = tnorm.t_norm(union_md, intersection_nmd)  # T(union_md, ¬intersection_md)
        nmd_result = tnorm.t_conorm(union_nmd, intersection_md)  # S(union_nmd, ¬intersection_nmd)

        return {'md': md_result, 'nmd': nmd_result, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized symmetric difference for IVQROFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Step 1: A ∪ B
        union_md = tnorm.t_conorm(mds1, mds2)
        union_nmd = tnorm.t_norm(nmds1, nmds2)
        
        # Step 2: A ∩ B
        intersection_md = tnorm.t_norm(mds1, mds2)
        intersection_nmd = tnorm.t_conorm(nmds1, nmds2)
        
        # Step 3: (A ∪ B) ∩ ¬(A ∩ B)
        # ¬(A ∩ B) has swapped md and nmd
        md_res = tnorm.t_norm(union_md, intersection_nmd)
        nmd_res = tnorm.t_conorm(union_nmd, intersection_md)

        backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)