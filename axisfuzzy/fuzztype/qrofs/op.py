#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
This module defines various operations for Q-Rung Orthopair Fuzzy Numbers (QROFNs).
It includes arithmetic, comparison, and set-theoretic operations, all implemented
as subclasses of `OperationMixin` and registered with the global operation registry.

Each operation class specifies:
- The operation name (e.g., 'add', 'mul', 'gt').
- The fuzzy number types it supports (currently 'qrofn').
- The core logic for executing the operation, often leveraging `OperationTNorm`
  for t-norm and t-conorm calculations.
"""

import warnings
from typing import List, Any, Dict, Union, Optional

import numpy as np

from ...config import get_config

from ...core import (
    Fuzznum,
    Fuzzarray,
    get_fuzztype_backend,
    OperationTNorm,
    OperationMixin,
    register_operation
)

from ...utils import experimental


def _prepare_operands(
        fuzzarray_1: Fuzzarray,
        other: Any) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Helper to get component arrays from operands."""
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

        # mds2 = np.full_like(mds1, other.md)
        # nmds2 = np.full_like(nmds1, other.nmd)

        # Create arrays from Fuzznum and let NumPy broadcast them
        mds2 = np.full((1,), other.md, dtype=mds1.dtype)
        nmds2 = np.full((1,), other.nmd, dtype=nmds1.dtype)
        try:
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        except ValueError:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {fuzzarray_1.shape} and ({mds2.shape}, {nmds2.shape})")
    else:
        raise TypeError(f"Unsupported operand type for vectorized operation: {type(other)}")


# --- QROFN Arithmetic Operations ---

@register_operation
class QROFNAddition(OperationMixin):
    """
    Implements the addition operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The addition of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined as:
    md = S(md_A, md_B)
    nmd = T(nmd_A, nmd_B)
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary addition operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-conorm (S-norm).
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        # Non-membership degree is calculated using the t-norm (T-norm).
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        High-performance vectorized addition for QROFN fuzzy arrays.

        Formula:
        - md_result = S(md1, md2) = t-conorm of membership degrees
        - nmd_result = T(nmd1, nmd2) = t-norm of non-membership degrees
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = S(md1, md2), nmd = T(nmd1, nmd2)
        md_res = tnorm.t_conorm(mds1, mds2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNSubtraction(OperationMixin):
    """
    Implements the subtraction operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The subtraction of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined based on specific conditions to ensure valid QROFN results.
    If conditions are not met, the result defaults to (0, 1).
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary subtraction operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance (minuend).
            strategy_2 (Any): The second QROFN strategy instance (subtrahend).
            tnorm (OperationTNorm): An instance of OperationTNorm (not directly used in this specific formula,
                                    but required by the OperationMixin interface).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.

        Notes:
            This operation is currently only applicable under the 'algebraic' t-norm;
            other norms are not supported for now.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The subtraction operation of 'qrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Subtraction operations of 'qrofn' based on other t-norms "
                          f"are temporarily performed using the 'algebraic' t-norm.")

        q = strategy_1.q  # Get the q-rung value.

        # Calculate conditions for valid subtraction.
        # These conditions ensure that the resulting membership and non-membership
        # degrees remain within the [0,1] range and satisfy the QROFN constraint.
        condition_1 = strategy_1.nmd / strategy_2.nmd
        condition_2 = ((1 - strategy_1.md ** q) / (1 - strategy_2.md ** q)) ** (1 / q)
        epsilon = get_config().DEFAULT_EPSILON  # Use a small epsilon for floating-point comparisons.

        # Check if the conditions for valid subtraction are met.
        if (epsilon <= condition_1 <= 1 - epsilon
                and epsilon <= condition_2 <= 1 - epsilon
                and condition_1 <= condition_2):
            # If conditions are met, calculate md and nmd using the subtraction formulas.
            md = ((strategy_1.md ** q - strategy_2.md ** q) / (1 - strategy_2.md ** q)) ** (1 / q)
            nmd = strategy_1.nmd / strategy_2.nmd
        else:
            # If conditions are not met, return a default "empty" or "invalid" QROFN.
            # This typically means the operation is not well-defined for the given inputs.
            md = 0.
            nmd = 1.

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The subtraction operation of 'qrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Vectorized subtraction will be performed using the 'algebraic' t-norm logic.")

        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Vectorize conditions
        # Use np.divide to handle potential division by zero safely
        with np.errstate(divide='ignore', invalid='ignore'):
            condition_1 = np.divide(nmds1, nmds2)
            condition_2 = ((1 - mds1 ** q) / (1 - mds2 ** q)) ** (1 / q)

        # Create a boolean mask where the subtraction is valid
        valid_mask = (
            (condition_1 >= epsilon) & (condition_1 <= 1 - epsilon) &
            (condition_2 >= epsilon) & (condition_2 <= 1 - epsilon) &
            (condition_1 <= condition_2)
        )

        # Calculate results using vectorized formulas
        with np.errstate(divide='ignore', invalid='ignore'):
            md_res_valid = ((mds1 ** q - mds2 ** q) / (1 - mds2 ** q)) ** (1 / q)
            nmd_res_valid = np.divide(nmds1, nmds2)

        # Initialize result arrays with default values (0, 1)
        md_res = np.zeros_like(mds1)
        nmd_res = np.ones_like(nmds1)

        # Use np.where to apply results only where the mask is True
        np.copyto(md_res, md_res_valid, where=valid_mask)
        np.copyto(nmd_res, nmd_res_valid, where=valid_mask)

        # Handle NaNs that might result from invalid calculations
        np.nan_to_num(md_res, copy=False, nan=0.0)
        np.nan_to_num(nmd_res, copy=False, nan=1.0)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNMultiplication(OperationMixin):
    """
    Implements the multiplication operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The multiplication of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined as:
    md = T(md_A, md_B)
    nmd = S(nmd_A, nmd_B)
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary multiplication operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-norm (T-norm).
        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        # Non-membership degree is calculated using the t-conorm (S-norm).
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = T(md1, md2), nmd = S(nmd1, nmd2)
        md_res = tnorm.t_norm(mds1, mds2)
        nmd_res = tnorm.t_conorm(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNDivision(OperationMixin):
    """
    Implements the division operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The division of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined based on specific conditions to ensure valid QROFN results.
    If conditions are not met, the result defaults to (1, 0).
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary division operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance (dividend).
            strategy_2 (Any): The second QROFN strategy instance (divisor).
            tnorm (OperationTNorm): An instance of OperationTNorm (not directly used in this specific formula,
                                    but required by the OperationMixin interface).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The division operation of 'qrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Division operations of 'qrofn' based on other t-norms "
                          f"are temporarily performed using the 'algebraic' t-norm.")

        q = strategy_1.q  # Get the q-rung value.

        # Calculate conditions for valid division.
        # These conditions ensure that the resulting membership and non-membership
        # degrees remain within the [0,1] range and satisfy the QROFN constraint.
        condition_1 = strategy_1.md / strategy_2.md
        condition_2 = ((1 - strategy_1.nmd ** q) / (1 - strategy_2.nmd ** q)) ** (1 / q)
        epsilon = get_config().DEFAULT_EPSILON  # Use a small epsilon for floating-point comparisons.

        # Check if the conditions for valid division are met.
        if (epsilon <= condition_1 <= 1 - epsilon
                and epsilon <= condition_2 <= 1 - epsilon
                and condition_1 <= condition_2):
            # If conditions are met, calculate md and nmd using the division formulas.
            md = strategy_1.md / strategy_2.md
            nmd = ((strategy_1.nmd ** q - strategy_2.nmd ** q) / (1 - strategy_2.nmd ** q)) ** (1 / q)
        else:
            # If conditions are not met, return a default "full" or "invalid" QROFN.
            # This typically means the operation is not well-defined for the given inputs.
            md = 1.
            nmd = 0.

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The division operation of 'qrofn' is currently only applicable to 'algebraic' t-norm. "
                          f"Vectorized division will be performed using the 'algebraic' t-norm logic.")

        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)
        q = fuzzarray_1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Vectorize conditions
        with np.errstate(divide='ignore', invalid='ignore'):
            condition_1 = np.divide(mds1, mds2)
            condition_2 = ((1 - nmds1 ** q) / (1 - nmds2 ** q)) ** (1 / q)

        # Create a boolean mask where the division is valid
        valid_mask = (
            (condition_1 >= epsilon) & (condition_1 <= 1 - epsilon) &
            (condition_2 >= epsilon) & (condition_2 <= 1 - epsilon) &
            (condition_1 <= condition_2)
        )

        # Calculate results using vectorized formulas
        with np.errstate(divide='ignore', invalid='ignore'):
            md_res_valid = np.divide(mds1, mds2)
            nmd_res_valid = ((nmds1 ** q - nmds2 ** q) / (1 - nmds2 ** q)) ** (1 / q)

        # Initialize result arrays with default values (1, 0)
        md_res = np.ones_like(mds1)
        nmd_res = np.zeros_like(nmds1)

        # Use np.where to apply results only where the mask is True
        np.copyto(md_res, md_res_valid, where=valid_mask)
        np.copyto(nmd_res, nmd_res_valid, where=valid_mask)

        # Handle NaNs
        np.nan_to_num(md_res, copy=False, nan=1.0)
        np.nan_to_num(nmd_res, copy=False, nan=0.0)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNPower(OperationMixin):
    """
    Implements the power operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This operation calculates A^operand, where A is a QROFN and operand is a scalar.
    It leverages the generator and pseudo-inverse functions of the underlying t-norm.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary power operation on a QROFN strategy with a scalar operand.

        Args:
            strategy (Any): The QROFN strategy instance (base).
            operand (Union[int, float]): The scalar exponent.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.g_inv_func(operand * tnorm.g_func(strategy.md))
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd))

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:

        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.g_inv_func(operand * tnorm.g_func(mds))
        nmd_res = tnorm.f_inv_func(operand * tnorm.f_func(nmds))

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNTimes(OperationMixin):
    """
    Implements scalar multiplication for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This operation calculates operand * A, where A is a QROFN and operand is a scalar.
    It uses the same underlying formulas as the power operation, as scalar multiplication
    in some fuzzy number contexts can be defined analogously to exponentiation.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary scalar multiplication operation on a QROFN strategy with a scalar operand.

        Args:
            strategy (Any): The QROFN strategy instance.
            operand (Union[int, float]): The scalar multiplier.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.f_inv_func(operand * tnorm.f_func(mds))
        nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(nmds))

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class QROFNExponential(OperationMixin):
    """
    Implements the exponential operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This operation calculates exp(operand * A), where A is a QROFN and operand is a scalar.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary exponential operation on a QROFN strategy with a scalar operand.

        Args:
            strategy (Any): The QROFN strategy instance.
            operand (Union[int, float]): The scalar multiplier for the exponent.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    @experimental
    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.f_inv_func(operand * tnorm.f_func(mds))
        nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(nmds))

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class QROFNLogarithmic(OperationMixin):
    """
    Implements the logarithmic operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    This operation calculates log_operand(A), where A is a QROFN and operand is the base.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary logarithmic operation on a QROFN strategy with a scalar operand (base).

        Args:
            strategy (Any): The QROFN strategy instance.
            operand (Union[int, float]): The base of the logarithm.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    generator and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.f_inv_func(tnorm.f_func(strategy.md) / operand)
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(tnorm.g_func(strategy.nmd) / operand)

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    @experimental
    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds, nmds = fuzzarray.backend.get_component_arrays()

        md_res = tnorm.f_inv_func(tnorm.f_func(mds) / operand)
        nmd_res = tnorm.g_inv_func(tnorm.g_func(nmds) / operand)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# --- QROFN Comparison Operations ---

@register_operation
class QROFNGreaterThan(OperationMixin):
    """
    Implements the greater than (>) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    A QROFN A is considered greater than B if its membership degree is greater
    and its non-membership degree is less.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater than comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is strictly greater than strategy_2.
        """
        return {'value': strategy_1.md - strategy_1.nmd > strategy_2.md - strategy_2.nmd}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(mds1 - nmds1 > mds2 - nmds2, True, False)


@register_operation
class QROFNLessThan(OperationMixin):
    """
    Implements the less than (<) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    A QROFN A is considered less than B if its membership degree is less
    and its non-membership degree is greater.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less than comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is strictly less than strategy_2.
        """
        return {'value': strategy_1.md - strategy_1.nmd < strategy_2.md - strategy_2.nmd}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(mds1 - nmds1 < mds2 - nmds2, True, False)


@register_operation
class QROFNEquals(OperationMixin):
    """
    Implements the equality (==) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    Two QROFNs are considered equal if their membership and non-membership degrees
    are approximately equal, considering floating-point precision.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the equality comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is approximately equal to strategy_2.
        """
        config = get_config()
        value = abs((strategy_1.md - strategy_1.nmd) - (strategy_2.md - strategy_2.nmd)) < config.DEFAULT_EPSILON

        return {'value': value}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(abs((mds1 - nmds1) - (mds2 - nmds2)) < get_config().DEFAULT_EPSILON, True, False)


@register_operation
class QROFNGreaterEquals(OperationMixin):
    """
    Implements the greater than or equal to (>=) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    A QROFN A is considered greater than or equal to B if it is strictly greater than B
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the greater than or equal to comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is greater than or equal to strategy_2.
        """
        config = get_config()
        value = (strategy_1.md - strategy_1.nmd > strategy_2.md - strategy_2.nmd) or \
            abs((strategy_1.md - strategy_1.nmd) - (strategy_2.md - strategy_2.nmd)) < config.DEFAULT_EPSILON

        return {'value': value}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(mds1 - nmds1 >= mds2 - nmds2, True, False)


@register_operation
class QROFNLessEquals(OperationMixin):
    """
    Implements the less than or equal to (<=) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    A QROFN A is considered less than or equal to B if it is strictly less than B
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the less than or equal to comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is less than or equal to strategy_2.
        """
        config = get_config()
        value = (strategy_1.md - strategy_1.nmd < strategy_2.md - strategy_2.nmd) or \
            abs((strategy_1.md - strategy_1.nmd) - (strategy_2.md - strategy_2.nmd)) < config.DEFAULT_EPSILON

        return {'value': value}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(mds1 - nmds1 <= mds2 - nmds2, True, False)


@register_operation
class QROFNNotEquals(OperationMixin):
    """
    Implements the not equal to (!=) comparison for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    Two QROFNs are considered not equal if they are not approximately equal.
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes the not equal to comparison between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this comparison).

        Returns:
            Dict[str, bool]: A dictionary with a 'value' key, indicating whether
                             strategy_1 is not equal to strategy_2.
        """
        config = get_config()
        # Not equal to logic: NOT (abs(md_1 - md_2) < epsilon AND abs(nmd_1 - nmd_2) < epsilon).
        value = abs((strategy_1.md - strategy_1.nmd) - (strategy_2.md - strategy_2.nmd)) > config.DEFAULT_EPSILON

        return {'value': value}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   fuzzarray_2: Fuzzarray,
                                   tnorm: OperationTNorm) -> np.ndarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, fuzzarray_2)

        return np.where(abs((mds1 - nmds1) - (mds2 - nmds2)) > get_config().DEFAULT_EPSILON, True, False)


# --- QROFN Set Operations ---

@register_operation
class QROFNIntersection(OperationMixin):
    """
    Implements the intersection (AND) operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The intersection of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined as:
    md = T(md_A, md_B)
    nmd = S(nmd_A, nmd_B)
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary intersection operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-norm (T-norm).
        md = tnorm.t_norm(strategy_1.md, strategy_2.md)
        # Non-membership degree is calculated using the t-conorm (S-norm).
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = T(md1, md2), nmd = S(nmd1, nmd2)
        md_res = tnorm.t_norm(mds1, mds2)
        nmd_res = tnorm.t_conorm(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNUnion(OperationMixin):
    """
    Implements the union (OR) operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The union of two QROFNs, A = (md_A, nmd_A) and B = (md_B, nmd_B),
    is defined as:
    md = S(md_A, md_B)
    nmd = T(nmd_A, nmd_B)
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary union operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm and t-conorm calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-conorm (S-norm).
        md = tnorm.t_conorm(strategy_1.md, strategy_2.md)
        # Non-membership degree is calculated using the t-norm (T-norm).
        nmd = tnorm.t_norm(strategy_1.nmd, strategy_2.nmd)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = S(md1, md2), nmd = T(nmd1, nmd2)
        md_res = tnorm.t_conorm(mds1, mds2)
        nmd_res = tnorm.t_norm(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNComplement(OperationMixin):
    """
    Implements the complement (NOT) operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The complement of a QROFN A = (md_A, nmd_A) is defined as (nmd_A, md_A).
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary complement operation on a QROFN strategy.

        Args:
            strategy (Any): The QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm (not used in this operation).

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Complement operation swaps membership and non-membership degrees.
        md = strategy.nmd
        nmd = strategy.md

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds, nmds = fuzzarray_1.backend.get_component_arrays()

        # Formula: md' = nmd, nmd' = md
        md_res = nmds.copy()
        nmd_res = mds.copy()

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNImplication(OperationMixin):
    """
    Implements the implication operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The implication A -> B for QROFNs is defined using t-conorms, t-norms,
    and the generator/pseudo-inverse functions of the underlying t-norm.
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'implication'.
        """
        return 'implication'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary implication operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance (antecedent).
            strategy_2 (Any): The second QROFN strategy instance (consequent).
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm, t-conorm, generator, and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-conorm (S-norm).
        md = tnorm.t_conorm(strategy_1.nmd, strategy_2.md)
        # Non-membership degree is calculated using the t-norm (T-norm).
        nmd = tnorm.t_norm(strategy_1.md, strategy_2.nmd)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = S(nmd1, md2), nmd = T(md1, nmd2)
        md_res = tnorm.t_conorm(nmds1, mds2)
        nmd_res = tnorm.t_norm(mds1, nmds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNEquivalence(OperationMixin):
    """
    Implements the equivalence operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The equivalence A <-> B for QROFNs is defined using t-norms, t-conorms,
    and the generator/pseudo-inverse functions of the underlying t-norm.
    It is typically defined as T(A -> B, B -> A).
    """

    def get_operation_name(self) -> str:
        """
        Returns the name of the operation.

        Returns:
            str: The string 'equivalence'.
        """
        return 'equivalence'

    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number types supported by this operation.

        Returns:
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary equivalence operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm, t-conorm, generator, and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        md = tnorm.t_norm(
            tnorm.t_conorm(strategy_1.nmd, strategy_2.md),
            tnorm.t_conorm(strategy_2.nmd, strategy_1.md))

        nmd = tnorm.t_conorm(
            tnorm.t_norm(strategy_1.md, strategy_2.nmd),
            tnorm.t_norm(strategy_2.md, strategy_1.nmd))

        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = T(S(nmd1, md2), S(nmd2, md1)), nmd = S(T(md1, nmd2), T(md2, nmd1))
        md_res = tnorm.t_norm(
            tnorm.t_conorm(nmds1, mds2),
            tnorm.t_conorm(nmds2, mds1))
        nmd_res = tnorm.t_conorm(
            tnorm.t_norm(mds1, nmds2),
            tnorm.t_norm(mds2, nmds1))

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNDifference(OperationMixin):
    """
    Implements the set difference (A - B) operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The difference A - B for QROFNs is typically defined as A AND (NOT B).
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary set difference operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance (set A).
            strategy_2 (Any): The second QROFN strategy instance (set B).
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm, t-conorm, generator, and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree is calculated using the t-norm (T-norm).
        md = tnorm.t_norm(strategy_1.md, strategy_2.nmd)
        # Non-membership degree is calculated using the t-conorm (S-norm).
        nmd = tnorm.t_conorm(strategy_1.nmd, strategy_2.md)

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = T(md1, nmd2), nmd = S(nmd1, md2)
        md_res = tnorm.t_norm(mds1, nmds2)
        nmd_res = tnorm.t_conorm(nmds1, mds2)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROFNSymmetricDifference(OperationMixin):
    """
    Implements the symmetric difference operation for Q-Rung Orthopair Fuzzy Numbers (QROFNs).

    The symmetric difference A XOR B for QROFNs is typically defined as
    (A AND (NOT B)) OR ((NOT A) AND B).
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
            List[str]: A list containing 'qrofn'.
        """
        return ['qrofn']

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the binary symmetric difference operation between two QROFN strategies.

        Args:
            strategy_1 (Any): The first QROFN strategy instance.
            strategy_2 (Any): The second QROFN strategy instance.
            tnorm (OperationTNorm): An instance of OperationTNorm to perform
                                    t-norm, t-conorm, generator, and pseudo-inverse calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the 'md', 'nmd', and 'q'
                            of the resulting QROFN.
        """
        # Membership degree calculation for symmetric difference.
        # It's S(md1/nmd2, nmd1/md2), T(md1/nmd2, nmd1/md2))

        md = tnorm.t_conorm(
            tnorm.t_norm(strategy_1.md, strategy_2.nmd),
            tnorm.t_norm(strategy_1.nmd, strategy_2.md))

        nmd = tnorm.t_norm(
            tnorm.t_conorm(strategy_1.md, strategy_2.nmd),
            tnorm.t_conorm(strategy_1.nmd, strategy_2.md)
        )

        # The q-rung of the result is the same as the input QROFNs.
        return {'md': md, 'nmd': nmd, 'q': strategy_1.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        mds1, nmds1, mds2, nmds2 = _prepare_operands(fuzzarray_1, other)

        # Formula: md = S(T(md1, nmd2), T(nmd1, md2)), nmd = T(S(md1, nmd2), S(nmd1, md2))
        md_res = tnorm.t_conorm(
            tnorm.t_norm(mds1, nmds2),
            tnorm.t_norm(nmds1, mds2))
        nmd_res = tnorm.t_norm(
            tnorm.t_conorm(mds1, nmds2),
            tnorm.t_conorm(nmds1, mds2))

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


# ---- Special Matrix Multiplication ----

@register_operation
class QROFNMatmul(OperationMixin):
    """
    Implements matrix multiplication (@) for Q-Rung Orthopair Fuzzy Arrays.
    """
    def get_operation_name(self) -> str:
        return 'matmul'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrofn']

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        if not isinstance(other, Fuzzarray):
            raise TypeError(f"Matrix multiplication requires two Fuzzarrays, "
                            f"but got '{type(fuzzarray_1)}' and '{type(other)}'")

        if fuzzarray_1.ndim != 2 or other.ndim != 2:
            raise ValueError(f"Matrix multiplication requires 2-D arrays, "
                             f"but got shapes '{fuzzarray_1.shape}' and '{other.shape}'")

        if fuzzarray_1.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch for matmul: '{fuzzarray_1.shape}' @ '{other.shape}'.")

        a_md, a_nmd = fuzzarray_1.backend.get_component_arrays()
        b_md, b_nmd = other.backend.get_component_arrays()

        # 模糊矩阵乘法的核心：
        # C_ij = V_k (A_ik * B_kj)
        # 其中 V 是 t-conorm, * 是 t-norm
        # 新的隶属度是 t-conorm(t-norm(a.md, b.md))
        # 新的非隶属度是 t-norm(t-conorm(a.nmd, b.nmd))

        # 使用 einsum 实现高效的张量点积
        # 'ik,kj->ij' 定义了标准的矩阵乘法
        # t-norm(a.md, b.md) -> a_md[:, :, np.newaxis] * b_md[np.newaxis, :, :]
        # t-conorm over k -> .sum(axis=1) for algebraic t-conorm
        # 这是一个简化的代数实现。
        # 更通用的实现将使用 tnorm.t_norm 和 tnorm.t_conorm

        # Step 1: 结果矩阵每个潜在元素的逐元素t-范数
        # 这将创建一个新的维度。 Shape: (ik, kj) -> (i, k, j)

        md_prod = tnorm.t_norm(a_md[:, :, np.newaxis], b_md[np.newaxis, :, :])
        nmd_sum = tnorm.t_conorm(a_nmd[:, :, np.newaxis], b_nmd[np.newaxis, :, :])

        # Step 2: 沿着公共维度“k”进行聚合
        # 对于 md，我们使用 t-余模进行聚合。对于 nmd，我们使用 t-模进行聚合。
        new_md = tnorm.t_conorm_reduce(md_prod, axis=1)     # type: ignore
        new_nmd = tnorm.t_norm_reduce(nmd_sum, axis=1)      # type: ignore

        # Ensure the results are proper float64 arrays
        new_md = np.asarray(new_md, dtype=np.float64)
        new_nmd = np.asarray(new_nmd, dtype=np.float64)

        backend_cls = get_fuzztype_backend('qrofn')
        new_backend = backend_cls.from_arrays(new_md, new_nmd, q=fuzzarray_1.q)
        return Fuzzarray(backend=new_backend)


# 兼容旧接口（现在无需再调用）
def register_qrofn_operations():
    import warnings
    warnings.warn("register_qrofn_operations 已不再需要（类已通过 @register_operation 自动注册）",
                  DeprecationWarning, stacklevel=2)
