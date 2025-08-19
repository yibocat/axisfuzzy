#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import warnings
from typing import Any, Callable, List, Dict, Optional, Union

import numpy as np

from ...config import get_config
from ...core import (
    Fuzznum, Fuzzarray,
    get_fuzztype_backend,
    OperationTNorm,
    OperationMixin,
    register_operation
)
from .utils import _pairwise_combinations

from ...utils import experimental


def _prepare_operands(
        operand1: Fuzzarray,
        operand2: Any) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Helper to get component arrays from operands."""
    mds1, nmds1 = operand1.backend.get_component_arrays()

    if isinstance(operand2, Fuzzarray):
        if operand2.mtype != operand1.mtype:
            raise ValueError(f"Cannot operate on Fuzzarrays with different mtypes: "
                             f"{operand1.mtype} and {operand2.mtype}")

        if operand2.q != operand1.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        mds2, nmds2 = operand2.backend.get_component_arrays()
        try:
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        except ValueError as e:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {operand1.shape} and {operand2.shape}")
    elif isinstance(operand2, Fuzznum):
        if operand2.mtype != operand1.mtype:
            raise TypeError(f"Cannot operate with different mtypes: "
                            f"Fuzzarray('{operand1.mtype}') and Fuzznum('{operand2.mtype}')")
        if operand2.q != operand1.q:
            raise ValueError(f"Cannot operate with different q values: "
                             f"Fuzzarray(q={operand1.q}) and Fuzznum(q={operand2.q})")

        mds2 = np.empty((1,), dtype=object)
        nmds2 = np.empty((1,), dtype=object)
        mds2[0] = operand2.md
        nmds2[0] = operand2.nmd
        try:
            return np.broadcast_arrays(mds1, nmds1, mds2, nmds2)
        except ValueError as e:
            raise ValueError(f"Shape mismatch: cannot broadcast shapes {operand1.shape} and (1,)") from e
    else:
        raise TypeError(f"Unsupported operand type for vectorized operation: {type(operand2)}")


# def _pairwise_combinations(
#         a: np.ndarray,
#         b: np.ndarray,
#         func: Callable) -> np.ndarray:
#     """Generate all pairwise combinations between two 1D arrays with a custom binary operation.
#
#     This function applies a given binary function to each combination of
#     elements from arrays `a` and `b`, and flattens the results into a 1D array.
#
#     Args:
#         a (np.ndarray): First 1D input array.
#         b (np.ndarray): Second 1D input array.
#         func (Callable): A binary function that takes two NumPy arrays `x` and `y`
#             of the same shape and returns an array of results.
#
#     Returns:
#         np.ndarray: A 1D array containing results of applying `func` to every pair (ai, bj).
#
#     Raises:
#         ValueError: If inputs are not 1D NumPy arrays.
#
#     Examples:
#         >>> a = np.array([1, 2, 3])
#         >>> b = np.array([10, 20])
#         >>> pairwise_combinations(a, b, lambda x, y: x + y)
#         array([11, 21, 12, 22, 13, 23])
#
#         >>> pairwise_combinations(a, b, lambda x, y: x * y)
#         array([10, 20, 20, 40, 30, 60])
#
#     Notes:
#         - Internally uses `np.meshgrid` for broadcasting all combinations.
#         - The resulting 2D matrix is flattened into one dimension.
#         - Order follows row-major (C-order) flattening, i.e. combinations are grouped by `a` first.
#
#     See Also:
#         np.add.outer, np.multiply.outer, np.fromfunction
#     """
#     if a is None or b is None:
#         raise ValueError("Inputs must not be None.")
#     if a.ndim != 1 or b.ndim != 1:
#         # This can happen if one of the elements in the object array is None
#         raise ValueError("Inputs must be 1D NumPy arrays.")
#
#     A, B = np.meshgrid(a, b, indexing="ij")
#     return func(A, B).ravel()


# --- QROHFN Arithmetic Operations ---


@register_operation
class QROHFNAddition(OperationMixin):
    """
    Addition operation for q-rung orthopair hesitant fuzzy numbers (QROHFN).
    """

    def get_operation_name(self) -> str:
        return 'add'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes addition for two single QROHFN strategies using pairwise combinations.
        """
        md_res = _pairwise_combinations(operand1.md, operand2.md, tnorm.t_conorm)
        nmd_res = _pairwise_combinations(operand1.nmd, operand2.nmd, tnorm.t_norm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized addition for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        # Create ufuncs by wrapping the pairwise combination logic.
        # The ufunc will iterate over the object arrays at C-speed and apply
        # this logic to each pair of hesitant sets.
        add_md_ufunc = np.frompyfunc(
            lambda md1, md2: _pairwise_combinations(md1, md2, tnorm.t_conorm),
            2, 1)

        add_nmd_ufunc = np.frompyfunc(
            lambda nmd1, nmd2: _pairwise_combinations(nmd1, nmd2, tnorm.t_norm),
            2, 1)

        # Apply the ufuncs to the object arrays.
        md_res = add_md_ufunc(mds1, mds2)
        nmd_res = add_nmd_ufunc(nmds1, nmds2)

        # Create a new Fuzzarray from the resulting object arrays.
        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNSubtraction(OperationMixin):
    """
    Implements the subtraction operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).
    This implementation is vectorized to avoid slow Python loops.
    """

    def get_operation_name(self) -> str:
        return 'sub'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes subtraction for two single QROHFN strategies using vectorized combinations.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The subtraction operation of 'qrohfn' is currently only applicable to 'algebraic' t-norm. "
                          f"It will be performed using the 'algebraic' t-norm logic.")

        q = operand1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Reshape 1D hesitant sets into 4D arrays for broadcasting
        md1_b = operand1.md[:, np.newaxis, np.newaxis, np.newaxis]
        nmd1_b = operand1.nmd[np.newaxis, :, np.newaxis, np.newaxis]
        md2_b = operand2.md[np.newaxis, np.newaxis, :, np.newaxis]
        nmd2_b = operand2.nmd[np.newaxis, np.newaxis, np.newaxis, :]

        # --- Vectorized Calculation of Conditions and Results ---
        # Safely handle boundary cases to avoid division by zero or invalid roots
        md2_b_safe = np.where(md2_b == 1.0, 1.0 - epsilon, md2_b)
        nmd2_b_safe = np.where(nmd2_b == 0, epsilon, nmd2_b)

        # Calculate conditions in the 4D broadcasted space
        condition_2 = ((1 - md1_b ** q) / (1 - md2_b_safe ** q)) ** (1 / q)
        condition_1 = nmd1_b / nmd2_b_safe

        # Create a boolean mask for valid results
        valid_mask = ((condition_1 >= epsilon) & (condition_1 <= 1 - epsilon) &
                      (condition_2 >= epsilon) & (condition_2 <= 1 - epsilon) &
                      (condition_1 <= condition_2))

        # Calculate potential results
        md_results_calc = ((md1_b ** q - md2_b_safe ** q) / (1 - md2_b_safe ** q)) ** (1 / q)
        nmd_results_calc = nmd1_b / nmd2_b_safe

        # Apply mask to get final results, with defaults for invalid pairs
        final_md = np.where(valid_mask, md_results_calc, 0.0)
        final_nmd = np.where(valid_mask, nmd_results_calc, 1.0)

        return {
            'md': final_md.ravel(),
            'nmd': final_nmd.ravel(),
            'q': q
        }

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized subtraction for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        def _sub_func(md1, nmd1, md2, nmd2):
            # Helper to call the efficient binary implementation for each element pair
            if any(x is None for x in [md1, nmd1, md2, nmd2]):
                return None, None
            op1_strat = type('strategy', (), {'md': md1, 'nmd': nmd1, 'q': operand1.q})()
            op2_strat = type('strategy', (), {'md': md2, 'nmd': nmd2, 'q': operand1.q})()
            result_dict = self._execute_binary_op_impl(op1_strat, op2_strat, tnorm)
            return result_dict['md'], result_dict['nmd']

        # Create a ufunc that takes 4 object arrays and returns 2 object arrays
        sub_ufunc = np.frompyfunc(_sub_func, 4, 2)
        md_res, nmd_res = sub_ufunc(mds1, nmds1, mds2, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNMultiplication(OperationMixin):
    """
    Implements the multiplication operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The multiplication of two QROHFNs uses t-norm for membership degrees and
    t-conorm for non-membership degrees, applied to all pairwise combinations.
    """

    def get_operation_name(self) -> str:
        return 'mul'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes multiplication for two single QROHFN strategies using pairwise combinations.
        """
        md_res = _pairwise_combinations(operand1.md, operand2.md, tnorm.t_norm)
        nmd_res = _pairwise_combinations(operand1.nmd, operand2.nmd, tnorm.t_conorm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized multiplication for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        mul_md_ufunc = np.frompyfunc(
            lambda md1, md2: _pairwise_combinations(md1, md2, tnorm.t_norm),
            2, 1
        )
        mul_nmd_ufunc = np.frompyfunc(
            lambda nmd1, nmd2: _pairwise_combinations(nmd1, nmd2, tnorm.t_conorm),
            2, 1
        )

        md_res = mul_md_ufunc(mds1, mds2)
        nmd_res = mul_nmd_ufunc(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNDivision(OperationMixin):
    """
    Implements the division operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).
    This implementation is vectorized to avoid slow Python loops.
    """

    def get_operation_name(self) -> str:
        return 'div'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes division for two single QROHFN strategies using vectorized combinations.
        """
        if tnorm.norm_type != 'algebraic':
            warnings.warn(f"The division operation of 'qrohfn' is currently only applicable to 'algebraic' t-norm. "
                          f"It will be performed using the 'algebraic' t-norm logic.")

        q = operand1.q
        epsilon = get_config().DEFAULT_EPSILON

        # Reshape 1D hesitant sets into 4D arrays for broadcasting
        md1_b = operand1.md[:, np.newaxis, np.newaxis, np.newaxis]
        nmd1_b = operand1.nmd[np.newaxis, :, np.newaxis, np.newaxis]
        md2_b = operand2.md[np.newaxis, np.newaxis, :, np.newaxis]
        nmd2_b = operand2.nmd[np.newaxis, np.newaxis, np.newaxis, :]

        # --- Vectorized Calculation of Conditions and Results ---
        md2_b_safe = np.where(md2_b == 0, epsilon, md2_b)
        nmd2_b_safe = np.where(nmd2_b == 1.0, 1.0 - epsilon, nmd2_b)

        condition_1 = md1_b / md2_b_safe
        condition_2 = ((1 - nmd1_b ** q) / (1 - nmd2_b_safe ** q)) ** (1 / q)

        valid_mask = ((condition_1 >= epsilon) & (condition_1 <= 1 - epsilon) &
                      (condition_2 >= epsilon) & (condition_2 <= 1 - epsilon) &
                      (condition_1 <= condition_2))

        md_results_calc = md1_b / md2_b_safe
        nmd_results_calc = ((nmd1_b ** q - nmd2_b_safe ** q) / (1 - nmd2_b_safe ** q)) ** (1 / q)

        final_md = np.where(valid_mask, md_results_calc, 1.0)
        final_nmd = np.where(valid_mask, nmd_results_calc, 0.0)

        return {
            'md': final_md.ravel(),
            'nmd': final_nmd.ravel(),
            'q': q
        }

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized division for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        def _div_func(md1, nmd1, md2, nmd2):
            if any(x is None for x in [md1, nmd1, md2, nmd2]):
                return None, None
            op1_strat = type('strategy', (), {'md': md1, 'nmd': nmd1, 'q': operand1.q})()
            op2_strat = type('strategy', (), {'md': md2, 'nmd': nmd2, 'q': operand1.q})()
            result_dict = self._execute_binary_op_impl(op1_strat, op2_strat, tnorm)
            return result_dict['md'], result_dict['nmd']

        div_ufunc = np.frompyfunc(_div_func, 4, 2)
        md_res, nmd_res = div_ufunc(mds1, nmds1, mds2, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNPower(OperationMixin):
    """
    Implements the power operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    This operation calculates A^operand, where A is a QROHFN and operand is a scalar.
    It applies the power operation to each element in the hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'pow'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary power operation on a QROHFN strategy with a scalar operand.
        """
        # Apply power operation to each element in the hesitant sets
        md_res = tnorm.g_inv_func(operand * tnorm.g_func(strategy.md))
        nmd_res = tnorm.f_inv_func(operand * tnorm.f_func(strategy.nmd))

        return {'md': md_res, 'nmd': nmd_res, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized power operation for QROHFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        def _power_func(hesitant_set, is_md=True):
            if hesitant_set is None:
                return None
            if is_md:
                return tnorm.g_inv_func(operand * tnorm.g_func(hesitant_set))
            else:
                return tnorm.f_inv_func(operand * tnorm.f_func(hesitant_set))

        pow_md_ufunc = np.frompyfunc(lambda x: _power_func(x, True), 1, 1)
        pow_nmd_ufunc = np.frompyfunc(lambda x: _power_func(x, False), 1, 1)

        md_res = pow_md_ufunc(mds)
        nmd_res = pow_nmd_ufunc(nmds)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNTimes(OperationMixin):
    """
    Implements scalar multiplication for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    This operation calculates operand * A, where A is a QROHFN and operand is a scalar.
    It applies the scalar multiplication to each element in the hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'tim'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes the unary scalar multiplication operation on a QROHFN strategy with a scalar operand.
        """
        # Apply scalar multiplication to each element in the hesitant sets
        md_res = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        nmd_res = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        return {'md': md_res, 'nmd': nmd_res, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   operand: Union[int, float],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized scalar multiplication for QROHFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        def _times_func(hesitant_set, is_md=True):
            if hesitant_set is None:
                return None
            if is_md:
                return tnorm.f_inv_func(operand * tnorm.f_func(hesitant_set))
            else:
                return tnorm.g_inv_func(operand * tnorm.g_func(hesitant_set))

        times_md_ufunc = np.frompyfunc(lambda x: _times_func(x, True), 1, 1)
        times_nmd_ufunc = np.frompyfunc(lambda x: _times_func(x, False), 1, 1)

        md_res = times_md_ufunc(mds)
        nmd_res = times_nmd_ufunc(nmds)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class QROHFNExponential(OperationMixin):
    """
    Implements the exponential operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    This operation calculates exp(A), where A is a QROHFN.
    It applies the exponential operation to each element in the hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'exp'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
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

        def _times_func(hesitant_set, is_md=True):
            if hesitant_set is None:
                return None
            if is_md:
                return tnorm.f_inv_func(operand * tnorm.f_func(hesitant_set))
            else:
                return tnorm.g_inv_func(operand * tnorm.g_func(hesitant_set))

        times_md_ufunc = np.frompyfunc(lambda x: _times_func(x, True), 1, 1)
        times_nmd_ufunc = np.frompyfunc(lambda x: _times_func(x, False), 1, 1)

        md_res = times_md_ufunc(mds)
        nmd_res = times_nmd_ufunc(nmds)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# TODO: 该运算还未实现, 暂时处在测试阶段. 仅用来测试.
@register_operation
class QROHFNLogarithmic(OperationMixin):
    """
    Implements the exponential operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    This operation calculates exp(A), where A is a QROHFN.
    It applies the exponential operation to each element in the hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'log'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    @experimental
    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
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

        def _times_func(hesitant_set, is_md=True):
            if hesitant_set is None:
                return None
            if is_md:
                return tnorm.f_inv_func(operand * tnorm.f_func(hesitant_set))
            else:
                return tnorm.g_inv_func(operand * tnorm.g_func(hesitant_set))

        times_md_ufunc = np.frompyfunc(lambda x: _times_func(x, True), 1, 1)
        times_nmd_ufunc = np.frompyfunc(lambda x: _times_func(x, False), 1, 1)

        md_res = times_md_ufunc(mds)
        nmd_res = times_nmd_ufunc(nmds)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


# --- QROHFN Comparison Operations ---


@register_operation
class QROHFNGreaterThan(OperationMixin):
    """
    Implements the greater than comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'gt'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes greater than comparison for two single QROHFN strategies.
        """

        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Score function: S(A) = md_mean^q - nmd_mean^q
        q = operand1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q

        return score1 > score2

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized greater than comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        q = operand1.q

        def _gt_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return False

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            score1 = md1_mean ** q - nmd1_mean ** q
            score2 = md2_mean ** q - nmd2_mean ** q

            return score1 > score2

        gt_ufunc = np.frompyfunc(_gt_func, 4, 1)
        return gt_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


@register_operation
class QROHFNLessThan(OperationMixin):
    """
    Implements the less than comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'lt'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes less than comparison for two single QROHFN strategies.
        """
        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Score function: S(A) = md_mean^q - nmd_mean^q
        q = operand1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q

        return score1 < score2

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized less than comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        q = operand1.q

        def _lt_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return False

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            score1 = md1_mean ** q - nmd1_mean ** q
            score2 = md2_mean ** q - nmd2_mean ** q

            return score1 < score2

        lt_ufunc = np.frompyfunc(_lt_func, 4, 1)
        return lt_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


@register_operation
class QROHFNEquals(OperationMixin):
    """
    Implements the equality comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'eq'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes equality comparison for two single QROHFN strategies.
        """
        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        epsilon = get_config().DEFAULT_EPSILON

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Check if means are approximately equal
        md_equal = abs(md1_mean - md2_mean) < epsilon
        nmd_equal = abs(nmd1_mean - nmd2_mean) < epsilon

        return md_equal and nmd_equal

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized equality comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        epsilon = get_config().DEFAULT_EPSILON

        def _eq_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return False

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            md_equal = abs(md1_mean - md2_mean) < epsilon
            nmd_equal = abs(nmd1_mean - nmd2_mean) < epsilon

            return md_equal and nmd_equal

        eq_ufunc = np.frompyfunc(_eq_func, 4, 1)
        return eq_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


@register_operation
class QROHFNGreaterEquals(OperationMixin):
    """
    Implements the greater than or equal comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'ge'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes greater than or equal comparison for two single QROHFN strategies.
        """
        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Score function: S(A) = md_mean^q - nmd_mean^q
        q = operand1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q

        return score1 >= score2

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized greater than or equal comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        q = operand1.q

        def _ge_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return False

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            score1 = md1_mean ** q - nmd1_mean ** q
            score2 = md2_mean ** q - nmd2_mean ** q

            return score1 >= score2

        ge_ufunc = np.frompyfunc(_ge_func, 4, 1)
        return ge_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


@register_operation
class QROHFNLessEquals(OperationMixin):
    """
    Implements the less than or equal comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'le'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes less than or equal comparison for two single QROHFN strategies.
        """
        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Score function: S(A) = md_mean^q - nmd_mean^q
        q = operand1.q
        score1 = md1_mean ** q - nmd1_mean ** q
        score2 = md2_mean ** q - nmd2_mean ** q

        return score1 <= score2

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized less than or equal comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        q = operand1.q

        def _le_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return False

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            score1 = md1_mean ** q - nmd1_mean ** q
            score2 = md2_mean ** q - nmd2_mean ** q

            return score1 <= score2

        le_ufunc = np.frompyfunc(_le_func, 4, 1)
        return le_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


@register_operation
class QROHFNNotEquals(OperationMixin):
    """
    Implements the not equal comparison for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    Comparison is based on the mean values of membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'ne'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> bool:
        """
        Executes not equal comparison for two single QROHFN strategies.
        """
        if operand1.q != operand2.q:
            raise ValueError(f"Cannot operate on Fuzzarrays with different q values: "
                             f"{operand1.q} and {operand2.q}")

        epsilon = get_config().DEFAULT_EPSILON

        # Calculate mean values for comparison
        md1_mean = np.mean(operand1.md)
        nmd1_mean = np.mean(operand1.nmd)
        md2_mean = np.mean(operand2.md)
        nmd2_mean = np.mean(operand2.nmd)

        # Check if means are not approximately equal
        md_not_equal = abs(md1_mean - md2_mean) >= epsilon
        nmd_not_equal = abs(nmd1_mean - nmd2_mean) >= epsilon

        return md_not_equal or nmd_not_equal

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> np.ndarray:
        """
        Vectorized not equal comparison for QROHFN fuzzy arrays.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)
        epsilon = get_config().DEFAULT_EPSILON

        def _ne_func(md1_set, nmd1_set, md2_set, nmd2_set):
            if md1_set is None or nmd1_set is None or md2_set is None or nmd2_set is None:
                return True

            md1_mean = np.mean(md1_set)
            nmd1_mean = np.mean(nmd1_set)
            md2_mean = np.mean(md2_set)
            nmd2_mean = np.mean(nmd2_set)

            md_not_equal = abs(md1_mean - md2_mean) >= epsilon
            nmd_not_equal = abs(nmd1_mean - nmd2_mean) >= epsilon

            return md_not_equal or nmd_not_equal

        ne_ufunc = np.frompyfunc(_ne_func, 4, 1)
        return ne_ufunc(mds1, nmds1, mds2, nmds2).astype(bool)


# --- QROFN Set Operations ---


@register_operation
class QROHFNIntersection(OperationMixin):
    """
    Implements the intersection (AND) operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The intersection of two QROHFNs uses t-norm for membership degrees and
    t-conorm for non-membership degrees, applied to all pairwise combinations.
    """

    def get_operation_name(self) -> str:
        return 'intersection'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes intersection for two single QROHFN strategies using pairwise combinations.
        """
        md_res = _pairwise_combinations(operand1.md, operand2.md, tnorm.t_norm)
        nmd_res = _pairwise_combinations(operand1.nmd, operand2.nmd, tnorm.t_conorm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized intersection for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        intersection_md_ufunc = np.frompyfunc(
            lambda md1, md2: _pairwise_combinations(md1, md2, tnorm.t_norm),
            2, 1
        )
        intersection_nmd_ufunc = np.frompyfunc(
            lambda nmd1, nmd2: _pairwise_combinations(nmd1, nmd2, tnorm.t_conorm),
            2, 1
        )

        md_res = intersection_md_ufunc(mds1, mds2)
        nmd_res = intersection_nmd_ufunc(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNUnion(OperationMixin):
    """
    Implements the union (OR) operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The union of two QROHFNs uses t-conorm for membership degrees and
    t-norm for non-membership degrees, applied to all pairwise combinations.
    """

    def get_operation_name(self) -> str:
        return 'union'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes union for two single QROHFN strategies using pairwise combinations.
        """
        md_res = _pairwise_combinations(operand1.md, operand2.md, tnorm.t_conorm)
        nmd_res = _pairwise_combinations(operand1.nmd, operand2.nmd, tnorm.t_norm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized union for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        union_md_ufunc = np.frompyfunc(
            lambda md1, md2: _pairwise_combinations(md1, md2, tnorm.t_conorm),
            2, 1
        )
        union_nmd_ufunc = np.frompyfunc(
            lambda nmd1, nmd2: _pairwise_combinations(nmd1, nmd2, tnorm.t_norm),
            2, 1
        )

        md_res = union_md_ufunc(mds1, mds2)
        nmd_res = union_nmd_ufunc(nmds1, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNComplement(OperationMixin):
    """
    Implements the complement (NOT) operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The complement of a QROHFN simply swaps the membership and non-membership hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'complement'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes complement for a single QROHFN strategy.
        """
        # Complement operation swaps membership and non-membership hesitant sets
        md_res = strategy.nmd.copy()
        nmd_res = strategy.md.copy()

        return {'md': md_res, 'nmd': nmd_res, 'q': strategy.q}

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray: Fuzzarray,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Vectorized complement for QROHFN fuzzy arrays.
        """
        mds, nmds = fuzzarray.backend.get_component_arrays()

        # Simply swap the arrays - each element's hesitant sets are swapped
        md_res = nmds.copy()
        nmd_res = mds.copy()

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=fuzzarray.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNImplication(OperationMixin):
    """
    Implements the implication operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The implication A -> B for QROHFNs is defined using pairwise combinations
    with specific t-norm and t-conorm operations.
    """

    def get_operation_name(self) -> str:
        return 'implication'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes implication for two single QROHFN strategies using pairwise combinations.
        """
        # Implication: A -> B = ~A ∨ B = S(nmd_A, md_B), T(md_A, nmd_B)
        md_res = _pairwise_combinations(operand1.nmd, operand2.md, tnorm.t_conorm)
        nmd_res = _pairwise_combinations(operand1.md, operand2.nmd, tnorm.t_norm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized implication for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        implication_md_ufunc = np.frompyfunc(
            lambda nmd1, md2: _pairwise_combinations(nmd1, md2, tnorm.t_conorm),
            2, 1
        )
        implication_nmd_ufunc = np.frompyfunc(
            lambda md1, nmd2: _pairwise_combinations(md1, nmd2, tnorm.t_norm),
            2, 1
        )

        md_res = implication_md_ufunc(nmds1, mds2)
        nmd_res = implication_nmd_ufunc(mds1, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNEquivalence(OperationMixin):
    """
    Implements the equivalence operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The equivalence A <-> B for QROHFNs is defined as T(A -> B, B -> A),
    using pairwise combinations for hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'equivalence'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes equivalence for two single QROHFN strategies using pairwise combinations.
        """
        # Equivalence: A <-> B = T(A -> B, B -> A)
        # A -> B: S(nmd_A, md_B), T(md_A, nmd_B)
        # B -> A: S(nmd_B, md_A), T(md_B, nmd_A)

        # Calculate A -> B
        impl_ab_md = _pairwise_combinations(operand1.nmd, operand2.md, tnorm.t_conorm)
        impl_ab_nmd = _pairwise_combinations(operand1.md, operand2.nmd, tnorm.t_norm)

        # Calculate B -> A
        impl_ba_md = _pairwise_combinations(operand2.nmd, operand1.md, tnorm.t_conorm)
        impl_ba_nmd = _pairwise_combinations(operand2.md, operand1.nmd, tnorm.t_norm)

        # T(A -> B, B -> A)
        md_res = _pairwise_combinations(impl_ab_md, impl_ba_md, tnorm.t_norm)
        nmd_res = _pairwise_combinations(impl_ab_nmd, impl_ba_nmd, tnorm.t_conorm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized equivalence for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        def _equivalence_md_func(nmd1, md1, nmd2, md2):
            if any(x is None for x in [nmd1, md1, nmd2, md2]):
                return None
            # A -> B: S(nmd1, md2)
            impl_ab = _pairwise_combinations(nmd1, md2, tnorm.t_conorm)
            # B -> A: S(nmd2, md1)
            impl_ba = _pairwise_combinations(nmd2, md1, tnorm.t_conorm)
            # T(A -> B, B -> A)
            return _pairwise_combinations(impl_ab, impl_ba, tnorm.t_norm)

        def _equivalence_nmd_func(md1, nmd1, md2, nmd2):
            if any(x is None for x in [md1, nmd1, md2, nmd2]):
                return None
            # A -> B: T(md1, nmd2)
            impl_ab = _pairwise_combinations(md1, nmd2, tnorm.t_norm)
            # B -> A: T(md2, nmd1)
            impl_ba = _pairwise_combinations(md2, nmd1, tnorm.t_norm)
            # S(A -> B, B -> A)
            return _pairwise_combinations(impl_ab, impl_ba, tnorm.t_conorm)

        equivalence_md_ufunc = np.frompyfunc(_equivalence_md_func, 4, 1)
        equivalence_nmd_ufunc = np.frompyfunc(_equivalence_nmd_func, 4, 1)

        md_res = equivalence_md_ufunc(nmds1, mds1, nmds2, mds2)
        nmd_res = equivalence_nmd_ufunc(mds1, nmds1, mds2, nmds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNDifference(OperationMixin):
    """
    Implements the set difference (A - B) operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The difference A - B for QROHFNs is defined as A ∩ ~B,
    using pairwise combinations for hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'difference'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes difference for two single QROHFN strategies using pairwise combinations.
        """
        # Difference: A - B = A ∩ ~B = T(md_A, nmd_B), S(nmd_A, md_B)
        md_res = _pairwise_combinations(operand1.md, operand2.nmd, tnorm.t_norm)
        nmd_res = _pairwise_combinations(operand1.nmd, operand2.md, tnorm.t_conorm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized difference for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        difference_md_ufunc = np.frompyfunc(
            lambda md1, nmd2: _pairwise_combinations(md1, nmd2, tnorm.t_norm),
            2, 1
        )
        difference_nmd_ufunc = np.frompyfunc(
            lambda nmd1, md2: _pairwise_combinations(nmd1, md2, tnorm.t_conorm),
            2, 1
        )

        md_res = difference_md_ufunc(mds1, nmds2)
        nmd_res = difference_nmd_ufunc(nmds1, mds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


@register_operation
class QROHFNSymmetricDifference(OperationMixin):
    """
    Implements the symmetric difference operation for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    The symmetric difference A ⊕ B for QROHFNs is defined as (A - B) ∪ (B - A),
    using pairwise combinations for hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'symdiff'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_binary_op_impl(self,
                                operand1: Any,
                                operand2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes symmetric difference for two single QROHFN strategies using pairwise combinations.
        """
        # Symmetric difference: A ⊕ B = (A - B) ∪ (B - A)
        # = S(T(md_A, nmd_B), T(md_B, nmd_A)), T(S(nmd_A, md_B), S(nmd_B, md_A))

        # Calculate A - B
        diff_ab_md = _pairwise_combinations(operand1.md, operand2.nmd, tnorm.t_norm)
        diff_ab_nmd = _pairwise_combinations(operand1.nmd, operand2.md, tnorm.t_conorm)

        # Calculate B - A
        diff_ba_md = _pairwise_combinations(operand2.md, operand1.nmd, tnorm.t_norm)
        diff_ba_nmd = _pairwise_combinations(operand2.nmd, operand1.md, tnorm.t_conorm)

        # (A - B) ∪ (B - A)
        md_res = _pairwise_combinations(diff_ab_md, diff_ba_md, tnorm.t_conorm)
        nmd_res = _pairwise_combinations(diff_ab_nmd, diff_ba_nmd, tnorm.t_norm)

        return {'md': md_res, 'nmd': nmd_res, 'q': operand1.q}

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Vectorized symmetric difference for QROHFN fuzzy arrays using np.frompyfunc.
        """
        mds1, nmds1, mds2, nmds2 = _prepare_operands(operand1, other)

        def _symdiff_md_func(md1, nmd1, md2, nmd2):
            if any(x is None for x in [md1, nmd1, md2, nmd2]):
                return None
            # A - B: T(md1, nmd2)
            diff_ab = _pairwise_combinations(md1, nmd2, tnorm.t_norm)
            # B - A: T(md2, nmd1)
            diff_ba = _pairwise_combinations(md2, nmd1, tnorm.t_norm)
            # S(A - B, B - A)
            return _pairwise_combinations(diff_ab, diff_ba, tnorm.t_conorm)

        def _symdiff_nmd_func(nmd1, md1, nmd2, md2):
            if any(x is None for x in [nmd1, md1, nmd2, md2]):
                return None
            # A - B: S(nmd1, md2)
            diff_ab = _pairwise_combinations(nmd1, md2, tnorm.t_conorm)
            # B - A: S(nmd2, md1)
            diff_ba = _pairwise_combinations(nmd2, md1, tnorm.t_conorm)
            # T(A - B, B - A)
            return _pairwise_combinations(diff_ab, diff_ba, tnorm.t_norm)

        symdiff_md_ufunc = np.frompyfunc(_symdiff_md_func, 4, 1)
        symdiff_nmd_ufunc = np.frompyfunc(_symdiff_nmd_func, 4, 1)

        md_res = symdiff_md_ufunc(mds1, nmds1, mds2, nmds2)
        nmd_res = symdiff_nmd_ufunc(nmds1, mds1, nmds2, mds2)

        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(md_res, nmd_res, q=operand1.q)
        return Fuzzarray(backend=new_backend)


# ---- Special Matrix Multiplication ----


@register_operation
class QROHFNMatrixMultiplication(OperationMixin):
    """
    Implements matrix multiplication for Q-Rung Orthopair Hesitant Fuzzy Numbers (QROHFNs).

    This operation performs standard matrix multiplication where each element multiplication
    uses the QROHFN multiplication logic with pairwise combinations of hesitant sets.
    """

    def get_operation_name(self) -> str:
        return 'matmul'

    def get_supported_mtypes(self) -> List[str]:
        return ['qrohfn']

    def _execute_fuzzarray_op_impl(self,
                                   operand1: Fuzzarray,
                                   other: Fuzzarray,
                                   tnorm: OperationTNorm) -> Fuzzarray:
        """
        Executes matrix multiplication for QROHFN fuzzy arrays.

        Args:
            operand1: Left matrix (shape: M x K)
            other: Right matrix (shape: K x N)
            tnorm: T-norm operations for fuzzy logic

        Returns:
            Result matrix (shape: M x N)
        """
        if not isinstance(other, Fuzzarray):
            raise TypeError("Matrix multiplication requires two Fuzzarray operands")

        if operand1.mtype != other.mtype:
            raise ValueError(f"Cannot multiply Fuzzarrays with different mtypes: "
                             f"{operand1.mtype} and {other.mtype}")

        if operand1.q != other.q:
            raise ValueError(f"Cannot multiply Fuzzarrays with different q values: "
                             f"{operand1.q} and {other.q}")

        # Validate matrix dimensions
        if operand1.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays")

        M, K1 = operand1.shape
        K2, N = other.shape

        if K1 != K2:
            raise ValueError(f"Cannot multiply matrices with incompatible shapes: "
                             f"({M}, {K1}) @ ({K2}, {N})")

        # Get component arrays
        mds1, nmds1 = operand1.backend.get_component_arrays()
        mds2, nmds2 = other.backend.get_component_arrays()

        # Initialize result arrays
        result_shape = (M, N)
        result_mds = np.empty(result_shape, dtype=object)
        result_nmds = np.empty(result_shape, dtype=object)

        # Perform matrix multiplication
        for i in range(M):
            for j in range(N):
                # Initialize accumulators for position (i, j)
                accumulated_md = None
                accumulated_nmd = None

                # Compute dot product for position (i, j)
                for k in range(K1):
                    # Get hesitant sets for multiplication
                    md1_k = mds1[i, k]
                    nmd1_k = nmds1[i, k]
                    md2_k = mds2[k, j]
                    nmd2_k = nmds2[k, j]

                    # Skip if any element is None
                    if any(x is None for x in [md1_k, nmd1_k, md2_k, nmd2_k]):
                        continue

                    # Perform QROHFN multiplication using pairwise combinations
                    product_md = _pairwise_combinations(md1_k, md2_k, tnorm.t_norm)
                    product_nmd = _pairwise_combinations(nmd1_k, nmd2_k, tnorm.t_conorm)

                    # Accumulate results using QROHFN addition (t-conorm for md, t-norm for nmd)
                    if accumulated_md is None:
                        accumulated_md = product_md
                        accumulated_nmd = product_nmd
                    else:
                        accumulated_md = _pairwise_combinations(accumulated_md, product_md, tnorm.t_conorm)
                        accumulated_nmd = _pairwise_combinations(accumulated_nmd, product_nmd, tnorm.t_norm)

                # Store results (handle case where all elements were None)
                if accumulated_md is None:
                    result_mds[i, j] = np.array([0.0])
                    result_nmds[i, j] = np.array([1.0])
                else:
                    result_mds[i, j] = accumulated_md
                    result_nmds[i, j] = accumulated_nmd

        # Create result Fuzzarray
        backend_cls = get_fuzztype_backend('qrohfn')
        new_backend = backend_cls.from_arrays(result_mds, result_nmds, q=operand1.q)
        return Fuzzarray(backend=new_backend)
