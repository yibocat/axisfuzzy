#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 21:15
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines various operations for Q-Rung Orthopair Fuzzy Numbers (QROFNs).
It includes arithmetic, comparison, and set-theoretic operations, all implemented
as subclasses of `OperationMixin` and registered with the global operation registry.

Each operation class specifies:
- The operation name (e.g., 'add', 'mul', 'gt').
- The fuzzy number types it supports (currently 'qrofn').
- The core logic for executing the operation, often leveraging `OperationTNorm`
  for t-norm and t-conorm calculations.

Classes:
    QROFNAddition: Implements addition for QROFNs.
    QROFNSubtraction: Implements subtraction for QROFNs.
    QROFNMultiplication: Implements multiplication for QROFNs.
    QROFNDivision: Implements division for QROFNs.
    QROFNPower: Implements power operation for QROFNs.
    QROFNTimes: Implements scalar multiplication for QROFNs.
    QROFNExponential: Implements exponential operation for QROFNs.
    QROFNLogarithmic: Implements logarithmic operation for QROFNs.
    QROFNGreaterThan: Implements greater than comparison for QROFNs.
    QROFNLessThan: Implements less than comparison for QROFNs.
    QROFNEquals: Implements equality comparison for QROFNs.
    QROFNGreaterEquals: Implements greater than or equal to comparison for QROFNs.
    QROFNLessEquals: Implements less than or equal to comparison for QROFNs.
    QROFNNotEquals: Implements not equal to comparison for QROFNs.
    QROFNIntersection: Implements intersection (AND) for QROFNs.
    QROFNUnion: Implements union (OR) for QROFNs.
    QROFNComplement: Implements complement (NOT) for QROFNs.
    QROFNImplication: Implements implication for QROFNs.
    QROFNEquivalence: Implements equivalence for QROFNs.
    QROFNDifference: Implements set difference for QROFNs.
    QROFNSymmetricDifference: Implements symmetric difference for QROFNs.

Functions:
    register_qrofn_operations: Registers all QROFN operations with the global registry.
"""
import warnings
from typing import List, Any, Dict, Union

from fuzzlab.config import get_config
from fuzzlab.core.mixin import OperationMixin, get_operation_registry
from fuzzlab.core.triangular import OperationTNorm


# --- QROFN Arithmetic Operations ---

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
        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}


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
        # TODO: exp 计算目前还存在缺陷。此处写出来仅用于测试
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.f_inv_func(operand * tnorm.f_func(strategy.md))
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(operand * tnorm.g_func(strategy.nmd))

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}


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
        # TODO: exp 计算目前还存在缺陷。此处写出来仅用于测试
        # Membership degree calculation using the dual generator (f_func) and its pseudo-inverse (f_inv_func).
        md = tnorm.f_inv_func(tnorm.f_func(strategy.md) / operand)
        # Non-membership degree calculation using the generator (g_func) and its pseudo-inverse (g_inv_func).
        nmd = tnorm.g_inv_func(tnorm.g_func(strategy.nmd) / operand)

        # The q-rung of the result is the same as the input QROFN.
        return {'md': md, 'nmd': nmd, 'q': strategy.q}


# --- QROFN Comparison Operations ---

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
        # Comparison logic: md_1 > md_2 AND nmd_1 < nmd_2.
        return {'value': strategy_1.md > strategy_2.md and strategy_1.nmd < strategy_2.nmd}


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
        # Comparison logic: md_1 < md_2 AND nmd_1 > nmd_2.
        return {'value': strategy_1.md < strategy_2.md and strategy_1.nmd > strategy_2.nmd}


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
        # Equality logic: abs(md_1 - md_2) < epsilon AND abs(nmd_1 - nmd_2) < epsilon.
        value = (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON
                 and abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


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
        # Greater than or equal to logic: (md_1 > md_2 AND nmd_1 < nmd_2) OR (md_1 == md_2 AND nmd_1 == nmd_2).
        value = (strategy_1.md > strategy_2.md and strategy_1.nmd < strategy_2.nmd) or \
                (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
                 abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


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
        # Less than or equal to logic: (md_1 < md_2 AND nmd_1 > nmd_2) OR (md_1 == md_2 AND nmd_1 == nmd_2).
        value = (strategy_1.md < strategy_2.md and strategy_1.nmd > strategy_2.nmd) or \
                (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
                 abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


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
        value = not (abs(strategy_1.md - strategy_2.md) < config.DEFAULT_EPSILON and
                     abs(strategy_1.nmd - strategy_2.nmd) < config.DEFAULT_EPSILON)

        return {'value': value}


# --- QROFN Set Operations ---

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


def register_qrofn_operations():
    """
    Registers all QROFN-related operational methods to the global operation registry.

    This function should be called to make all QROFN arithmetic, comparison,
    and set operations available for use within the FuzzLab framework.
    It instantiates each `OperationMixin` subclass and registers it.
    """
    registry = get_operation_registry()
    registry.register(QROFNAddition())
    registry.register(QROFNSubtraction())
    registry.register(QROFNMultiplication())
    registry.register(QROFNDivision())
    registry.register(QROFNPower())
    registry.register(QROFNTimes())
    registry.register(QROFNExponential())
    registry.register(QROFNLogarithmic())
    registry.register(QROFNGreaterThan())
    registry.register(QROFNLessThan())
    registry.register(QROFNEquals())
    registry.register(QROFNGreaterEquals())
    registry.register(QROFNLessEquals())
    registry.register(QROFNNotEquals())
    registry.register(QROFNIntersection())
    registry.register(QROFNUnion())
    registry.register(QROFNComplement())
    registry.register(QROFNImplication())
    registry.register(QROFNEquivalence())
    registry.register(QROFNDifference())
    registry.register(QROFNSymmetricDifference())