#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 19:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import warnings
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union, Optional

from fuzzlab.config import get_config
from fuzzlab.core.triangular import OperationTNorm


class OperationMixin(ABC):

    @abstractmethod
    def get_operation_name(self) -> str:
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        pass

    def supports(self, mtype: str) -> bool:
        return mtype in self.get_supported_mtypes()

    # ======================= Preprocessing before calculation ============================

    @staticmethod
    def _preprocess_binary_operands(strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> None:
        """
        Data preprocessing and validation before binary operations

        Args:
            strategy_1: First strategy instance
            strategy_2: Second strategy instance
            tnorm: T-norm instance

        Raises:
            ValueError: When data validation fails
            TypeError: When the types do not match
        """
        if strategy_1 is None or strategy_2 is None:
            raise ValueError('strategy_1 and strategy_2 cannot be None')

        if hasattr(strategy_1, 'mtype') and hasattr(strategy_2, 'mtype'):
            if strategy_1.mtype != strategy_2.mtype:
                raise TypeError(
                    f"mtype mismatch: strategy_1.mtype='{strategy_1.mtype}' != "
                    f"strategy_2.mtype='{strategy_2.mtype}'"
                )

        if hasattr(strategy_1, 'q') and hasattr(strategy_2, 'q'):
            if strategy_1.q != strategy_2.q:
                raise TypeError(
                    f"Q-rung mismatch: strategy_1.q={strategy_1.q} != strategy_2.q={strategy_2.q}"
                )

        if tnorm is None:
            raise ValueError("T-norm instance cannot be None")

    @staticmethod
    def _preprocess_unary_operand(strategy: Any,
                                  operand: Union[int, float],
                                  tnorm: OperationTNorm) -> None:
        """
        Data preprocessing and validation before unary operations (with operands)

        Args:
            strategy: Strategy Instance
            operand: Operand
            tnorm: T-norm instance

        Raises:
            ValueError: When data validation fails
            TypeError: When the types do not match
        """
        if strategy is None:
            raise ValueError('strategy cannot be None')

        if not isinstance(operand, (int, float)):
            raise TypeError(f"Operand must be int or float, got '{type(operand)}'.")

        if tnorm is None:
            raise ValueError("T-norm instance cannot be None")

    @staticmethod
    def _preprocess_pure_unary(strategy: Any,
                               tnorm: OperationTNorm) -> None:
        """
        Data preprocessing and validation before unary operations

        Args:
            strategy: Strategy Instance
            tnorm: T-norm instance

        Raises:
            ValueError: When data validation fails
        """
        if strategy is None:
            raise ValueError("Strategy instance cannot be None")

        if tnorm is None:
            raise ValueError("T-norm instance cannot be None")

    @staticmethod
    def _preprocess_comparison(strategy_1: Any,
                               strategy_2: Any,
                               tnorm: OperationTNorm) -> None:
        """
        Data preprocessing and validation before comparison operations

        Args:
            strategy_1: First strategy instance
            strategy_2: Second strategy instance
            tnorm: T-Norm instance

        Raises:
            ValueError: When data validation fails
        """
        if strategy_1 is None or strategy_2 is None:
            raise ValueError('strategy_1 and strategy_2 cannot be None')

        if hasattr(strategy_1, 'mtype') and hasattr(strategy_2, 'mtype'):
            if strategy_1.mtype != strategy_2.mtype:
                raise TypeError(
                    f"mtype mismatch: strategy_1.mtype='{strategy_1.mtype}' != "
                    f"strategy_2.mtype='{strategy_2.mtype}'"
                )

        if hasattr(strategy_1, 'q') and hasattr(strategy_2, 'q'):
            if strategy_1.q != strategy_2.q:
                raise TypeError(
                    f"Q-rung mismatch: strategy_1.q={strategy_1.q} != strategy_2.q={strategy_2.q}"
                )

        if tnorm is None:
            raise ValueError("T-norm instance cannot be None")

    # ======================= Post-processing of calculation results =======================

    def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-processing of the calculation results

        Args:
            result: Original calculation result

        Returns:
            Dict[str, Any]: Processed result
        """
        if not isinstance(result, dict):
            return result

        config = get_config()
        processed_result = {}

        for key, value in result.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    processed_result[key] = round(value, config.DEFAULT_PRECISION)
                else:
                    processed_result[key] = value
            elif isinstance(value, bool):
                processed_result[key] = value
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                processed_result[key] = self._postprocess_result(value)
            else:
                processed_result[key] = value

        return processed_result

    # ========================= Calculation Execution Method ===============================

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:

        self._preprocess_binary_operands(strategy_1, strategy_2, tnorm)
        result = self._execute_binary_op_impl(strategy_1, strategy_2, tnorm)
        processed_result = self._postprocess_result(result)
        return processed_result

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:

        self._preprocess_unary_operand(strategy, operand, tnorm)
        result = self._execute_unary_op_operand_impl(strategy, operand, tnorm)
        processed_result = self._postprocess_result(result)
        return processed_result

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:

        self._preprocess_pure_unary(strategy, tnorm)
        result = self._execute_unary_op_pure_impl(strategy, tnorm)
        processed_result = self._postprocess_result(result)
        return processed_result

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:

        self._preprocess_comparison(strategy_1, strategy_2, tnorm)
        result = self._execute_comparison_op_impl(strategy_1, strategy_2, tnorm)
        processed_result = self._postprocess_result(result)
        return processed_result

    def execute_fuzzarray_op(self,
                             fuzzarray_1: Any,
                             other: Optional[Any],
                             tnorm: OperationTNorm) -> Any:

        result = self._execute_fuzzarray_op_impl(fuzzarray_1, other, tnorm)
        return result

    # ========= Actual operation execution method (subclasses need to override)==============

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """The actual implementation of the binary operation (subclass override)）"""
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """Actual implementation of unary operations (with operands) (subclasses override)）"""
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """Actual implementation of pure unary operations (subclass override)"""
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """Actual implementation of comparison operations (overridden in subclasses)"""
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """Actual implementation of Fuzzarray operations (subclass override)"""
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationRegistry:

    def __init__(self):
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}
        self._default_t_norm_type: str = 'algebraic'
        self._default_t_norm_params: Dict[str, Any] = {}

    def switch_t_norm(self, t_norm_type: str, **params: Any):
        self._default_t_norm_type = t_norm_type
        self._default_t_norm_params = params

    def get_default_t_norm_config(self) -> tuple[str, Dict[str, Any]]:
        return self._default_t_norm_type, self._default_t_norm_params

    def register(self, operation: OperationMixin) -> None:

        op_name = operation.get_operation_name()
        if op_name not in self._operations:
            self._operations[op_name] = {}

        for mtype in operation.get_supported_mtypes():
            if mtype in self._operations[op_name]:
                warnings.warn(f"Operation '{op_name}' for mtype '{mtype}' already registered.")
            self._operations[op_name][mtype] = operation

    def get_operation(self, op_name: str, mtype: str) -> Optional[OperationMixin]:
        """Get operation instance."""
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_ops(self, mtype: str) -> List[str]:
        """Get list of available operations for mtype."""
        operations = []
        for op_name, mtype_ops in self._operations.items():
            if mtype in mtype_ops:
                operations.append(op_name)
        return operations


# ==================== Get Global Calculation Registry ======================

_operation_registry = OperationRegistry()


def get_operation_registry() -> OperationRegistry:
    """Get the global operation registry instance."""
    return _operation_registry
