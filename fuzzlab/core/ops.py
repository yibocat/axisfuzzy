#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 19:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
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
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_binary_operands(strategy_1, strategy_2, tnorm)
            result = self._execute_binary_op_impl(strategy_1, strategy_2, tnorm)
            processed_result = self._postprocess_result(result)
            return processed_result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'binary', execution_time)

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_unary_operand(strategy, operand, tnorm)
            result = self._execute_unary_op_operand_impl(strategy, operand, tnorm)
            processed_result = self._postprocess_result(result)
            return processed_result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'unary_operand', execution_time)

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_pure_unary(strategy, tnorm)
            result = self._execute_unary_op_pure_impl(strategy, tnorm)
            processed_result = self._postprocess_result(result)
            return processed_result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'unary_pure', execution_time)

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_comparison(strategy_1, strategy_2, tnorm)
            result = self._execute_comparison_op_impl(strategy_1, strategy_2, tnorm)
            processed_result = self._postprocess_result(result)
            return processed_result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'comparison', execution_time)

    def execute_fuzzarray_op(self,
                             fuzzarray_1: Any,
                             other: Optional[Any],
                             tnorm: OperationTNorm) -> Any:
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            result = self._execute_fuzzarray_op_impl(fuzzarray_1, other, tnorm)
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'fuzzarray', execution_time)

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


class OperationScheduler:

    def __init__(self):
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}
        self._default_t_norm_type: str = 'algebraic'
        self._default_t_norm_params: Dict[str, Any] = {}

        # Global performance statistics
        self._performance_stats = {
            'total_operations': 0,
            'total_time': 0.0,
            'operation_counts': defaultdict(lambda: defaultdict(int)),  # {op_name: {op_type: count}}
            'operation_times': defaultdict(lambda: defaultdict(list)),  # {op_name: {op_type: [times]}}
            'average_times': defaultdict(lambda: defaultdict(float))  # {op_name: {op_type: avg_time}}
        }
        self._stats_lock = threading.Lock()

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

    # ======================== Performance Monitoring Method ========================

    def _record_operation_time(self, op_name: str, op_type: str, execution_time: float) -> None:
        """
        记录操作执行时间
        Args:
            op_name: 操作名称 (e.g., 'add', 'sub')
            op_type: 操作类型 (e.g., 'binary', 'unary_operand', 'fuzzarray')
            execution_time: 执行时间（秒）
        """
        with self._stats_lock:
            self._performance_stats['total_operations'] += 1
            self._performance_stats['total_time'] += execution_time
            self._performance_stats['operation_counts'][op_name][op_type] += 1
            self._performance_stats['operation_times'][op_name][op_type].append(execution_time)

            # 计算平均时间
            times = self._performance_stats['operation_times'][op_name][op_type]
            self._performance_stats['average_times'][op_name][op_type] = sum(times) / len(times)

    def get_performance_stats(self, time_unit: str = 'us') -> Dict[str, Any]:
        """
        Obtain global performance statistics.

        - 'total_operations': Total number of operations.
        - 'total_time({time_unit})': Total execution time, converted to the specified
            time_unit and rounded to the default precision.
        - 'average_time_per_total_operation({time_unit})': The average execution time per operation, 
            converted to the specified time_unit and rounded to the default precision.
            If the total number of operations is 0, then it is 0.0.
        - 'operation_counts_by_type': Operation count grouped by operation name and type. 
            The format is {op_name: {op_type: count}}.
        - 'average_times_by_operation_type({time_unit})': Average execution time grouped 
            by operation name and type. The format is {op_name: {op_type: avg_time}},
            the time has been converted to the specified time_unit and rounded to the default precision.

        Args:
            time_unit (str, optional):
                Unit used to display time.
                The optional values include 's' (seconds), 'ms' (milliseconds),
                'us' (microseconds), and 'ns' (nanoseconds).
                Defaults to 'us' (microseconds).

        Returns:
            Dict[str, Any]: A dictionary containing performance statistics.
        """
        time_unit_dict = {
            's': 1,
            'ms': 1e+3,
            'us': 1e+6,
            'ns': 1e+9
        }

        if time_unit not in time_unit_dict:
            raise RuntimeError(f"Unsupported time unit '{time_unit}'. Supported time units are {list(time_unit_dict)}")

        with self._stats_lock:
            stats = {
                'total_operations': self._performance_stats['total_operations'],
                f'total_time({time_unit})':
                    round(
                        self._performance_stats['total_time']
                        * time_unit_dict[time_unit],
                        get_config().DEFAULT_PRECISION),
                f'average_time_per_total_operation({time_unit})': (
                    round(
                        (self._performance_stats['total_time'] / self._performance_stats['total_operations'])
                        * time_unit_dict[time_unit],
                        get_config().DEFAULT_PRECISION
                    )
                    if self._performance_stats['total_operations'] > 0 else 0.0
                ),
                'operation_counts_by_type': {
                    op_name: dict(op_types)
                    for op_name, op_types in self._performance_stats['operation_counts'].items()
                },
                f'average_times_by_operation_type({time_unit})': {
                    op_name: {
                        op_type: round(
                            avg_time * time_unit_dict[time_unit],
                            get_config().DEFAULT_PRECISION)
                        for op_type, avg_time in op_types.items()
                    }
                    for op_name, op_types in self._performance_stats['average_times'].items()
                }
            }
            return stats

    def reset_performance_stats(self) -> None:
        """Reset all performance statistics"""
        with self._stats_lock:
            self._performance_stats = {
                'total_operations': 0,
                'total_time': 0.0,
                'operation_counts': defaultdict(lambda: defaultdict(int)),
                'operation_times': defaultdict(lambda: defaultdict(list)),
                'average_times': defaultdict(lambda: defaultdict(float))
            }


# ==================== Get Global Calculation Registry ======================

_operation_registry = OperationScheduler()


def get_operation_registry() -> OperationScheduler:
    """Get the global operation registry instance."""
    return _operation_registry
