#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/2 19:50
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines the abstract base classes for operations (`OperationMixin`)
and a central registry (`OperationScheduler`) for managing and dispatching
various fuzzy number operations within the FuzzLab framework.

It provides a structured way to:
- Define common interfaces for different types of operations (binary, unary, comparison, Fuzzarray).
- Implement preprocessing and post-processing steps for operation execution.
- Register and retrieve specific operation implementations based on fuzzy number types (mtype).
- Monitor and report performance statistics for all executed operations.

Classes:
    OperationMixin: An abstract base class that defines the interface for all
                    fuzzy number operations. Subclasses must implement specific
                    operation logic.
    OperationScheduler: A singleton class that acts as a registry for `OperationMixin`
                        instances, manages default T-norm configurations, and
                        collects performance metrics.

Functions:
    get_operation_registry(): Returns the global singleton instance of `OperationScheduler`.
"""
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Any, Dict, Union, Optional

from fuzzlab.config import get_config
from fuzzlab.core.triangular import OperationTNorm


class OperationMixin(ABC):
    """
    Abstract base class for all fuzzy number operations.

    This class defines the common interface and provides utility methods for
    various operations (e.g., addition, subtraction, comparison) that can be
    performed on fuzzy numbers. Subclasses must implement the specific logic
    for each operation type.

    It includes:
    - Abstract methods for defining operation name and supported fuzzy number types.
    - Static methods for preprocessing operands before execution, ensuring type
      and value validity.
    - Methods for post-processing results, such as rounding floating-point numbers.
    - Execution methods that wrap the actual operation logic, including performance
      timing and error handling.
    - Abstract methods for the actual implementation of different operation types,
      to be overridden by concrete subclasses.
    """

    @abstractmethod
    def get_operation_name(self) -> str:
        """
        Returns the unique name of the operation (e.g., 'add', 'mul', 'gt').

        This name is used to identify and retrieve the operation from the
        `OperationScheduler`.

        Returns:
            str: The name of the operation.
        """
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number membership types (mtypes) that this
        operation implementation supports.

        For example, an operation might only be defined for 'intuitionistic'
        fuzzy numbers, or for both 'intuitionistic' and 'pythagorean'.

        Returns:
            List[str]: A list of supported mtype strings.
        """
        pass

    def supports(self, mtype: str) -> bool:
        """
        Checks if this operation supports a given fuzzy number membership type.

        Args:
            mtype (str): The membership type to check.

        Returns:
            bool: True if the operation supports the mtype, False otherwise.
        """
        return mtype in self.get_supported_mtypes()

    # ======================= Preprocessing before calculation ============================

    @staticmethod
    def _preprocess_binary_operands(strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> None:
        """
        Performs data preprocessing and validation for binary operations.

        This static method ensures that both operands are not None, and that
        their `mtype` and `q` (q-rung) attributes match, if present. It also
        validates the presence of a T-norm instance.

        Args:
            strategy_1 (Any): The first strategy instance (e.g., FuzznumStrategy subclass).
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance to be used for the operation.

        Raises:
            ValueError: If `strategy_1`, `strategy_2`, or `tnorm` is None.
            TypeError: If `mtype` or `q` values of the strategies do not match.
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
        Performs data preprocessing and validation for unary operations involving a scalar operand.

        This static method ensures the strategy is not None, the operand is an
        integer or float, and a T-norm instance is provided.

        Args:
            strategy (Any): The strategy instance.
            operand (Union[int, float]): The scalar operand (e.g., for multiplication by a scalar).
            tnorm (OperationTNorm): The T-norm instance.

        Raises:
            ValueError: If `strategy` or `tnorm` is None.
            TypeError: If `operand` is not an int or float.
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
        Performs data preprocessing and validation for pure unary operations (no additional operand).

        This static method ensures the strategy is not None and a T-norm instance is provided.

        Args:
            strategy (Any): The strategy instance.
            tnorm (OperationTNorm): The T-norm instance.

        Raises:
            ValueError: If `strategy` or `tnorm` is None.
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
        Performs data preprocessing and validation for comparison operations.

        Similar to binary operations, this method checks for non-None operands,
        matching `mtype` and `q` values, and the presence of a T-norm instance.

        Args:
            strategy_1 (Any): The first strategy instance.
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance.

        Raises:
            ValueError: If `strategy_1`, `strategy_2`, or `tnorm` is None.
            TypeError: If `mtype` or `q` values of the strategies do not match.
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

    # def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Performs post-processing on the calculation results.
    #
    #     This method recursively rounds floating-point values in the result
    #     dictionary to the `DEFAULT_PRECISION` defined in the FuzzLab configuration.
    #     It ensures consistent precision for all numerical outputs.
    #
    #     Args:
    #         result (Dict[str, Any]): The original calculation result dictionary.
    #
    #     Returns:
    #         Dict[str, Any]: The processed result dictionary with rounded float values.
    #     """
    #     if not isinstance(result, dict):
    #         return result
    #
    #     config = get_config()
    #     processed_result = {}
    #
    #     for key, value in result.items():
    #         if isinstance(value, (int, float)):
    #             if isinstance(value, float):
    #                 processed_result[key] = round(value, config.DEFAULT_PRECISION)
    #             else:
    #                 processed_result[key] = value
    #         elif isinstance(value, bool):
    #             processed_result[key] = value
    #         elif isinstance(value, dict):
    #             # Recursively process nested dictionaries
    #             processed_result[key] = self._postprocess_result(value)
    #         else:
    #             processed_result[key] = value
    #
    #     return processed_result

    # ========================= Calculation Execution Method ===============================

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a binary operation between two strategy instances.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Args:
            strategy_1 (Any): The first strategy instance.
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance to use.

        Returns:
            Dict[str, Any]: The result of the binary operation.

        Raises:
            ValueError: If preprocessing fails.
            TypeError: If preprocessing fails due to type mismatch.
            NotImplementedError: If `_execute_binary_op_impl` is not overridden by subclass.
        """
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_binary_operands(strategy_1, strategy_2, tnorm)
            result = self._execute_binary_op_impl(strategy_1, strategy_2, tnorm)
            # processed_result = self._postprocess_result(result)
            # return processed_result
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'binary', execution_time)

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a unary operation involving a strategy instance and a scalar operand.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Args:
            strategy (Any): The strategy instance.
            operand (Union[int, float]): The scalar operand.
            tnorm (OperationTNorm): The T-norm instance to use.

        Returns:
            Dict[str, Any]: The result of the unary operation.

        Raises:
            ValueError: If preprocessing fails.
            TypeError: If preprocessing fails due to type mismatch.
            NotImplementedError: If `_execute_unary_op_operand_impl` is not overridden by subclass.
        """
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_unary_operand(strategy, operand, tnorm)
            result = self._execute_unary_op_operand_impl(strategy, operand, tnorm)
            # processed_result = self._postprocess_result(result)
            # return processed_result
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'unary_operand', execution_time)

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a pure unary operation on a strategy instance (no additional operand).

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Args:
            strategy (Any): The strategy instance.
            tnorm (OperationTNorm): The T-norm instance to use.

        Returns:
            Dict[str, Any]: The result of the pure unary operation.

        Raises:
            ValueError: If preprocessing fails.
            NotImplementedError: If `_execute_unary_op_pure_impl` is not overridden by subclass.
        """
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_pure_unary(strategy, tnorm)
            result = self._execute_unary_op_pure_impl(strategy, tnorm)
            # processed_result = self._postprocess_result(result)
            # return processed_result
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'unary_pure', execution_time)

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes a comparison operation between two strategy instances.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Args:
            strategy_1 (Any): The first strategy instance.
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance to use.

        Returns:
            Dict[str, bool]: The result of the comparison operation (e.g., {'value': True/False}).

        Raises:
            ValueError: If preprocessing fails.
            TypeError: If preprocessing fails due to type mismatch.
            NotImplementedError: If `_execute_comparison_op_impl` is not overridden by subclass.
        """
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            self._preprocess_comparison(strategy_1, strategy_2, tnorm)
            result = self._execute_comparison_op_impl(strategy_1, strategy_2, tnorm)
            # processed_result = self._postprocess_result(result)
            # return processed_result
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_operation_registry()._record_operation_time(op_name, 'comparison', execution_time)

    def execute_fuzzarray_op(self,
                             fuzzarray_1: Any,
                             other: Optional[Any],
                             tnorm: OperationTNorm) -> Any:
        """
        Executes an operation specifically designed for Fuzzarray instances.

        This method calls the concrete implementation and records performance metrics.
        Preprocessing for Fuzzarray operations is typically handled within the
        `_execute_fuzzarray_op_impl` itself due to their complex nature.

        Args:
            fuzzarray_1 (Any): The first Fuzzarray instance.
            other (Optional[Any]): The second operand, which can be another Fuzzarray,
                                   a Fuzznum, a scalar, or a NumPy array.
            tnorm (OperationTNorm): The T-norm instance to use.

        Returns:
            Any: The result of the Fuzzarray operation (e.g., a new Fuzzarray or a boolean).

        Raises:
            NotImplementedError: If `_execute_fuzzarray_op_impl` is not overridden by subclass.
        """
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
        """
        Abstract method for the actual implementation of a binary operation.

        Subclasses can override this method to define the specific logic for
        operations like addition, subtraction, etc., between two fuzzy number strategies.

        Args:
            strategy_1 (Any): The first strategy instance.
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance.

        Returns:
            Dict[str, Any]: The raw result of the operation before post-processing.
        """
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Abstract method for the actual implementation of a unary operation with a scalar operand.

        Subclasses can override this method to define the specific logic for
        operations like scalar multiplication or division.

        Args:
            strategy (Any): The strategy instance.
            operand (Union[int, float]): The scalar operand.
            tnorm (OperationTNorm): The T-norm instance.

        Returns:
            Dict[str, Any]: The raw result of the operation before post-processing.
        """
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Abstract method for the actual implementation of a pure unary operation.

        Subclasses can override this method to define the specific logic for
        operations that only involve a single fuzzy number strategy, without
        an additional scalar operand (e.g., negation, complement).

        Args:
            strategy (Any): The strategy instance.
            tnorm (OperationTNorm): The T-norm instance.

        Returns:
            Dict[str, Any]: The raw result of the operation before post-processing.
        """
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Abstract method for the actual implementation of a comparison operation.

        Subclasses can override this method to define the specific logic for
        comparison operations (e.g., greater than, less than, equality).

        Args:
            strategy_1 (Any): The first strategy instance.
            strategy_2 (Any): The second strategy instance.
            tnorm (OperationTNorm): The T-norm instance.

        Returns:
            Dict[str, bool]: The raw boolean result of the comparison (e.g., {'value': True}).
        """
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Abstract method for the actual implementation of a Fuzzarray operation.

        Subclasses can override this method to define the specific logic for
        operations involving `Fuzzarray` instances. This might include element-wise
        operations or operations that change the structure of the array.

        Args:
            fuzzarray_1 (Any): The first Fuzzarray instance.
            other (Optional[Any]): The second operand.
            tnorm (OperationTNorm): The T-norm instance.

        Returns:
            Any: The result of the Fuzzarray operation.
        """
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationScheduler:
    """
    A singleton class that acts as a central registry and dispatcher for fuzzy number operations.

    This scheduler manages:
    - Registration of `OperationMixin` implementations for different fuzzy number types.
    - Configuration of the default T-norm used for operations.
    - Collection and reporting of performance statistics for all executed operations.

    Attributes:
        _operations (Dict[str, Dict[str, OperationMixin]]): A nested dictionary
            storing registered operations. The structure is `{op_name: {mtype: operation_instance}}`.
        _default_t_norm_type (str): The type of the default T-norm (e.g., 'algebraic').
        _default_t_norm_params (Dict[str, Any]): Parameters for the default T-norm.
        _performance_stats (Dict[str, Any]): A dictionary holding global performance metrics.
        _stats_lock (threading.Lock): A lock to ensure thread-safe access to performance statistics.
    """

    def __init__(self):
        """
        Initializes the OperationScheduler.

        Sets up the internal dictionaries for operations and performance statistics,
        and initializes the default T-norm configuration.
        """
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
        """
        Switches the default T-norm used for operations.

        Args:
            t_norm_type (str): The type of the T-norm (e.g., 'algebraic', 'lukasiewicz').
            **params (Any): Additional parameters specific to the T-norm type.
        """
        self._default_t_norm_type = t_norm_type
        self._default_t_norm_params = params

    def get_default_t_norm_config(self) -> tuple[str, Dict[str, Any]]:
        """
        Returns the current default T-norm configuration.

        Returns:
            tuple[str, Dict[str, Any]]: A tuple containing the T-norm type and its parameters.
        """
        return self._default_t_norm_type, self._default_t_norm_params

    def register(self, operation: OperationMixin) -> None:
        """
        Registers an `OperationMixin` instance with the scheduler.

        Operations are registered based on their `op_name` and supported `mtype`s.
        A warning is issued if an operation for a specific `mtype` is already registered.

        Args:
            operation (OperationMixin): The operation instance to register.
        """
        op_name = operation.get_operation_name()
        if op_name not in self._operations:
            self._operations[op_name] = {}

        for mtype in operation.get_supported_mtypes():
            if mtype in self._operations[op_name]:
                warnings.warn(f"Operation '{op_name}' for mtype '{mtype}' already registered.")
            self._operations[op_name][mtype] = operation

    def get_operation(self, op_name: str, mtype: str) -> Optional[OperationMixin]:
        """
        Retrieves a registered operation instance.

        Args:
            op_name (str): The name of the operation.
            mtype (str): The fuzzy number membership type.

        Returns:
            Optional[OperationMixin]: The `OperationMixin` instance if found, otherwise None.
        """
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_ops(self, mtype: str) -> List[str]:
        """
        Returns a list of operation names available for a specific fuzzy number membership type.

        Args:
            mtype (str): The fuzzy number membership type.

        Returns:
            List[str]: A list of operation names.
        """
        operations = []
        for op_name, mtype_ops in self._operations.items():
            if mtype in mtype_ops:
                operations.append(op_name)
        return operations

    # ======================== Performance Monitoring Method ========================

    def _record_operation_time(self, op_name: str, op_type: str, execution_time: float) -> None:
        """
        Records the execution time for a specific operation.

        This method is called internally by `OperationMixin` execution methods
        to collect performance data. It is thread-safe.

        Args:
            op_name (str): The name of the operation (e.g., 'add', 'sub').
            op_type (str): The type of operation (e.g., 'binary', 'unary_operand', 'fuzzarray').
            execution_time (float): The execution time in seconds.
        """
        with self._stats_lock:
            self._performance_stats['total_operations'] += 1
            self._performance_stats['total_time'] += execution_time
            self._performance_stats['operation_counts'][op_name][op_type] += 1
            self._performance_stats['operation_times'][op_name][op_type].append(execution_time)

            # Calculate average time
            times = self._performance_stats['operation_times'][op_name][op_type]
            self._performance_stats['average_times'][op_name][op_type] = sum(times) / len(times)

    def get_performance_stats(self, time_unit: str = 'us') -> Dict[str, Any]:
        """
        Obtains global performance statistics for all operations.

        The statistics include total operations, total time, average time per operation,
        and counts/average times grouped by operation name and type. Time values
        are converted to the specified unit and rounded to the default precision.

        Args:
            time_unit (str, optional):
                Unit used to display time.
                Supported values include 's' (seconds), 'ms' (milliseconds),
                'us' (microseconds), and 'ns' (nanoseconds).
                Defaults to 'us' (microseconds).

        Returns:
            Dict[str, Any]: A dictionary containing performance statistics.

        Raises:
            RuntimeError: If an unsupported `time_unit` is provided.
        """
        time_unit_dict = {
            's': 1,
            'ms': 1e+3,
            'us': 1e+6,
            'ns': 1e+9
        }

        if time_unit not in time_unit_dict:
            raise RuntimeError(f"Unsupported time unit '{time_unit}'. Supported time units are {list(time_unit_dict.keys())}")

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
        """
        Resets all collected performance statistics to their initial state.

        This method is thread-safe.
        """
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
    """
    Returns the global singleton instance of the `OperationScheduler`.

    This function provides a convenient way to access the central registry
    for operations and performance monitoring throughout the FuzzLab application.

    Returns:
        OperationScheduler: The global `OperationScheduler` instance.
    """
    return _operation_registry