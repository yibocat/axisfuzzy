#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
axisfuzzy.core.operation
========================

This module defines the abstract base classes for operations (``OperationMixin``)
and a central registry (``OperationScheduler``) for managing and dispatching
various fuzzy number operations within the FuzzLab framework.

It provides a structured way to:
- Define common interfaces for different types of operations (binary, unary, comparison, Fuzzarray).
- Implement preprocessing and post-processing steps for operation execution.
- Register and retrieve specific operation implementations based on fuzzy number types (``mtype``).
- Monitor and report performance statistics for all executed operations.

This module defines:
- OperationMixin: abstract interface for per-mtype operation implementations.
- OperationScheduler: singleton registry and performance monitor for operations.
- Utilities: registration decorator and access to the global scheduler.

"""

import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Any, Dict, Union, Optional

from ..config import get_config
from .triangular import OperationTNorm


class OperationMixin(ABC):
    """
    Abstract base class for fuzzy number operations.

    This class defines the common interface and provides utility methods for
    various operations (e.g., addition, subtraction, comparison) that can be
    performed on fuzzy numbers. Subclasses must implement the specific logic
    for each operation type.

    Includes:
    ------------
    - Abstract methods for defining operation name and supported fuzzy number types.
    - Static methods for preprocessing operands before execution, ensuring type
      and value validity.
    - Methods for post-processing results, such as rounding floating-point numbers.
    - Execution methods that wrap the actual operation logic, including performance
      timing and error handling.
    - Abstract methods for the actual implementation of different operation types,
      to be overridden by concrete subclasses.

    Function:
    --------
    get_operation_name()
        Return the unique operation name used by the registry.
    get_supported_mtypes()
        Return a list of supported mtype strings.
    execute_binary_op(strategy_1, strategy_2, tnorm)
        Preprocess, execute and time a binary operation.
    execute_unary_op_operand(strategy, operand, tnorm)
        Preprocess, execute and time a unary operation with scalar operand.
    execute_unary_op_pure(strategy, tnorm)
        Preprocess, execute and time a pure unary operation.
    execute_comparison_op(strategy_1, strategy_2, tnorm)
        Preprocess, execute and time a comparison operation.
    execute_fuzzarray_op(fuzzarray_1, other, tnorm)
        Execute an operation specialized for Fuzzarray instances.

    Notes
    -----
    Concrete subclasses must override the `_execute_*_impl` methods to
    provide operation logic. The public `execute_*` wrappers handle
    preprocessing and performance recording via :func:`get_registry_operation`.
    """

    @abstractmethod
    def get_operation_name(self) -> str:
        """
        Returns the unique name of the operation (e.g., 'add', 'mul', 'gt').

        This name is used to identify and retrieve the operation from the
        ``OperationScheduler``.

        Returns
        -------
        str
            Operation name used as key in the operation registry.
        """
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number membership types (mtypes) that this
        operation implementation supports.

        For example, an operation might only be defined for 'intuitionistic'
        fuzzy numbers, or for both 'intuitionistic' and 'pythagorean'.

        Returns
        -------
        list of str
            mtype strings for which this operation implementation is applicable.
        """
        pass

    def supports(self, mtype: str) -> bool:
        """
        Check whether the operation supports a given mtype.

        Parameters
        ----------
        mtype : str
            Membership-type string to check.

        Returns
        -------
        bool
            True if supported, False otherwise.
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
        their ``mtype`` and ``q`` (q-rung) attributes match, if present. It also
        validates the presence of a T-norm instance.

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Strategy instances (typically subclasses of FuzznumStrategy).
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If ``strategy_1``, ``strategy_2``, or ``tnorm`` is None.
        TypeError
            If ``mtype`` or ``q`` values of the strategies do not match.
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

        Parameters
        ----------
        strategy : object
            Strategy instance.
        operand : int or float
            Scalar operand.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If `strategy` or `tnorm` is None.
        TypeError
            If `operand` is not an int or float.
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

        Parameters
        ----------
        strategy : object
            Strategy instance.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If ``strategy`` or ``tnorm`` is None.
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

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Strategy instances.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
            ValueError
                If ``strategy_1``, ``strategy_2``, or `tnorm` is None.
            TypeError
                If ``mtype`` or ``q`` values of the strategies do not match.
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

    # ========================= Calculation Execution Method ===============================

    def execute_binary_op(self,
                          strategy_1: Any,
                          strategy_2: Any,
                          tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a binary operation between two strategy instances.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Operand strategy instances.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Return
        -------
        Dict[str, Any]
            The result of the binary operation.

        Raises
        ------
        ValueError
            If preprocessing fails.
        TypeError
            If preprocessing fails due to type mismatch.
        NotImplementedError
            If ``_execute_binary_op_impl`` is not overridden by subclass.
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
            get_registry_operation()._record_operation_time(op_name, 'binary', execution_time)

    def execute_unary_op_operand(self,
                                 strategy: Any,
                                 operand: Union[int, float],
                                 tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a unary operation involving a strategy instance and a scalar operand.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Parameters
        ----------
        strategy : object
            Strategy instance.
        operand : int or float
            Scalar operand.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Dict[str, Any]
            The result of the unary operation.

        Raises
        ------
        ValueError
            If preprocessing fails.
        TypeError
            If preprocessing fails due to type mismatch.
        NotImplementedError
            If ``_execute_unary_op_operand_impl`` is not overridden by subclass.
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
            get_registry_operation()._record_operation_time(op_name, 'unary_operand', execution_time)

    def execute_unary_op_pure(self,
                              strategy: Any,
                              tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Executes a pure unary operation on a strategy instance (no additional operand).

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Parameters
        ----------
        strategy : object
            Strategy instance.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Dict[str, Any]
            The result of the pure unary operation.

        Raises
        ------
        ValueError
            If preprocessing fails.
        NotImplementedError
            If ``_execute_unary_op_pure_impl`` is not overridden by subclass.
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
            get_registry_operation()._record_operation_time(op_name, 'unary_pure', execution_time)

    def execute_comparison_op(self,
                              strategy_1: Any,
                              strategy_2: Any,
                              tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Executes a comparison operation between two strategy instances.

        This method handles preprocessing, calls the concrete implementation,
        post-processes the result, and records performance metrics.

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Operand strategy instances.
        tnorm : OperationTNorm
            T-norm configuration used for the comparison.

        Returns
        -------
        Dict[str, bool]
            The result of the comparison operation (e.g., {'value': True/False}).

        Raises
        ------
        ValueError
            If preprocessing fails.
        TypeError
            If preprocessing fails due to type mismatch.
        NotImplementedError
            If ``_execute_comparison_op_impl`` is not overridden by subclass.
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
            get_registry_operation()._record_operation_time(op_name, 'comparison', execution_time)

    def execute_fuzzarray_op(self,
                             fuzzarray_1: Any,
                             other: Optional[Any],
                             tnorm: OperationTNorm) -> Any:
        """
        Executes an operation specifically designed for Fuzzarray instances.

        This method calls the concrete implementation and records performance metrics.
        Preprocessing for Fuzzarray operations is typically handled within the
        ``_execute_fuzzarray_op_impl`` itself due to their complex nature.

        Parameters
        ----------
        fuzzarray_1 : object
            Fuzzarray instance.
        other : object or None
            Second operand (Fuzzarray, Fuzznum, scalar or ndarray).
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Any
            The result of the Fuzzarray operation (e.g., a new Fuzzarray or a boolean).

        Raises
        ------
        NotImplementedError
            If ``_execute_fuzzarray_op_impl`` is not overridden by subclass.
        """
        op_name = self.get_operation_name()
        start_time = time.perf_counter()

        try:
            result = self._execute_fuzzarray_op_impl(fuzzarray_1, other, tnorm)
            return result
        finally:
            execution_time = time.perf_counter() - start_time
            get_registry_operation()._record_operation_time(op_name, 'fuzzarray', execution_time)

    # ========= Actual operation execution method (subclasses need to override)==============

    def _execute_binary_op_impl(self,
                                strategy_1: Any,
                                strategy_2: Any,
                                tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Implementation hook for binary operations.

        Subclasses should override this method.

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Operand strategy instances.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Implementation hook for unary operations with scalar operand.

        Parameters
        ----------
        strategy : object
            Strategy instance.
        operand : int or float
            Scalar operand.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Implementation hook for pure unary operations.

        Parameters
        ----------
        strategy : object
            Strategy instance.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Implementation hook for comparison operations.

        Parameters
        ----------
        strategy_1, strategy_2 : object
            Operand strategy instances.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Boolean result dictionary.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Implementation hook for Fuzzarray-level operations.

        Parameters
        ----------
        fuzzarray_1 : object
            Fuzzarray instance.
        other : object or None
            Second operand.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        object
            Result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationScheduler:
    """
    A singleton class that acts as a central registry and dispatcher for fuzzy number operations.

    This scheduler manages:
        - Registration of ``OperationMixin`` implementations for different fuzzy number types.
        - Configuration of the default T-norm used for operations.
        - Collection and reporting of performance statistics for all executed operations.

    Attributes
    ----------
    _operations : dict of str to dict of str to OperationMixin
        A nested dictionary storing registered operations. The structure is
        ``{op_name: {mtype: operation_instance}}``.
    _default_t_norm_type : str
        The type of the default T-norm (e.g., 'algebraic').
    _default_t_norm_params : dict of str to Any
        Parameters for the default T-norm.
    _performance_stats : dict of str to Any
        A dictionary holding global performance metrics.
    _stats_lock : threading.Lock
        A lock to ensure thread-safe access to performance statistics.
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
        self._stats_lock = threading.Lock()
        self.reset_performance_stats()

    def switch_t_norm(self, t_norm_type: str, **params: Any):
        """
        Change the scheduler's default t-norm configuration.

        Parameters
        ----------
        t_norm_type : str
            Name of the t-norm implementation (e.g., 'algebraic').
        **params : dict
            Implementation-specific parameters.
        """
        self._default_t_norm_type = t_norm_type
        self._default_t_norm_params = params

    def get_default_t_norm_config(self) -> tuple[str, Dict[str, Any]]:
        """
        Get the current default t-norm configuration.

        Returns
        -------
        tuple
            (t_norm_type, params)
        """
        return self._default_t_norm_type, self._default_t_norm_params

    def register(self, operation: OperationMixin) -> None:
        """
        Registers an ``OperationMixin`` instance with the scheduler.

        Operations are registered based on their ``op_name`` and supported ``mtype``s.
        A warning is issued if an operation for a specific ``mtype`` is already registered.

        Parameters
        ----------
        operation : OperationMixin
            The operation instance to register.
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
        Retrieve a registered operation implementation.

        Parameters
        ----------
        op_name : str
            Operation name.
        mtype : str
            Membership type.

        Returns
        -------
        OperationMixin or None
            Registered operation instance or None if not found.
        """
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_ops(self, mtype: str) -> List[str]:
        """
        List operation names available for a given mtype.

        Parameters
        ----------
        mtype : str
            Membership type.

        Returns
        -------
        list of str
            Available operation names.
        """
        operations = []
        for op_name, mtype_ops in self._operations.items():
            if mtype in mtype_ops:
                operations.append(op_name)
        return operations

    # ======================== Performance Monitoring Method ========================

    def _record_operation_time(self, op_name: str, op_type: str, execution_time: float) -> None:
        """
        Record execution time for an operation (thread-safe).

        Parameters
        ----------
        op_name : str
            Operation name.
        op_type : str
            One of 'binary', 'unary_operand', 'unary_pure', 'comparison', 'fuzzarray'.
        execution_time : float
            Execution time in seconds.
        """
        with self._stats_lock:
            self._performance_stats['total_operations'] += 1
            self._performance_stats['total_time'] += execution_time

            counts = self._performance_stats['operation_counts'][op_name]
            averages = self._performance_stats['average_times'][op_name]

            # Get old values before update
            old_count = counts[op_type]
            old_avg = averages[op_type]  # default_dict(float) will return 0.0 if not exists

            # Update count
            new_count = old_count + 1
            counts[op_type] = new_count

            # Update average time incrementally. This is crucial for performance.
            # The previous implementation with sum() over a growing list caused
            # a linear slowdown in execution time for repeated operations.
            new_avg = old_avg + (execution_time - old_avg) / new_count
            averages[op_type] = new_avg

    def get_performance_stats(self, time_unit: str = 'us') -> Dict[str, Any]:
        """
        Obtains global performance statistics for all operations.

        The statistics include total operations, total time, average time per operation,
        and counts/average times grouped by operation name and type. Time values
        are converted to the specified unit and rounded to the default precision.

        Parameters
        ----------
        time_unit : str, optional
            Unit used to display time.
            Supported values include 's' (seconds), 'ms' (milliseconds),
            'us' (microseconds), and 'ns' (nanoseconds).
            Defaults to 'us' (microseconds).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing performance statistics.

        Raises
        ------
        RuntimeError
            If an unsupported `time_unit` is provided.
        """
        time_unit_dict = {
            's': 1,
            'ms': 1e+3,
            'us': 1e+6,
            'ns': 1e+9
        }

        if time_unit not in time_unit_dict:
            raise RuntimeError(
                f"Unsupported time unit '{time_unit}'. Supported time units are {list(time_unit_dict.keys())}")

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
                'average_times': defaultdict(lambda: defaultdict(float))
            }


# ==================== Get Global Calculation Registry ======================

_operation_registry = OperationScheduler()


def get_registry_operation() -> OperationScheduler:
    """
    Access the global OperationScheduler singleton.

    Returns
    -------
    OperationScheduler
        The module-level scheduler instance.
    """
    return _operation_registry


# ==================== Operation Auto-Registration Decorator =================

def register_operation(cls_or_eager=None, *, eager: bool = True):
    """
    Class decorator that auto-registers OperationMixin subclasses with the global scheduler.

    This decorator supports two usage patterns:
    1. With parentheses: ``@register_operation()`` or ``@register_operation(eager=False)``
    2. Without parentheses: ``@register_operation``

    Parameters
    ----------
    cls_or_eager : type or None
        When used without parentheses this is the decorated class; otherwise None.
    eager : bool, optional
        If True (default) instantiate and register the operation immediately.

    Returns
    -------
    type
        The original class object.

    Raises
    ------
    TypeError
        If the decorated object is not a subclass of OperationMixin.

    Examples
    --------
    >>> @register_operation
    >>> class MyAddition(OperationMixin):
    >>>    operation_name = 'add'
    >>>    supported_mtypes = ['my_type']
    >>>    def execute(self, a, b, tnorm=None):
    >>>        # Implement addition logic
    >>>        return result

    Notes
    -----
    - The decorated class must inherit from ``OperationMixin``.
    - The class must define ``operation_name`` and ``supported_mtypes`` attributes.
    - Re-registering operations with the same name/type will follow the overriding policy of ``OperationScheduler``.
    - If eager is True the decorator will instantiate the class and call
      :meth:`OperationScheduler.register`. Errors during instantiation are
      reported as RuntimeError with contextual information.
    """
    def _register_class(cls):
        """Actual registration logic"""
        if not issubclass(cls, OperationMixin):
            raise TypeError(
                f"@register_operation can only be used on subclasses of OperationMixin, "
                f"but got {cls.__name__} (inherited from: {[base.__name__ for base in cls.__bases__]})"
            )

        if eager:
            try:
                operation_instance = cls()
                get_registry_operation().register(operation_instance)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register operation class {cls.__name__}: {e}. "
                    f"Please check if the class correctly implements the OperationMixin interface."
                ) from e

        return cls

    # Determine whether it is a no-parenthesis or parenthesized invocation
    if cls_or_eager is not None:
        # No-parenthesis form: @register_operation
        # cls_or_eager is the decorated class
        if isinstance(cls_or_eager, type):
            return _register_class(cls_or_eager)
        else:
            # This case should not happen, but for safety's sake
            raise TypeError(f"The first argument of @register_operation should be a class or None, "
                            f"got {type(cls_or_eager)}")
    else:
        # Parenthesized form: @register_operation() or @register_operation(eager=False)
        # Return the decorator function
        return _register_class
