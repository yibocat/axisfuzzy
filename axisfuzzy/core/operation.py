#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
Operating system for fuzzy numbers and arrays.

This module defines the abstract base classes for operations (``OperationMixin``)
and a central registry (``OperationScheduler``) for managing and dispatching
various fuzzy number operations within the AxisFuzzy framework.

Overview
--------
- Provides a unified interface for all fuzzy number operations (arithmetic, logic, comparison, etc).
- Supports registration and dispatch of per-mtype operation implementations.
- Handles preprocessing, validation, and performance monitoring for all operations.
- Enables efficient batch operations on Fuzzarray via backend-aware implementations.

Key Classes
-----------
- :class:`OperationMixin`: Abstract interface for per-mtype operation implementations.
- :class:`OperationScheduler`: Singleton registry and performance monitor for operations.
- Utilities: Registration decorator and access to the global scheduler.

Notes
-----
- All operation implementations must inherit from :class:`OperationMixin` and register via ``@register_operation``.
- The scheduler supports dynamic switching of t-norms and collects global performance statistics.
- Operator overloading in :class:`~.fuzznums.Fuzznum` and :class:`~.fuzzarray.Fuzzarray` is dispatched via this system.

Examples
--------
.. code-block:: python

    # Register a new operation for a custom mtype
    @register_operation
    class MyAdd(OperationMixin):
        def get_operation_name(self): return 'add'
        def get_supported_mtypes(self): return ['mytype']
        def _execute_binary_op_impl(self, s1, s2, tnorm): ...
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

    Responsibilities
    ----------------
    - Define operation name and supported mtypes.
    - Provide preprocessing and validation for operands.
    - Implement core logic for binary, unary, comparison, and Fuzzarray-level operations.
    - Record performance statistics for each operation type.

    Notes
    -----
    - Subclasses must override the `_execute_*_impl` methods for supported operations.
    - Public `execute_*` methods handle preprocessing, timing, and error handling.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.core import OperationMixin, register_operation, OperationTNorm

        @register_operation
        class QROFNAddition(OperationMixin):
            def get_operation_name(self) -> str: return 'add'
            def get_supported_mtypes(self) -> list[str]: return ['qrofn']
            def _execute_binary_op_impl(self, s1, s2, tnorm: OperationTNorm):
                md = tnorm.t_conorm(s1.md, s2.md)
                nmd = tnorm.t_norm(s1.nmd, s2.nmd)
                return {'md': md, 'nmd': nmd, 'q': s1.q}
    """

    @abstractmethod
    def get_operation_name(self) -> str:
        """
        Returns the unique name of the operation (e.g., 'add', 'mul', 'gt').

        This name is used to identify and retrieve the operation from the
        :class:`OperationScheduler`.

        Returns
        -------
        str
            Operation name used as key in the operation registry.

        Examples
        --------
        .. code-block:: python

            op = QROFNAddition()
            print(op.get_operation_name())  # 'add'
        """
        pass

    @abstractmethod
    def get_supported_mtypes(self) -> List[str]:
        """
        Returns a list of fuzzy number membership types (mtypes) that this
        operation implementation supports.

        For example, an operation might only be defined for 'qrofn'
        fuzzy numbers, or for both 'qrofn' and 'qrohfn'.

        Returns
        -------
        list of str
            mtype strings for which this operation implementation is applicable.

        Examples
        --------
        .. code-block:: python

            op = QROFNAddition()
            print(op.get_supported_mtypes())  # ['qrofn']
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

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.fuzztype.qrofs.op import QROFNAddition
            op = QROFNAddition()
            print(op.supports('qrofn'))  # True
            print(op.supports('qrohfn')) # False
        """
        return mtype in self.get_supported_mtypes()

    # ======================= Preprocessing before calculation ============================

    @staticmethod
    def _preprocess_binary_operands(strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> None:
        """
        Performs data preprocessing and validation for binary operations.

        Ensures both operands are not None, and that their ``mtype`` and ``q``
        attributes match, if present. Also validates the presence of a T-norm instance.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first strategy instance.
        strategy_2 : FuzznumStrategy
            The second strategy instance.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If `strategy_1`, `strategy_2`, or `tnorm` is None.
        TypeError
            If `mtype` or `q` values of the strategies do not match.
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
        Performs data preprocessing for unary operations involving a scalar operand.

        Ensures the strategy is not None, the operand is an integer or float,
        and a T-norm instance is provided.

        Parameters
        ----------
        strategy : FuzznumStrategy
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
        Performs data preprocessing for pure unary operations (no additional operand).

        Ensures the strategy is not None and a T-norm instance is provided.

        Parameters
        ----------
        strategy : FuzznumStrategy
            Strategy instance.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If `strategy` or `tnorm` is None.
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
        Performs data preprocessing for comparison operations.

        Checks for non-None operands, matching `mtype` and `q` values, and the presence of a T-norm instance.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first strategy instance.
        strategy_2 : FuzznumStrategy
            The second strategy instance.
        tnorm : OperationTNorm
            T-norm configuration object.

        Raises
        ------
        ValueError
            If `strategy_1`, `strategy_2`, or `tnorm` is None.
        TypeError
            If `mtype` or `q` values of the strategies do not match.
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

        Handles preprocessing, calls the concrete implementation,
        and records performance metrics.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first operand strategy instance.
        strategy_2 : FuzznumStrategy
            The second operand strategy instance.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Dict[str, Any]
            The result of the binary operation, typically a dictionary of
            component values for a new fuzzy number.

        Raises
        ------
        NotImplementedError
            If `_execute_binary_op_impl` is not overridden by the subclass.

        Examples
        --------
        .. code-block:: python

            from axisfuzzy.fuzztype.qrofs.op import QROFNAddition
            from axisfuzzy.core import fuzznum, OperationTNorm

            op = QROFNAddition()
            a = fuzznum(mtype='qrofn', md=0.5, nmd=0.3, q=2)._strategy_instance
            b = fuzznum(mtype='qrofn', md=0.6, nmd=0.2, q=2)._strategy_instance
            tnorm = OperationTNorm(norm_type='algebraic', q=2)
            res = op.execute_binary_op(a, b, tnorm)
            # res is {'md': 0.8, 'nmd': 0.06, 'q': 2}
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

        Handles preprocessing, calls the concrete implementation,
        and records performance metrics.

        Parameters
        ----------
        strategy : FuzznumStrategy
            The strategy instance.
        operand : int or float
            The scalar operand (e.g., the exponent in a power operation).
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Dict[str, Any]
            The result of the unary operation.

        Raises
        ------
        NotImplementedError
            If `_execute_unary_op_operand_impl` is not overridden by the subclass.
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

        Handles preprocessing, calls the concrete implementation,
        and records performance metrics.

        Parameters
        ----------
        strategy : FuzznumStrategy
            The strategy instance.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Dict[str, Any]
            The result of the pure unary operation.

        Raises
        ------
        NotImplementedError
            If `_execute_unary_op_pure_impl` is not overridden by the subclass.
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

        Handles preprocessing, calls the concrete implementation,
        and records performance metrics.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first operand strategy instance.
        strategy_2 : FuzznumStrategy
            The second operand strategy instance.
        tnorm : OperationTNorm
            T-norm configuration used for the comparison.

        Returns
        -------
        Dict[str, bool]
            The result of the comparison, e.g., ``{'value': True}``.

        Raises
        ------
        NotImplementedError
            If `_execute_comparison_op_impl` is not overridden by the subclass.
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

        Calls the concrete implementation and records performance metrics.
        Preprocessing for Fuzzarray operations is typically handled within the
        `_execute_fuzzarray_op_impl` itself due to their complex nature.

        Parameters
        ----------
        fuzzarray_1 : Fuzzarray
            The first Fuzzarray instance.
        other : Fuzzarray, Fuzznum, scalar, or None
            The second operand.
        tnorm : OperationTNorm
            T-norm configuration used for the operation.

        Returns
        -------
        Any
            The result of the Fuzzarray operation (e.g., a new Fuzzarray or a boolean array).

        Raises
        ------
        NotImplementedError
            If `_execute_fuzzarray_op_impl` is not overridden by the subclass.
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
        Implementation hook for binary operations. Subclasses must override this.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first operand strategy instance.
        strategy_2 : FuzznumStrategy
            The second operand strategy instance.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by the subclass.
        """
        raise NotImplementedError(f"Binary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_operand_impl(self,
                                       strategy: Any,
                                       operand: Union[int, float],
                                       tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Implementation hook for unary operations with a scalar operand. Subclasses must override this.

        Parameters
        ----------
        strategy : FuzznumStrategy
            The strategy instance.
        operand : int or float
            The scalar operand.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by the subclass.
        """
        raise NotImplementedError(f"Unary operation with operand '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_unary_op_pure_impl(self,
                                    strategy: Any,
                                    tnorm: OperationTNorm) -> Dict[str, Any]:
        """
        Implementation hook for pure unary operations. Subclasses must override this.

        Parameters
        ----------
        strategy : FuzznumStrategy
            The strategy instance.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Raw result produced by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If not overridden by the subclass.
        """
        raise NotImplementedError(f"Pure unary operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_comparison_op_impl(self,
                                    strategy_1: Any,
                                    strategy_2: Any,
                                    tnorm: OperationTNorm) -> Dict[str, bool]:
        """
        Implementation hook for comparison operations. Subclasses must override this.

        Parameters
        ----------
        strategy_1 : FuzznumStrategy
            The first operand strategy instance.
        strategy_2 : FuzznumStrategy
            The second operand strategy instance.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        dict
            Boolean result dictionary, e.g., ``{'value': True}``.

        Raises
        ------
        NotImplementedError
            If not overridden by the subclass.
        """
        raise NotImplementedError(f"Comparison operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")

    def _execute_fuzzarray_op_impl(self,
                                   fuzzarray_1: Any,
                                   other: Optional[Any],
                                   tnorm: OperationTNorm) -> Any:
        """
        Implementation hook for Fuzzarray-level operations. Subclasses must override this.

        Parameters
        ----------
        fuzzarray_1 : Fuzzarray
            The first Fuzzarray instance.
        other : Fuzzarray, Fuzznum, scalar, or None
            The second operand.
        tnorm : OperationTNorm
            T-norm configuration.

        Returns
        -------
        object
            Result produced by the concrete implementation (e.g., a new Fuzzarray).

        Raises
        ------
        NotImplementedError
            If not overridden by the subclass.
        """
        raise NotImplementedError(f"Fuzzarray operation '{self.get_operation_name()}' "
                                  f"not implemented for {self.__class__.__name__}")


class OperationScheduler:
    """
    A singleton registry and dispatcher for all fuzzy number operations.

    The ``OperationScheduler`` is the central nervous system for all mathematical
    and logical operations within the AxisFuzzy framework. It acts as a singleton,
    accessible via ``get_registry_operation()``, ensuring a single source of truth
    for how operations are defined, configured, and executed.

    Key Responsibilities:
    ---------------------
    1.  **Operation Registration**:
        It maintains a registry mapping an operation's name (e.g., `'add'`) and
        a specific fuzzy number type (``mtype``, e.g., `'qrofn'`) to a concrete
        :class:`OperationMixin` implementation. This registration is typically
        handled automatically by the ``@register_operation`` decorator. This
        decoupled design allows new operations or support for new `mtype` to be
        added modularly without altering the core framework.

    2.  **Operation Dispatch**:
        When an operation is invoked (e.g., ``fuzznum1 + fuzznum2``), the dispatch
        system queries the scheduler using ``get_operation(op_name, mtype)`` to
        find the correct implementation for the given operands. If no specific
        implementation is found, it signals that the operation is not supported.

    3.  **Global T-Norm Configuration**:
        The scheduler holds the global default T-norm configuration (e.g.,
        'algebraic', 'einstein'). This ensures that all fuzzy operations, by
        default, use a consistent mathematical basis. This can be changed at
        runtime via ``set_t_norm()``, allowing for framework-wide adjustments
        to the underlying logic for experimental purposes.

    4.  **Performance Monitoring**:
        It includes a built-in, thread-safe performance monitor. For every
        operation executed through the system, the scheduler records the
        execution time and updates statistics, such as total call counts and
        average execution times per operation type. This is invaluable for
        debugging, performance tuning, and understanding computational costs.

    Attributes
    ----------
    _operations : dict
        The core registry, structured as ``{op_name: {mtype: operation_instance}}``.
    _default_t_norm_type : str
        The name of the default T-norm (e.g., 'algebraic').
    _default_t_norm_params : dict
        Parameters for the default T-norm (e.g., ``{'p': 2}`` for 'hamacher').
    _performance_stats : dict
        A dictionary holding all performance metrics.
    _stats_lock : threading.Lock
        A lock ensuring thread-safe updates to the performance statistics.

    Examples
    --------
    While direct interaction is rare for end-users, understanding its internal
    role is key for developers.

    .. code-block:: python

        # 1. Access the global scheduler instance
        scheduler = get_registry_operation()

        # 2. Change the global T-norm for all subsequent operations
        scheduler.set_t_norm('hamacher', p=2)

        # 3. Retrieve a specific operation implementation
        # This is what FuzznumStrategy does internally.
        add_op_for_qrofn = scheduler.get_operation('add', 'qrofn')

        # 4. Check which operations are available for a given mtype
        print(scheduler.get_available_ops('qrofn'))
        # Output: ['add', 'sub', 'mul', 'div', 'gt', 'lt', ...]

        # 5. After running some operations, get performance stats
        # stats = scheduler.get_performance_stats(time_unit='us')
        # print(stats['average_times_by_operation_type'])
    """

    def __init__(self):
        """
        Initializes the OperationScheduler.

        Sets up the internal dictionaries for operations and performance statistics,
        and initializes the default T-norm configuration from the global config.
        """
        self._operations: Dict[str, Dict[str, OperationMixin]] = {}
        self._default_t_norm_type: str = 'algebraic'
        self._default_t_norm_params: Dict[str, Any] = {}

        # Global performance statistics
        self._stats_lock = threading.Lock()
        self.reset_performance_stats()

    def set_t_norm(self, t_norm_type: str, **params: Any):
        """
        Change the scheduler's default t-norm configuration.

        Available t-norm types include:
        
        **Archimedean T-Norms:**

        - 'algebraic': Algebraic product t-norm, T(a,b) = a × b
        - 'lukasiewicz': Łukasiewicz t-norm, T(a,b) = max(0, a + b - 1)
        - 'einstein': Einstein t-norm, T(a,b) = (a × b) / (1 + (1-a) × (1-b))
        - 'hamacher': Hamacher t-norm family, requires parameter `hamacher_param` (gamma)
        - 'yager': Yager t-norm family, requires parameter `yager_param` (p)
        - 'schweizer_sklar': Schweizer-Sklar t-norm family, requires parameter `sklar_param` (p)
        - 'dombi': Dombi t-norm family, requires parameter `dombi_param` (p)
        - 'aczel_alsina': Aczel-Alsina t-norm family, requires parameter `aa_param` (p)
        - 'frank': Frank t-norm family, requires parameter `frank_param` (s)
        
        **Non-Archimedean T-Norms:**

        - 'minimum': Minimum t-norm, T(a,b) = min(a,b), the most commonly used standard t-norm
        - 'drastic': Drastic product t-norm, extreme operation under boundary conditions
        - 'nilpotent': Nilpotent t-norm, T(a,b) = min(a,b) when a+b > 1, otherwise 0
        

        Parameters
        ----------
        t_norm_type : str
            Name of the t-norm implementation (e.g., 'algebraic', 'einstein').
        **params : dict
            Implementation-specific parameters (e.g., `p=2` for 'hamacher').

        Examples
        --------
        .. code-block:: python

            scheduler = get_registry_operation()
            scheduler.set_t_norm('hamacher', p=2)
        """
        self._default_t_norm_type = t_norm_type
        self._default_t_norm_params = params

    def get_default_t_norm_config(self) -> tuple[str, Dict[str, Any]]:
        """
        Get the current default t-norm configuration.

        Returns
        -------
        tuple[str, dict]
            A tuple containing (t_norm_type, params).
        """
        return self._default_t_norm_type, self._default_t_norm_params

    def register(self, operation: OperationMixin) -> None:
        """
        Registers an :class:`OperationMixin` instance with the scheduler.

        Operations are registered based on their `op_name` and supported `mtype`s.
        A warning is issued if an operation for a specific `mtype` is already registered.

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
            The name of the operation (e.g., 'add').
        mtype : str
            The membership type (e.g., 'qrofn').

        Returns
        -------
        OperationMixin or None
            The registered operation instance, or None if not found.
        """
        return self._operations.get(op_name, {}).get(mtype)

    def get_available_ops(self, mtype: str) -> List[str]:
        """
        List all operation names available for a given mtype.

        Parameters
        ----------
        mtype : str
            The membership type to query.

        Returns
        -------
        list of str
            A list of available operation names.
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
        are converted to the specified unit and rounded.

        Parameters
        ----------
        time_unit : str, optional
            Unit for displaying time. Supported: 's', 'ms', 'us', 'ns'. Defaults to 'us'.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing performance statistics.

        Raises
        ------
        ValueError
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
        If True (default), instantiate and register the operation immediately.

    Returns
    -------
    type
        The original class object.

    Raises
    ------
    TypeError
        If the decorated object is not a subclass of :class:`OperationMixin`.
    RuntimeError
        If eager instantiation and registration fails.

    Examples
    --------
    .. code-block:: python

        @register_operation
        class MyAddition(OperationMixin):
            def get_operation_name(self): return 'add'
            def get_supported_mtypes(self): return ['my_type']
            # ... implement _execute_*_impl methods ...
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
