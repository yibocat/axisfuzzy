#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/10 13:19
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Optional, Union, List, Any, Tuple, Iterator
import numpy as np

from ..config import get_config
from .fuzznums import Fuzznum
from .t_backend import FuzzarrayBackend
from .registry import get_fuzznum_registry
from .operation import get_operation_registry, OperationMixin
from .triangular import OperationTNorm


class Fuzzarray:
    """
    High-performance fuzzy array using Struct of Arrays (SoA) architecture.
    """

    def __init__(self,
                 data: Optional[Union[np.ndarray, list, tuple, Fuzznum]] = None,
                 backend: Optional[FuzzarrayBackend] = None,
                 mtype: Optional[str] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 **mtype_kwargs):
        """
        Initialize Fuzzarray with either data or existing backend.

        Args:
            data: Input data (list, ndarray, Fuzznum, etc.)
            backend: Pre-constructed FuzzarrayBackend instance
            mtype: Fuzzy number type (e.g., 'qrofn')
            shape: Target shape for data
            **mtype_kwargs: Type-specific parameters (e.g., q for qrofn)
        """
        # Direct backend assignment - fast path
        if backend is not None:
            self._backend = backend
            self._mtype = backend.mtype
            self._mtype_kwargs = backend.mtype_kwargs
            return
        else:
            # Construct from data
            if mtype is None:
                mtype = get_config().DEFAULT_MTYPE
            self._mtype = mtype
            self._mtype_kwargs = mtype_kwargs
            self._backend = self._create_backend_from_data(data, shape)

    def _create_backend_from_data(self, data, shape: Optional[Tuple[int, ...]]) -> FuzzarrayBackend:
        """Create backend from input data"""
        registry = get_fuzznum_registry()
        backend_cls = registry.get_backend(self._mtype)
        if backend_cls is None:
            raise ValueError(f"No backend registered for mtype '{self._mtype}'")

        # Handle different data types
        if data is None and shape is not None:
            # Create empty backend with specified shape
            return backend_cls(shape=shape, **self._mtype_kwargs)

        # For now, use a simplified approach - will improve in later phases
        if isinstance(data, Fuzznum):
            if shape is None:
                shape = ()
            backend = backend_cls(shape=shape, **self._mtype_kwargs)
            if shape == ():
                backend.set_fuzznum_data((), data)
            else:
                # Broadcast single Fuzznum to the shape
                for idx in np.ndindex(shape):
                    backend.set_fuzznum_data(idx, data)
            return backend

        # TODO: Handle list, ndarray of Fuzznum objects
        raise NotImplementedError("Complex data initialization not yet implemented")

    # ========================= Properties =========================

    @property
    def backend(self) -> FuzzarrayBackend:
        """Access to the underlying backend."""
        return self._backend

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the fuzzy array."""
        return self._backend.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._backend.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._backend.size

    @property
    def mtype(self) -> str:
        """fuzzy type of fuzzy numbers."""
        return self._mtype

    @property
    def q(self) -> Optional[int]:
        """Q-rung parameter (if applicable)."""
        return self._mtype_kwargs.get('q')

    @property
    def dtype(self):
        """Data type (always object for compatibility)."""
        return object

    @property
    def T(self):
        # TODO: 实现转置功能
        return None

    # ========================= Indexing & Access =========================

    def __len__(self) -> int:
        """Length of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __getitem__(self, key) -> Union[Fuzznum, 'Fuzzarray']:
        """
        实现索引和切片操作。

        - 标量索引返回 Fuzznum
        - 切片索引返回新的 Fuzzarray
        """

        def _is_scalar_index(k) -> bool:
            if isinstance(k, (int, np.integer)):
                return True
            elif isinstance(k, tuple):
                return all(isinstance(i, (int, np.integer)) for i in k)
            return False

        if _is_scalar_index(key):
            # Scalar index: Returns a single Fuzznum view
            return self._backend.get_fuzznum_view(key)
        else:
            # Slice index: return a new Fuzzarray
            sliced_backend = self._backend.slice_view(key)
            return Fuzzarray(backend=sliced_backend)

    def __setitem__(self, key, value):
        """Set item(s) in the fuzzy array."""
        if isinstance(value, Fuzznum):
            if value.mtype != self._mtype:
                raise ValueError(f"Mtype mismatch: expected '{self._mtype}', got '{value.mtype}'")
            if value.q != self.q:
                raise ValueError(f"Q parameter mismatch: expected q={self.q}, got q={value.q}")
            self._backend.set_fuzznum_data(key, value)
        else:
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray objects, got {type(value)}")

    def __delitem__(self, key):
        # TODO: 实现删除操作
        raise NotImplementedError("Fuzzarray does not support item deletion.")

    def __contains__(self, item) -> bool:
        """检查元素是否在数组中"""
        # TODO: 实现包含检查, 已实现代码是否合理?
        ...
        # if isinstance(item, Fuzznum):
        #     if item.mtype != self._mtype or item.q != self.q:
        #         return False
        #     # 检查是否存在于后端
        #     for idx in np.ndindex(self.shape):
        #         if self._backend.get_fuzznum_view(idx) == item:
        #             return True
        #     return False
        # return False

    def __iter__(self) -> Iterator:
        """Iterate over the fuzzy array."""
        if self.ndim == 0:
            raise TypeError("Iteration over 0-d arrays")
        for i in range(self.shape[0]):
            yield self[i]

    # ========================= Core Operations =========================

    def execute_vectorized_op(self, op_name: str, other=None):
        """
        Execute vectorized operation using registered operations.

        Args:
            op_name: Name of operation (e.g., 'add', 'mul', 'gt')
            other: Second operand (Fuzzarray, Fuzznum, scalar, ndarray)

        Returns:
            Result of operation (Fuzzarray for arithmetic, ndarray for comparison)
        """
        registry = get_operation_registry()
        op = registry.get_operation(op_name, self.mtype)
        if op is None:
            raise NotImplementedError(
                f"Operation '{op_name}' not registered for mtype '{self.mtype}'")

        # Get t-norm configuration
        norm_type, params = registry.get_default_t_norm_config()
        tnorm = OperationTNorm(norm_type=norm_type, q=self.q or 1, **params)

        # --- Dispatcher Logic ---
        # Check if the concrete `OperationMixin` subclass has overridden the
        # `execute_fuzzarray_op` method with a specialized implementation.
        # This is done by comparing the method's function object to the one
        # from the base `OperationMixin` class.
        has_fuzzarray_impl = (
            hasattr(op, '_execute_fuzzarray_op_impl') and
            getattr(op, '_execute_fuzzarray_op_impl').__func__ is not OperationMixin._execute_fuzzarray_op_impl
        )

        if has_fuzzarray_impl:
            # Path 1: Use the specialized implementation. This path typically
            # does not involve `np.vectorize` and might be optimized for speed.
            # Caching is usually handled within the specialized implementation if needed.
            return op.execute_fuzzarray_op(self, other, tnorm)
        else:
            # Path 2: Fallback to a generic element-wise operation.
            # This uses `np.vectorize` to apply the Fuzznum-level operation
            # to each element of the Fuzzarray. Caching is handled at the Fuzznum level.
            return self._fallback_vectorized_op(op, other, tnorm)

    def _fallback_vectorized_op(self, operation, other, tnorm):
        """
        Fallback to element-wise operation when no specialized implementation exists.
        This is a temporary bridge until all operations are optimized.
        """
        # For now, use numpy vectorize as fallback
        # TODO: Optimize this in later phases

        def element_op(idx):
            elem1 = self._backend.get_fuzznum_view(idx)
            if other is None:
                # Unary operation
                result_dict = elem1.get_strategy_instance().execute_operation(
                    operation.get_operation_name(), None)
            else:
                if isinstance(other, Fuzzarray):
                    elem2 = other._backend.get_fuzznum_view(idx)
                    result_dict = elem1.get_strategy_instance().execute_operation(
                        operation.get_operation_name(), elem2.get_strategy_instance())
                elif isinstance(other, Fuzznum):
                    result_dict = elem1.get_strategy_instance().execute_operation(
                        operation.get_operation_name(), other.get_strategy_instance())
                elif isinstance(other, (int, float)):
                    result_dict = elem1.get_strategy_instance().execute_operation(
                        operation.get_operation_name(), other)
                else:
                    raise TypeError(f"Unsupported operand type: {type(other)}")

            return result_dict

        results = []
        for idx in np.ndindex(self.shape):
            results.append(element_op(idx))

        # Check if this is a comparison operation (returns bool)
        if operation.get_operation_name() in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
            bool_results = np.array([r.get('result', False) for r in results])
            return bool_results.reshape(self.shape)
        else:
            # Create new Fuzzarray from results
            # This is inefficient but works as fallback
            new_backend = self._backend.copy()
            from .fuzznums import Fuzznum
            for idx, result in zip(np.ndindex(self.shape), results):
                new_fuzznum = Fuzznum(mtype=self.mtype, **self._mtype_kwargs).create(**result)
                new_backend.set_fuzznum_data(idx, new_fuzznum)
            return Fuzzarray(backend=new_backend)

    # ========================= Operator Overloads =========================
    # These methods overload standard Python operators to enable intuitive
    # arithmetic and comparison operations directly on Fuzzarray objects.
    # They delegate the actual operation logic to the `dispatcher.operate` function,
    # which handles type dispatching and calls `execute_vectorized_op` internally.

    def __add__(self, other: Any) -> 'Fuzzarray':
        """Implements the addition operator (+)."""
        from .dispatcher import operate
        return operate('add', self, other)

    def __radd__(self, other: Any) -> 'Fuzzarray':
        """Implements the reverse addition operator (+)."""
        from .dispatcher import operate
        return operate('add', self, other)

    def __sub__(self, other: Any) -> 'Fuzzarray':
        """Implements the subtraction operator (-)."""
        from .dispatcher import operate
        return operate('sub', self, other)

    def __mul__(self, other: Any) -> 'Fuzzarray':
        """Implements the multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __rmul__(self, other: Any) -> 'Fuzzarray':
        """Implements the reverse multiplication operator (*)."""
        from .dispatcher import operate
        return operate('mul', self, other)

    def __truediv__(self, other: Any) -> 'Fuzzarray':
        """Implements the true division operator (/)."""
        from .dispatcher import operate
        return operate('div', self, other)

    def __pow__(self, power: Any, modulo: Optional[Any] = None) -> 'Fuzzarray':
        """Implements the power operator (**)."""
        from .dispatcher import operate
        return operate('pow', self, power)

    def __gt__(self, other: Any) -> np.ndarray:
        """Implements the greater than operator (>). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('gt', self, other)

    def __lt__(self, other: Any) -> np.ndarray:
        """Implements the less than operator (<). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('lt', self, other)

    def __ge__(self, other: Any) -> np.ndarray:
        """Implements the greater than or equal to operator (>=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('ge', self, other)

    def __le__(self, other: Any) -> np.ndarray:
        """Implements the less than or equal to operator (<=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('le', self, other)

    def __eq__(self, other: Any) -> np.ndarray:
        """Implements the equality operator (==). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('eq', self, other)

    def __ne__(self, other: Any) -> np.ndarray:
        """Implements the inequality operator (!=). Returns a boolean NumPy array."""
        from .dispatcher import operate
        return operate('ne', self, other)

    def __and__(self, other):
        """Overloads the and operator (&).
            intersection operation.
        """
        from .dispatcher import operate
        return operate('intersection', self, other)

    def __or__(self, other):
        from .dispatcher import operate
        return operate('union', self, other)

    def __invert__(self):
        """Overloads the invert operator (~).
            Complement operation.
        """
        from .dispatcher import operate
        return operate('complement', self, None)

    def __lshift__(self, other):
        """Overloads the left shift operator (<<).
            Denotes the left implication operation: self <- other
        """
        from .dispatcher import operate
        return operate('implication', other, self)

    def __rshift__(self, other):
        """Overloads the shift operator (>>).
            Denotes the right implication operation: self -> other
        """
        from .dispatcher import operate
        return operate('implication', self, other)

    def __xor__(self, other):
        """Overloads the xor operator (^).
            Denotes the symmetric difference operation.
        """
        from .dispatcher import operate
        return operate('symdiff', self, other)

    def equivalent(self, other):
        """
        Calculate the equivalence level between two fuzzy numbers

        Corresponding to the "if and only if" operation in classical logic,
        it represents the degree to which two fuzzy propositions are
        equivalent to each other.
        """
        from .dispatcher import operate
        return operate('equivalence', self, other)

    def __matmul__(self, other):
        # TODO: 基于后端的矩阵乘法有点复杂,后续实现
        pass

    # ========================= Utility Methods =========================

    def copy(self) -> 'Fuzzarray':
        """Create a deep copy"""
        copied_backend = self._backend.copy()
        return Fuzzarray(backend=copied_backend)

    def __repr__(self) -> str:
        """字符串表示"""
        if self.size == 0:
            return f"Fuzzarray([], mtype='{self.mtype}', shape={self.shape})"

        # 对于小数组，展示部分内容
        if self.size <= 15:
            elements = []
            for idx in np.ndindex(self.shape):
                fuzznum = self._backend.get_fuzznum_view(idx)
                elements.append(str(fuzznum))
            content = ', '.join(elements)
        else:
            content = f"... {self.size} elements ..."

        return f"Fuzzarray([{content}], mtype='{self.mtype}', q={self.q}, shape={self.shape})"

    def __str__(self) -> str:
        """用户友好的字符串表示"""
        return self.__repr__()

    # TODO: 实现特殊方法
    def __bool__(self) -> bool: ...
    def __format__(self, format_spec: str) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __setstate__(self, state: Any): ...


# ================================= 工厂函数 =================================

def fuzzarray(data,
              mtype: Optional[str] = None,
              shape: Optional[Tuple[int, ...]] = None,
              copy: bool = True,
              **mtype_kwargs) -> Fuzzarray:
    """
    Factory function to create Fuzzarray instances.

    Args:
        data: Input data
        mtype: Membership type
        shape: Target shape
        copy: Whether to copy data (for compatibility)
        **mtype_kwargs: Type-specific parameters

    Returns:
        New Fuzzarray instance
    """
    return Fuzzarray(data=data, mtype=mtype, shape=shape, **mtype_kwargs)
