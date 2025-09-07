#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
"""
High-level Fuzznum container built on a Struct-of-Arrays (SoA) backend.

This module exposes the :class:`Fuzzarray` class, which provides ndarray-like API
semantics for collections of fuzzy numbers while delegating storage and
bulk computation to a specialized :class:`FuzzarrayBackend` implementation.

Overview
--------
- Fuzzarray is the main container for high-dimensional fuzzy numbers.
- It supports efficient batch operations and slicing, similar to numpy arrays.
- All storage and vectorized computation is delegated to a backend, which is
  selected based on the fuzzy number type (`mtype`).

Notes
-----
- The backend system enables high performance by avoiding per-element Python object overhead.
- Operator overloading is supported for elementwise arithmetic and comparison.
- Specialized vectorized operations can be registered for each fuzzy type.

Examples
--------
.. code-block:: python

    from axisfuzzy.core.fuzznums import fuzznum
    from axisfuzzy.core.fuzzarray import fuzzarray

    # Create a 1D Fuzzarray by broadcasting a Fuzznum
    a = fuzznum(mtype='qrofn', q=2, md=0.6, nmd=0.3)
    arr = fuzzarray(a, shape=(3,))

    # Create from a list of Fuzznum
    arr2 = fuzzarray([fuzznum(md=0.1, nmd=0.2), fuzznum(md=0.2, nmd=0.3), fuzznum(md=0.3, nmd=0.4)])

    # Elementwise arithmetic
    result = arr + arr2

    # Comparison yields boolean ndarray
    mask = arr > arr2
    print(mask)
"""

from typing import Optional, Union, Any, Tuple, Iterator, Dict
import numpy as np

from ..config import get_config
from ..utils import deprecated

from .fuzznums import Fuzznum
from .backend import FuzzarrayBackend
from .registry import get_registry_fuzztype


class Fuzzarray:
    """
    High-performance fuzzy array using Struct of Arrays (SoA) architecture.

    The Fuzzarray provides an ndarray-like interface for collections of fuzzy
    numbers while delegating memory layout and vectorized computations to a
    backend class (see :class:`FuzzarrayBackend`).

    Parameters
    ----------
    data : array-like, Fuzznum or None, optional
        Input content used to initialize the array. Accepted types:
        - None : create an empty backend (requires ``shape``).
        - Fuzznum : broadcast a single fuzznum to fill the target shape.
        - list/tuple/numpy.ndarray of Fuzznum : element-wise initialization.
    backend : FuzzarrayBackend, optional
        Pre-constructed backend instance. When provided, ``data`` is ignored.
    mtype : str, optional
        Membership type name (e.g. ``'qrofn'``). If omitted, the default from
        configuration is used.
    q : int, optional
        q-rung parameter for q-rung based mtypes. If omitted, default is used.
    shape : tuple of int, optional
        Target logical shape for the array (required when ``data`` is None).
    **kwargs
        Additional backend/mtype-specific keyword arguments.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.core.fuzznums import fuzznum
        from axisfuzzy.core.fuzzarray import fuzzarray

        # Create a 2D Fuzzarray from a list of lists
        arr = fuzzarray([[fuzznum(md=0.1, nmd=0.2), fuzznum(md=0.2, nmd=0.3)],
                         [fuzznum(md=0.3, nmd=0.4), fuzznum(md=0.4, nmd=0.5)]])

        # Indexing returns Fuzznum or Fuzzarray
        print(arr[0, 1])  # Fuzznum
        print(arr[0:1])   # Fuzzarray

        # Elementwise operations
        arr2 = arr + 0.1  # Broadcasting supported if operation registered

        # Copy
        arr3 = arr.copy()
    """

    def __init__(self,
                 data: Optional[Union[np.ndarray, list, tuple, Fuzznum, "Fuzzarray"]] = None,
                 backend: Optional[FuzzarrayBackend] = None,
                 mtype: Optional[str] = None,
                 q: Optional[int] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        """
        Initialize Fuzzarray with either data or existing backend.

        Parameters
        ----------
        data : array-like or Fuzznum or None, optional
            Input data to populate the Fuzzarray.
        backend : FuzzarrayBackend, optional
            If provided, used directly as the storage backend.
        mtype : str, optional
            Membership type string.
        q : int, optional
            q-rung parameter.
        shape : tuple of int, optional
            Desired shape when constructing from no data or a scalar.
        **kwargs : dict
            Extra parameters forwarded to backend constructor.
        """
        # This attribute will hold a reference to the original array if this is a transpose
        self._transposed_of: Optional['Fuzzarray'] = None

        # Direct backend assignment - fast path
        if backend is not None:
            self._backend = backend
            self._mtype = backend.mtype
            self._q = backend.q
            self._kwargs = backend.kwargs
            return
        else:
            backend, mtype, q = self._build_backend_from_data(data, shape, mtype, q, **kwargs)
            self._backend = backend
            self._mtype = mtype
            self._q = q
            self._kwargs = kwargs

            # # Construct from data
            # self._mtype = get_config().DEFAULT_MTYPE if mtype is None else mtype
            # self._q = q if q is not None else get_config().DEFAULT_Q
            # self._kwargs = kwargs
            # self._backend = self._create_backend_from_data(data, shape)

    def _build_backend_from_data(self,
                                 data: Optional[Union[np.ndarray, list, tuple, Fuzznum, "Fuzzarray"]],
                                 shape: Optional[Tuple[int, ...]],
                                 mtype: Optional[str],
                                 q: Optional[int],
                                 **kwargs) -> Tuple[FuzzarrayBackend, str, int]:

        registry = get_registry_fuzztype()

        if mtype is None:
            if data is not None:
                if isinstance(data, Union[Fuzznum, Fuzzarray]):
                    mtype = data.mtype
                elif isinstance(data, Union[np.ndarray, list, tuple]):
                    data_array = np.asarray(data)
                    if data_array.size > 0:
                        mtype = data_array.flatten()[0].mtype
            else:
                mtype = get_config().DEFAULT_MTYPE
        else:
            mtype = mtype.lower()

        if q is None:
            if data is not None:
                if isinstance(data, Union[Fuzznum, Fuzzarray]):
                    q = data.q
                elif isinstance(data, Union[np.ndarray, list, tuple]):
                    data_array = np.asarray(data)
                    if data_array.size > 0:
                        q = data_array.flatten()[0].q
            else:
                q = get_config().DEFAULT_Q
        else:
            q = int(q)

        backend_cls = registry.get_backend(mtype)
        if backend_cls is None:
            raise ValueError(f"No backend registered for mtype '{mtype}'")

        # Case 1: data is None → 默认空数组
        if data is None:
            shape = shape or (0,)
            backend = backend_cls(shape=shape, q=q, **kwargs)
            return backend, mtype, q

        # Case 2: data is already a Fuzzarray
        if isinstance(data, Fuzzarray):
            # copy backend
            backend = data.backend.copy()
            return backend, data.mtype, data.q

        # Case 3: Single Fuzznum
        if isinstance(data, Fuzznum):
            shape = shape or ()   # default scalar
            backend = backend_cls(shape=shape, q=data.q, **kwargs)
            for idx in np.ndindex(shape):
                backend.set_fuzznum_data(idx, data)
            return backend, mtype, data.q

        # Case 3: list/tuple/ndarray of Fuzznum
        if isinstance(data, (list, tuple, np.ndarray)):
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=object)

            # 空 → shape 汲取自 data.shape，默认为 (0,)
            if data.size == 0:
                shape = shape or data.shape
                backend = backend_cls(shape=shape, q=q, **kwargs)
                return backend, mtype, q

            shape = shape or data.shape
            if data.shape != shape:
                try:
                    data = data.reshape(shape)
                except ValueError:
                    raise ValueError(f"Cannot reshape array of size {data.size} into shape {shape}")

            flat = data.flatten()
            if not all(isinstance(item, Fuzznum) for item in flat):
                raise TypeError(f"All elements must be Fuzznum, found {type(flat[0])}")

            q = flat[0].q
            backend = backend_cls(shape=shape, q=q, **kwargs)
            for idx in np.ndindex(shape):
                backend.set_fuzznum_data(idx, data[idx])
            return backend, mtype, q

        # Unsupported
        raise TypeError(f"Unsupported data type for Fuzzarray creation: {type(data)}")

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
        return self._q

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Additional parameters for the fuzzy type."""
        return self._kwargs

    @property
    def dtype(self):
        """Data type (always object for compatibility)."""
        return object

    # ========================= Indexing & Access =========================

    def __len__(self) -> int:
        """Length of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __getitem__(self, key) -> Union['Fuzznum', 'Fuzzarray']:
        """
        Index or slice the Fuzzarray.

        - Scalar indexing returns a lightweight Fuzznum view.
        - Slice / ndarray-style indexing returns a new Fuzzarray (view when backend supports it).

        Parameters
        ----------
        key : int, tuple, slice or other valid numpy-style index
            Indexing key.

        Returns
        -------
        Fuzznum | Fuzzarray
            Single-element view or a new Fuzzarray for slices.
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
        """
        Assign a Fuzznum to specified location(s).

        Parameters
        ----------
        key : index
            Target location to set.
        value : Fuzznum
            Value to assign.

        Raises
        ------
        TypeError
            If ``value`` is not a Fuzznum.
        ValueError
            If ``value`` has mismatched ``mtype`` or ``q``.
        """
        if isinstance(value, Fuzznum):
            if value.mtype != self._mtype:
                raise ValueError(f"Mtype mismatch: expected '{self._mtype}', got '{value.mtype}'")
            if value.q != self.q:
                raise ValueError(f"Q parameter mismatch: expected q={self.q}, got q={value.q}")
            self._backend.set_fuzznum_data(key, value)
        else:
            raise TypeError(f"Can only assign Fuzznum or Fuzzarray objects, got {type(value)}")

    def __delitem__(self, key):
        raise NotImplementedError("Fuzzarray does not support item deletion.")

    def __contains__(self, item: Any) -> bool:
        """
        Test membership of a Fuzznum in the array.

        Parameters
        ----------
        item : Any
            Object to test for containment.

        Returns
        -------
        bool
            True if a matching element exists in the array; False otherwise.

        Notes
        -----
        - Only objects that are instances of :class:`Fuzznum` with matching
          ``mtype`` and ``q`` are considered.
        """
        if not isinstance(item, Fuzznum):
            return False
        if item.mtype != self._mtype or item.q != self.q:
            return False

        for idx in np.ndindex(self.shape):
            fuzznum_view = self._backend.get_fuzznum_view(idx)
            if fuzznum_view == item:
                return True
        return False

    def __iter__(self) -> Iterator:
        """Iterate over the fuzzy array."""
        if self.ndim == 0:
            raise TypeError("Fuzzarray iteration is not supported for zero-dimensional arrays.")
        for i in range(self.shape[0]):
            yield self[i]

    # ========================= Core Operations =========================

    def execute_vectorized_op(self, op_name: str, other=None):
        """
        Execute a vectorized operation using the registered operation handlers.

        The method queries the global operation registry for the named operation
        implementation for this Fuzzarray's ``mtype``. If a backend/vectorized
        specialization exists it is used; otherwise a fallback element-wise
        path is taken.

        Parameters
        ----------
        op_name : str
            Operation name (e.g. ``'add'``, ``'mul'``, ``'gt'``).
        other : Fuzzarray, Fuzznum, scalar, ndarray or None, optional
            Second operand.

        Returns
        -------
        Fuzzarray or numpy.ndarray
            Result of the vectorized operation. Comparison operations yield
            boolean numpy arrays; arithmetic operations yield Fuzzarray.
        """
        from .operation import get_registry_operation, OperationMixin
        from .triangular import OperationTNorm

        registry = get_registry_operation()
        op = registry.get_operation(op_name, self.mtype)
        if op is None:
            raise NotImplementedError(
                f"Operation '{op_name}' not registered for mtype '{self.mtype}'")

        # Get t-norm configuration
        norm_type, params = registry.get_default_t_norm_config()
        tnorm = OperationTNorm(norm_type=norm_type, q=self.q or get_config().DEFAULT_Q, **params)

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
        Generic element-wise fallback for operations without specialized vectorized implementations.

        This method applies the per-element Fuzznum operation across all
        indices and reconstructs a result container. It is slower than a
        specialized backend implementation and intended as a portability fallback.

        Parameters
        ----------
        operation : OperationMixin
            Operation handler object obtained from the registry.
        other : Fuzzarray, Fuzznum, scalar or None
            Second operand for the operation.
        tnorm : OperationTNorm
            T-norm configuration passed to per-element executions.

        Returns
        -------
        Fuzzarray or numpy.ndarray
            Resulting container (Fuzzarray for arithmetic, ndarray of bool for comparisons).
        """
        # For now, use numpy vectorize as fallback

        def element_op(index):
            elem1 = self._backend.get_fuzznum_view(index)
            if other is None:
                # Unary operation
                result_dict = elem1.get_strategy_instance().execute_operation(
                    operation.get_operation_name(), None)
            else:
                if isinstance(other, Fuzzarray):
                    elem2 = other._backend.get_fuzznum_view(index)
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
            for idx, result in zip(np.ndindex(self.shape), results):
                new_fuzznum = Fuzznum(mtype=self.mtype, **self._kwargs).create(**result)
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
        """Implements matrix multiplication (@)."""
        from .dispatcher import operate
        return operate('matmul', self, other)

    # ========================= Utility Methods =========================

    def copy(self) -> 'Fuzzarray':
        """Create a deep copy"""
        # The copy method in the backend already creates new data arrays.
        copied_backend = self._backend.copy()
        # The new Fuzzarray is a standalone object, so it has no _transposed_of reference.
        return Fuzzarray(backend=copied_backend)

    def __repr__(self) -> str:
        if self.size == 0:
            return f"Fuzzarray([], mtype='{self.mtype}', q={self.q}, shape={self.shape})"
        formatted = self._backend.format_elements()
        # Key: prefix='Fuzzarray(' makes the continued line indentation align with the desired left side.
        array_str = np.array2string(
            formatted,
            separator=' ',
            formatter={'object': lambda x: x},  # type: ignore
            prefix='Fuzzarray(',
            max_line_width=100
        )
        return f"Fuzzarray({array_str}, mtype='{self.mtype}', q={self.q}, shape={self.shape})"

    def __str__(self) -> str:
        if self.size == 0:
            return "[]"
        formatted = self._backend.format_elements()
        return np.array2string(
            formatted,
            separator=' ',
            formatter={'object': lambda x: x},  # type: ignore
            prefix='',
            max_line_width=90
        )

    def __format__(self, format_spec: str = "") -> Any:
        formatted = self._backend.format_elements(format_spec)
        return np.array2string(
            formatted,
            separator=' ',
            formatter={'object': lambda x: x},  # type: ignore
            prefix='',
            max_line_width=80
        )

    def __bool__(self) -> bool:
        if self.size > 1:
            raise ValueError(
                "The truth value of a Fuzzarray with more than one element is ambiguous. "
                "Use .any() or .all()"
            )
        if self.size == 1:
            # For a single element array, its truthiness is determined by the Fuzznum itself.
            # Fuzznum.__bool__ is True, so this will be True.
            return bool(self[0])
        # For a 0-element array
        return False

    def __getstate__(self) -> Dict[str, Any]:
        """For pickling"""
        return {
            'mtype': self.mtype,
            'q': self.q,
            'kwargs': self.kwargs,
            'backend_state': self.backend.__dict__  # A simple way, might need refinement
        }

    def __setstate__(self, state: Dict[str, Any]):
        """For unpickling"""
        self._mtype = state['mtype']
        self._q = state['q']
        self._kwargs = state['kwargs']

        registry = get_registry_fuzztype()
        backend_cls = registry.get_backend(self._mtype)

        # Reconstruct backend from its state
        backend_state = state['backend_state']
        shape = backend_state['shape']
        self._backend = backend_cls(shape=shape, q=self._q, **self._kwargs)

        # Restore component arrays
        for key, value in backend_state.items():
            if isinstance(value, np.ndarray):
                setattr(self._backend, key, value)


# ================================= Factory function =================================

@deprecated(message="Please use 'fuzzyset' instead.")
def fuzzarray(data,
              mtype: Optional[str] = None,
              shape: Optional[Tuple[int, ...]] = None,
              copy: bool = True,
              **mtype_kwargs) -> Fuzzarray:
    """
    Factory function to create Fuzzarray instances.

    Parameters
    ----------
    data : array-like or Fuzznum or None
        Input data to populate the returned Fuzzarray.
    mtype : str, optional
        Fuzztype name.
    shape : tuple of int, optional
        Desired shape when constructing from scalars or empty data.
    copy : bool, optional
        Reserved for API compatibility; current implementation forwards data
        to Fuzzarray constructor which controls copying semantics.
    **mtype_kwargs : dict
        Additional mtype-specific parameters forwarded to Fuzzarray.

    Returns
    -------
    Fuzzarray
        New Fuzzarray instance constructed from the provided inputs.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy.core.fuzznums import fuzznum
        from axisfuzzy.core.fuzzarray import fuzzarray

        arr = fuzzarray([fuzznum(md=0.1, nmd=0.2), fuzznum(md=0.2, nmd=0.3)])
        print(arr)
    """
    return Fuzzarray(data=data, mtype=mtype, shape=shape, **mtype_kwargs)
