#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Implementation factory for AxisFuzzy mixin system structural operations.

This module provides the core implementation functions for mtype-agnostic
structural and container operations on :class:`Fuzznum` and :class:`Fuzzarray`
objects. These functions work directly with the SoA (Struct-of-Arrays) backend
architecture to provide efficient NumPy-like functionality across all fuzzy
number types.

Architecture
------------
The factory follows a direct implementation approach where each function:

1. **Input Validation**: Checks input types and converts :class:`Fuzznum` to
   :class:`Fuzzarray` when necessary for consistent processing.

2. **Backend Interaction**: Extracts component arrays from the SoA backend,
   performs NumPy operations on each component, then reconstructs the backend.

3. **Result Construction**: Creates new :class:`Fuzzarray` instances with the
   modified backend, preserving mtype and fuzzy-specific metadata.

4. **Type Consistency**: Maintains proper return types (:class:`Fuzznum` vs
   :class:`Fuzzarray`) based on the mathematical properties of each operation.

Core Implementation Functions
-----------------------------
This module implements the following categories of structural operations:

**Shape Manipulation Functions**:
    - ``_reshape_factory``: Changes array shape without modifying data, supports automatic dimension inference (-1)
    - ``_flatten_factory``: Collapses multi-dimensional arrays into 1D while preserving element order
    - ``_squeeze_factory``: Removes dimensions of size 1, with optional axis specification
    - ``_ravel_factory``: Returns contiguous flattened view (currently implemented as copy, future optimization possible)

**Transformation Functions**:
    - ``_transpose_factory``: Permutes array dimensions according to specified axes order, supports back-reference optimization
    - ``_broadcast_to_factory``: Broadcasts arrays to specified shapes following NumPy broadcasting rules

**Data Access Functions**:
    - ``_copy_factory``: Creates deep independent copies of fuzzy objects with full data duplication
    - ``_item_factory``: Extracts scalar :class:`Fuzznum` elements from arrays, supports multi-dimensional indexing

**Container Operations**:
    - ``_concat_factory``: Joins multiple arrays along existing axes with shape and type compatibility checking
    - ``_stack_factory``: Combines arrays along new axes with strict shape matching requirements
    - ``_append_factory``: Flexible append operation supporting various input types and in-place modification
    - ``_pop_factory``: Removes and returns elements from 1D arrays with optional in-place operation

**Boolean Testing Functions**:
    - ``_any_factory``: Tests if arrays contain any elements (always True for non-empty valid fuzzy arrays)
    - ``_all_factory``: Tests if all elements are truthy (always True for valid fuzzy arrays)

**Future Implementation Placeholders**:
    - ``_sort_factory``: Sorting operations (not yet implemented)
    - ``_argsort_factory``: Argument sorting (not yet implemented)
    - ``_argmax_factory``: Argument maximum (not yet implemented)
    - ``_argmix_factory``: Argument minimum (not yet implemented)

Backend Integration
-------------------
All functions work with the SoA backend architecture:

1. **Component Extraction**: Uses ``backend.get_component_arrays()`` to access
   individual fuzzy number components (e.g., membership, non-membership values).

2. **Parallel Processing**: Applies the same NumPy operation to all components
   simultaneously, maintaining data alignment and type consistency.

3. **Backend Reconstruction**: Uses ``backend_cls.from_arrays()`` to create new
   backends from modified components, preserving mtype-specific parameters.

4. **Metadata Preservation**: Maintains q-values, mtype information, and other
   fuzzy-specific metadata through the transformation process.

Error Handling
--------------
Functions provide comprehensive error handling for:

- **Type Validation**: Ensures inputs are valid :class:`Fuzznum` or :class:`Fuzzarray` objects
- **Shape Compatibility**: Validates shapes for operations like concatenation, broadcasting, and stacking
- **Index Bounds**: Checks array bounds for element access and removal operations
- **Mathematical Constraints**: Enforces fuzzy number constraints and backend limitations
- **Empty Array Handling**: Special cases for zero-size arrays and edge conditions

Performance Considerations
--------------------------
The factory functions are optimized for:

- **Vectorized Operations**: Leverages NumPy's optimized C implementations through component arrays
- **Memory Efficiency**: Minimizes object creation and copying where possible
- **Backend Reuse**: Reuses backend classes and metadata to avoid unnecessary allocations
- **Lazy Evaluation**: Some operations (like transpose) include optimizations for repeated use

Type Safety
-----------
Functions maintain strict type consistency:

- **Input Promotion**: :class:`Fuzznum` inputs are promoted to single-element :class:`Fuzzarray` when needed
- **Return Type Logic**: Operations that may result in scalars return :class:`Fuzznum`, others return :class:`Fuzzarray`
- **Metadata Consistency**: All type-specific metadata (mtype, q-values) is preserved through operations
- **Backend Compatibility**: Ensures all operations preserve backend-specific constraints and properties

Notes
-----
- Functions are designed to be called from registration wrappers in :mod:`axisfuzzy.mixin.register`
- All operations preserve the original object when creating copies or views
- Backend-specific optimizations may be added in future versions without changing the API
- Thread safety depends on the underlying NumPy operations and backend implementations

See Also
--------
axisfuzzy.mixin.register : Registration layer that exposes these functions as methods
axisfuzzy.mixin.registry : Infrastructure for dynamic injection and registration
axisfuzzy.core.backend : SoA backend architecture used by all operations
axisfuzzy.core.fuzzarray : Primary data structure for array operations
axisfuzzy.core.fuzznums : Scalar fuzzy number data structure

Examples
--------
Factory functions are typically not called directly, but through registered methods:

.. code-block:: python

    from axisfuzzy.mixin.factory import _reshape_factory
    from axisfuzzy.core import fuzzarray, fuzznum

    # Direct factory call (not typical usage)
    a = fuzznum((0.6, 0.3), mtype='qrofn')
    arr = fuzzarray([a, a, a, a])
    reshaped = _reshape_factory(arr, 2, 2)

    # Normal usage through registered methods
    reshaped = arr.reshape(2, 2)  # Calls _reshape_factory internally

Backend Compatibility
---------------------
All factory functions work with any backend that implements:

- ``get_component_arrays()``: Returns list of NumPy arrays for each fuzzy component
- ``from_arrays(cls, *arrays, q, **kwargs)``: Class method to reconstruct from component arrays
- Proper mtype and parameter preservation through the reconstruction process

References
----------
- NumPy documentation for array manipulation functions
- AxisFuzzy backend architecture documentation
- SoA (Struct-of-Arrays) design patterns for high-performance computing
"""

from typing import Union, Tuple, Optional, List

import numpy as np

from ..core import Fuzzarray, Fuzznum


# ========================= Core Structural Operations =========================

def _reshape_factory(obj: Union[Fuzznum, Fuzzarray], *shape: int) -> Fuzzarray:
    """
    Gives a new shape to an array without changing its data.
    Works with the SoA backend infrastructure.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if isinstance(obj, Fuzznum):
        # Create a scalar Fuzzarray first, then reshape
        arr = Fuzzarray(data=obj, shape=())
        return _reshape_factory(arr, *shape)

    # Get component arrays from backend
    components = obj.backend.get_component_arrays()

    # Reshape each component array
    try:
        reshaped_components = [comp.reshape(shape) for comp in components]
    except ValueError as e:
        raise ValueError(f"Cannot reshape array of size {obj.size} into shape {shape}") from e

    # Create new backend from reshaped components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*reshaped_components, q=obj.q)

    return Fuzzarray(backend=new_backend)


def _flatten_factory(obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    """
    Return a copy of the array collapsed into one dimension.
    """
    if isinstance(obj, Fuzznum):
        return Fuzzarray(data=obj, shape=(1,))

    # Get component arrays and flatten them
    components = obj.backend.get_component_arrays()
    flattened_components = [comp.flatten() for comp in components]

    # Create new backend from flattened components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*flattened_components, q=obj.q)

    return Fuzzarray(backend=new_backend)


def _squeeze_factory(obj: Union[Fuzznum, Fuzzarray],
                     axis: Union[int, Tuple[int, ...], None] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    Remove single-dimensional entries from the shape of an array.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    # Get component arrays
    components = obj.backend.get_component_arrays()
    squeezed_components = [comp.squeeze(axis=axis) for comp in components]

    # Check if result is scalar
    if squeezed_components[0].ndim == 0:
        # Return a Fuzznum for scalar result
        # We need to extract the scalar values and create a Fuzznum
        # This requires knowledge of the mtype, so we'll use the backend
        flattened_obj = _flatten_factory(obj)
        return flattened_obj[0]  # Get the first (and only) element
    else:
        # Create new backend from squeezed components
        backend_cls = obj.backend.__class__
        new_backend = backend_cls.from_arrays(*squeezed_components, q=obj.q)
        return Fuzzarray(backend=new_backend)


def _copy_factory(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Returns a deep copy of the fuzzy object.

    This method is mainly used to implement top-level functions,
    as it copies instance methods to achieve this.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()
    elif isinstance(obj, Fuzzarray):
        return obj.copy()
    else:
        raise TypeError(f"Unsupported type for copy: {type(obj)}")


def _ravel_factory(obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    """
    Return a contiguous flattened array.
    """
    if isinstance(obj, Fuzznum):
        return Fuzzarray(data=obj, shape=(1,))

    # Ravel is essentially reshape(-1)
    return _reshape_factory(obj, -1)


def _transpose_factory(obj: Union[Fuzzarray, Fuzznum], *axes) -> Union[Fuzzarray, Fuzznum]:
    """
    Returns a view of the fuzzy object with axes transposed.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    if not isinstance(obj, Fuzzarray):
        raise TypeError(f"Unsupported type for transpose: {type(obj)}")

    # Check if we are transposing an already transposed array
    if obj._transposed_of is not None and not axes:
        return obj._transposed_of

    # Handle different ways axes can be passed
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = axes[0]

    # Get component arrays and transpose them
    components = obj.backend.get_component_arrays()
    transposed_components = [comp.transpose(axes) for comp in components]

    # Create new backend from transposed components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*transposed_components, q=obj.q)

    new_array = Fuzzarray(backend=new_backend)

    # If it's a simple transpose (no custom axes), set the back-reference
    if axes is None:
        new_array._transposed_of = obj

    return new_array


def _broadcast_to_factory(obj: Union[Fuzzarray, Fuzznum], *shape: int) -> Fuzzarray:
    """
    Broadcasts the fuzzy object to a new shape.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)

    if isinstance(obj, Fuzznum):
        # Create a scalar array first
        arr = Fuzzarray(data=obj, shape=())
        return _broadcast_to_factory(arr, target_shape)

    # Get component arrays and broadcast them
    components = obj.backend.get_component_arrays()

    try:
        broadcasted_components = [np.broadcast_to(comp, target_shape) for comp in components]
    except ValueError as e:
        raise ValueError(f"Cannot broadcast object with shape {obj.shape} to shape {target_shape}: {e}")

    # Create new backend from broadcasted components
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*broadcasted_components, q=obj.q)
    return Fuzzarray(backend=new_backend)


def _item_factory(obj: Union[Fuzzarray, Fuzznum], *args) -> Fuzznum:
    """
    Returns the scalar item of the fuzzy object.
    """
    if isinstance(obj, Fuzznum):
        return obj.copy()

    if isinstance(obj, Fuzzarray):
        if obj.size == 1 and not args:
            # Use backend to get the single element
            return obj.backend.get_fuzznum_view(0 if obj.ndim == 1 else ())
        elif args:
            # Get element at specified index
            return obj.backend.get_fuzznum_view(args[0])
        else:
            raise ValueError("can only convert an array of size 1 to a Python scalar")

    raise TypeError(f"Unsupported type for item() method: {type(obj)}")


def _any_factory(obj: Fuzzarray) -> bool:
    """
    Test whether any array element evaluates to True.
    For a Fuzzarray, this returns True if the array is not empty.
    """
    return obj.size > 0


def _all_factory(obj: Fuzzarray) -> bool:
    """
    Test whether all array elements evaluate to True.
    For a Fuzzarray, this is always True as all valid Fuzznum instances are truthy.
    """
    return True


# ========================= Container Operations =========================

def _concat_factory(obj: Fuzzarray, *others: Fuzzarray, axis: int = 0) -> Fuzzarray:
    """
    Join a sequence of Fuzzarrays along an existing axis.
    """
    if not isinstance(obj, Fuzzarray):
        raise TypeError("concat: first argument must be Fuzzarray")

    all_arrays = [obj] + list(others)

    # Filter out empty arrays
    effective_arrays = [arr for arr in all_arrays if arr.size > 0]
    if not effective_arrays:
        # Return empty array with same mtype
        return Fuzzarray(data=None, shape=(0,), mtype=obj.mtype, q=obj.q)

    # Check compatibility
    ref_array = effective_arrays[0]
    for arr in effective_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"concat: all arguments must be Fuzzarray, got {type(arr)}")
        if arr.mtype != ref_array.mtype or arr.q != ref_array.q:
            raise ValueError("concat: all Fuzzarrays must have the same mtype and parameters")

    # Get component arrays from all arrays
    component_lists = [[] for _ in ref_array.backend.get_component_arrays()]
    for arr in effective_arrays:
        components = arr.backend.get_component_arrays()
        for i, comp in enumerate(components):
            component_lists[i].append(comp)

    # Concatenate each component list
    new_components = [np.concatenate(comp_list, axis=axis) for comp_list in component_lists]

    # Create new backend
    backend_cls = ref_array.backend.__class__
    new_backend = backend_cls.from_arrays(*new_components, q=ref_array.q)

    return Fuzzarray(backend=new_backend)


def _stack_factory(obj: Fuzzarray, *others: Fuzzarray, axis: int = 0) -> Fuzzarray:
    """
    Stack Fuzzarrays along a new axis.
    """
    if not isinstance(obj, Fuzzarray):
        raise TypeError("stack: first argument must be Fuzzarray")

    all_arrays = [obj] + list(others)

    # Check compatibility
    ref_mtype = obj.mtype
    ref_q = obj.q
    ref_shape = obj.shape

    for arr in all_arrays:
        if not isinstance(arr, Fuzzarray):
            raise TypeError(f"stack: all arguments must be Fuzzarray, got {type(arr)}")
        if arr.mtype != ref_mtype or arr.q != ref_q:
            raise ValueError("stack: all Fuzzarrays must have the same mtype and parameters")
        if arr.shape != ref_shape:
            raise ValueError(f"stack: all Fuzzarrays must have the same shape, expected {ref_shape}, got {arr.shape}")

    # Get component arrays from all arrays
    component_lists = [[] for _ in obj.backend.get_component_arrays()]
    for arr in all_arrays:
        components = arr.backend.get_component_arrays()
        for i, comp in enumerate(components):
            component_lists[i].append(comp)

    # Stack each component list
    new_components = [np.stack(comp_list, axis=axis) for comp_list in component_lists]

    # Create new backend
    backend_cls = obj.backend.__class__
    new_backend = backend_cls.from_arrays(*new_components, q=obj.q)

    return Fuzzarray(backend=new_backend)


def _append_factory(obj: Union[Fuzznum, Fuzzarray],
                    item: Union[Fuzznum, Fuzzarray, list],
                    axis: Optional[int] = 0,
                    inplace: bool = False) -> Union[Fuzzarray, None]:
    """
    Append elements to a fuzzy object (Fuzznum or Fuzzarray).

    Provides numpy-like ``append`` semantics extended for fuzzy containers,
    with support for broadcasting, multi-axis operations, and list input.

    Parameters
    ----------
    obj : Fuzznum or Fuzzarray
        Target fuzzy object:
        - ``Fuzznum``: promoted to a single-element Fuzzarray before append.
        - ``Fuzzarray``: append directly.
    item : Fuzznum, Fuzzarray, or list
        Elements to append:
        - ``Fuzznum``: single fuzzy number, can broadcast.
        - ``Fuzzarray``: must be shape-compatible for concatenation.
        - ``list[Fuzznum]``: converted into a Fuzzarray then appended.
        - ``list[Fuzzarray]``: concatenated directly along the given axis.
    axis : int or None, default=0
        Axis along which to append.
        - ``None``: both inputs are flattened and concatenated into 1D.
        - ``int``: append along this axis.
    inplace : bool, default=False
        If True, modify ``obj`` in place (only valid when ``obj`` is Fuzzarray).
        If False, return a new Fuzzarray.

    Returns
    -------
    Fuzzarray or None
        If ``inplace=False``, returns a new Fuzzarray with appended elements.
        If ``inplace=True``, modifies ``obj`` and returns None.

    Raises
    ------
    TypeError
        If inputs are of unsupported types.
    ValueError
        If shapes are incompatible for concatenation or broadcasting.
    RuntimeError
        If item is [] (empty list) with no type context.

    Examples
    --------
    >>> a = fuzznum(mtype="qrofn").create(md=0.2, nmd=0.5)
    >>> b = fuzznum(mtype="qrofn").create(md=0.7, nmd=0.3)
    >>> arr = Fuzzarray(data=a, shape=(2,))
    >>> arr2 = _append_factory(arr, b)
    >>> arr2.shape
    (3,)

    >>> # Append multiple fuzznums
    >>> arr3 = _append_factory(arr, [a, b])
    >>> arr3.shape
    (4,)

    >>> # Append another fuzzarray along new axis
    >>> arr4 = _append_factory(arr, arr, axis=1)
    >>> arr4.shape
    (2, 2)
    """
    # === Helpers ===
    def ensure_fuzzarray(x, ref_mtype=None, ref_q=None) -> Fuzzarray:
        """Convert input into Fuzzarray."""
        if isinstance(x, Fuzzarray):
            return x
        if isinstance(x, Fuzznum):
            return Fuzzarray(data=x, shape=(1,))
        if isinstance(x, list):
            if len(x) == 0:
                raise RuntimeError("Cannot append empty list [] without context")
            if all(isinstance(e, Fuzznum) for e in x):
                return Fuzzarray(data=np.array(x, dtype=object))
            if all(isinstance(e, Fuzzarray) for e in x):
                return _concat_fuzzarrays(*x, axis=axis)
            raise TypeError(f"Unsupported list element types: {[type(e) for e in x]}")
        raise TypeError(f"Unsupported append input type: {type(x)}")

    def _concat_fuzzarrays(*arrays: Fuzzarray, axis: int = 0) -> Fuzzarray:
        """Efficient concat of multiple Fuzzarrays at backend level."""
        arrays = [a for a in arrays if a.size > 0]
        if not arrays:
            raise ValueError("Cannot concat empty arrays.")
        ref = arrays[0]
        for a in arrays:
            if a.mtype != ref.mtype or a.q != ref.q:
                raise ValueError("Mtype/q mismatch in append.")
        comps = list(zip(*[arr.backend.get_component_arrays() for arr in arrays]))
        new_components = [np.concatenate(comp_list, axis=axis) for comp_list in comps]
        backend_cls = ref.backend.__class__
        new_backend = backend_cls.from_arrays(*new_components, q=ref.q, **ref.kwargs)
        return Fuzzarray(backend=new_backend)

    # === Normalize inputs ===
    if isinstance(obj, Fuzznum):
        obj = Fuzzarray(data=obj, shape=(1,))
    if not isinstance(obj, Fuzzarray):
        raise TypeError(f"obj must be Fuzznum or Fuzzarray, got {type(obj)}")

    item = ensure_fuzzarray(item, obj.mtype, obj.q)

    # empty obj
    if obj.size == 0:
        result = item
    else:
        if axis is None:
            # flatten + concat 1D
            comps_obj = obj.backend.get_component_arrays()
            comps_item = item.backend.get_component_arrays()
            comps = [np.concatenate([o.ravel(), i.ravel()]) for o, i in zip(comps_obj, comps_item)]
            backend_cls = obj.backend.__class__
            new_backend = backend_cls.from_arrays(*comps, q=obj.q, **obj.kwargs)
            result = Fuzzarray(backend=new_backend)
        else:
            result = _concat_fuzzarrays(obj, item, axis=axis)

    # inplace
    if inplace:
        obj._backend = result.backend
        return None
    return result


def _pop_factory(obj: Union[Fuzznum, Fuzzarray],
                 index: int = -1,
                 inplace: bool = False) -> Union[Fuzznum, Tuple[Fuzznum, Fuzzarray], None]:
    """
    Remove and return an element from a one-dimensional array.
    """
    if isinstance(obj, Fuzznum):
        raise TypeError("pop: Fuzznum object does not support pop operation")

    if obj.ndim != 1:
        raise ValueError("pop: only one-dimensional Fuzzarray is supported")

    if obj.size == 0:
        raise IndexError("pop from empty Fuzzarray")

    # Get the element to pop
    popped_item = obj[index]

    # Create new array without the popped element
    indices = list(range(obj.size))
    if index < 0:
        index = obj.size + index
    indices.pop(index)

    if len(indices) == 0:
        # Result is empty array
        new_array = Fuzzarray(data=None, shape=(0,), mtype=obj.mtype, q=obj.q)
    else:
        # Create new array with remaining elements
        components = obj.backend.get_component_arrays()
        new_components = [comp[indices] for comp in components]

        backend_cls = obj.backend.__class__
        new_backend = backend_cls.from_arrays(*new_components, q=obj.q)
        new_array = Fuzzarray(backend=new_backend)

    if inplace:
        obj._backend = new_array.backend
        return popped_item
    else:
        return popped_item, new_array


# ========================== Sort operation ===========================
# 暂未实现
def _sort_factory(obj: Union[Fuzznum, Fuzzarray]): ...


def _argsort_factory(obj: Union[Fuzznum, Fuzzarray], axis: int = -1) -> Fuzzarray: ...


def _argmax_factory(obj: Union[Fuzznum, Fuzzarray], axis: int = -1) -> Fuzzarray: ...


def _argmix_factory(obj: Union[Fuzznum, Fuzzarray], axis: int = -1) -> Fuzzarray: ...
