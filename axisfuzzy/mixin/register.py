#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Registration of structural mixin operations for AxisFuzzy core classes.

This module registers mtype-agnostic structural and container operations that
extend the functionality of :class:`axisfuzzy.core.fuzznums.Fuzznum` and
:class:`axisfuzzy.core.fuzzarray.Fuzzarray` with NumPy-like capabilities.

The registered functions focus on shape manipulation, view operations, and
common container utilities that work uniformly across all fuzzy number types
without requiring mtype-specific dispatch logic.

Architecture
------------
The registration follows a three-layer pattern:

1. **Implementation Layer** (:mod:`axisfuzzy.mixin.factory`): Contains the actual
   function implementations that operate on the SoA (Struct-of-Arrays) backend.

2. **Registration Layer** (this module): Uses :func:`axisfuzzy.mixin.registry.register_mixin`
   decorator to declare how each function should be exposed (instance method,
   top-level function, or both).

3. **Injection Layer** (:mod:`axisfuzzy.mixin.registry`): Dynamically attaches
   the registered functions to target classes and the module namespace during
   library initialization.

Registered Functions
--------------------
This module registers the following categories of mixin functions:

**Shape Manipulation Functions**:
    - ``_reshape_impl``: Delegates to :func:`factory._reshape_factory` to give arrays new shapes without changing data
    - ``_flatten_impl``: Delegates to :func:`factory._flatten_factory` to collapse arrays into 1D
    - ``_squeeze_impl``: Delegates to :func:`factory._squeeze_factory` to remove single-dimensional entries
    - ``_ravel_impl``: Delegates to :func:`factory._ravel_factory` to return contiguous flattened arrays

**Transformation Functions**:
    - ``_transpose_impl``: Delegates to :func:`factory._transpose_factory` to transpose array dimensions
    - ``_T_impl``: Property wrapper for transpose operation, provides ``.T`` attribute access
    - ``_broadcast_to_impl``: Delegates to :func:`factory._broadcast_to_factory` to broadcast arrays to new shapes

**Data Access Functions**:
    - ``_copy_impl``: Delegates to :func:`factory._copy_factory` to create deep copies of fuzzy objects
    - ``_item_impl``: Delegates to :func:`factory._item_factory` to extract scalar items from arrays

**Container Operations**:
    - ``_concat_impl``: Delegates to :func:`factory._concat_factory` to join arrays along existing axes
    - ``_stack_impl``: Delegates to :func:`factory._stack_factory` to stack arrays along new axes
    - ``_append_impl``: Delegates to :func:`factory._append_factory` to append elements to arrays
    - ``_pop_impl``: Delegates to :func:`factory._pop_factory` to remove and return elements from arrays

**Boolean Testing Functions**:
    - ``_any_impl``: Delegates to :func:`factory._any_factory` to test if any array elements are truthy
    - ``_all_impl``: Delegates to :func:`factory._all_factory` to test if all array elements are truthy

Injection Types
---------------
Functions are registered with different injection types to control their exposure:

- **'both'**: Available as both instance methods (e.g., ``arr.reshape(2, 3)``) and
  top-level functions (e.g., ``axisfuzzy.reshape(arr, 2, 3)``). Used for most operations.
- **'top_level_function'**: Only available as module-level functions (e.g., ``axisfuzzy.copy(obj)``).
- **'instance_function'**: Only available as bound methods on target classes (e.g., ``arr.T``).

Key Differences from Extension System
-------------------------------------
- **Mixin System**: mtype-agnostic, focuses on structural operations like reshape,
  flatten, transpose that work the same way for any fuzzy number type.
- **Extension System**: mtype-sensitive, provides specialized implementations for
  different fuzzy types (e.g., distance calculation varies between qrofn and ivfn).

Registration Pattern
--------------------
All functions follow a consistent registration pattern:

.. code-block:: python

    @register_mixin(name='function_name',
                    target_classes=['Fuzznum', 'Fuzzarray'],
                    injection_type='both')
    def _function_name_impl(self, *args, **kwargs):
        return _function_name_factory(self, *args, **kwargs)

This pattern ensures:
- Uniform naming convention (``_function_name_impl`` for registration wrappers)
- Consistent delegation to factory implementations
- Proper metadata attachment for injection
- Complete documentation inheritance from factory functions

Notes
-----
- All registered functions delegate to factory implementations in
  :mod:`axisfuzzy.mixin.factory` for actual computation.
- Injection happens at library initialization via :func:`axisfuzzy.mixin.registry.MixinFunctionRegistry.build_and_inject`.
- Functions are designed to work with the SoA backend architecture of :class:`Fuzzarray`.
- Registration order does not affect functionality but follows logical grouping for maintainability.

See Also
--------
axisfuzzy.mixin.factory : Implementation layer for mixin operations.
axisfuzzy.mixin.registry : Registration and injection infrastructure.
axisfuzzy.core.fuzzarray : Primary target class for structural operations.
axisfuzzy.core.fuzznums : Secondary target class for scalar operations.

Examples
--------
After library initialization, registered functions are available as both
instance methods and top-level functions:

.. code-block:: python

    from axisfuzzy.core import fuzznum, fuzzarray
    from axisfuzzy import reshape, copy  # top-level functions

    # Create sample data
    a = fuzznum((0.6, 0.3), mtype='qrofn')
    arr = fuzzarray([a, a, a, a])

    # Instance methods (injected via mixin system)
    reshaped = arr.reshape(2, 2)
    flattened = arr.flatten()
    copied = arr.copy()

    # Top-level functions (also injected via mixin system)
    reshaped2 = reshape(arr, 2, 2)
    copied2 = copy(arr)

References
----------
- Backend architecture: axisfuzzy.core.backend
"""

from .registry import register_mixin
from .factory import (
    _reshape_factory, _flatten_factory, _squeeze_factory, _copy_factory,
    _ravel_factory, _transpose_factory, _broadcast_to_factory, _item_factory,
    _concat_factory, _stack_factory, _append_factory, _pop_factory, _any_factory, _all_factory
)


@register_mixin(name='reshape', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _reshape_impl(self, *shape: int):
    """
    Give a new shape to a fuzzy array without changing its data.

    This method provides NumPy-like ``reshape`` functionality for fuzzy containers.
    It works by reshaping the underlying component arrays in the SoA backend
    while preserving the fuzzy number data.

    Parameters
    ----------
    *shape : int
        New shape for the array. One dimension can be -1, in which case
        its value is inferred from the array size and remaining dimensions.

    Returns
    -------
    Fuzzarray
        A new Fuzzarray with the specified shape. For Fuzznum input,
        returns a Fuzzarray with the requested shape filled with the input value.

    Raises
    ------
    ValueError
        If the new shape is incompatible with the array size.

    Examples
    --------
    .. code-block:: python

        # Reshape a 1D array to 2D
        arr = fuzzarray([a, a, a, a])  # shape (4,)
        reshaped = arr.reshape(2, 2)   # shape (2, 2)

        # Reshape with automatic dimension inference
        reshaped2 = arr.reshape(2, -1)  # shape (2, 2)

        # Broadcast a Fuzznum to an array shape
        scalar = fuzznum((0.6, 0.3))
        broadcasted = scalar.reshape(3, 3)  # shape (3, 3)
    """
    return _reshape_factory(self, *shape)


@register_mixin(name='flatten', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _flatten_impl(self):
    """
    Return a copy of the array collapsed into one dimension.

    This method provides NumPy-like ``flatten`` functionality, converting
    a multi-dimensional fuzzy array into a 1D array containing the same elements.

    Returns
    -------
    Fuzzarray
        A 1D copy of the input array. For Fuzznum input, returns a
        Fuzzarray with shape (1,) containing the input value.

    Examples
    --------
    .. code-block:: python

        # Flatten a 2D array
        arr = fuzzarray([[a, b], [c, d]])  # shape (2, 2)
        flat = arr.flatten()               # shape (4,)

        # Flatten a Fuzznum (creates single-element array)
        scalar = fuzznum((0.6, 0.3))
        flat_scalar = scalar.flatten()     # shape (1,)
    """
    return _flatten_factory(self)


@register_mixin(name='squeeze', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _squeeze_impl(self, axis=None):
    """
    Remove single-dimensional entries from the shape of an array.

    This method provides NumPy-like ``squeeze`` functionality, eliminating
    dimensions of size 1 from the array shape.

    Parameters
    ----------
    axis : int, tuple of int, or None, optional
        Selects a subset of the single-dimensional entries in the shape.
        If None (default), all single-dimensional entries are removed.

    Returns
    -------
    Fuzznum or Fuzzarray
        The input array with single-dimensional axes removed. If the result
        becomes 0-dimensional, returns a Fuzznum; otherwise returns a Fuzzarray.

    Examples
    --------
    .. code-block:: python

        # Remove all single dimensions
        arr = fuzzarray([[[a]], [[b]]])  # shape (2, 1, 1)
        squeezed = arr.squeeze()         # shape (2,)

        # Remove specific single dimension
        squeezed_axis = arr.squeeze(axis=1)  # shape (2, 1)

        # Squeeze Fuzznum (no change)
        scalar = fuzznum((0.6, 0.3))
        squeezed_scalar = scalar.squeeze()   # still a Fuzznum
    """
    return _squeeze_factory(self, axis)


@register_mixin(name='copy', injection_type='top_level_function')
def _copy_impl(obj):
    """
    Return a deep copy of the fuzzy object.

    This function creates an independent copy of a Fuzznum or Fuzzarray,
    ensuring that modifications to the copy do not affect the original.

    Parameters
    ----------
    obj : Fuzznum or Fuzzarray
        The fuzzy object to copy.

    Returns
    -------
    Fuzznum or Fuzzarray
        A deep copy of the input object with the same type, shape, and values.

    Raises
    ------
    TypeError
        If the input object is not a Fuzznum or Fuzzarray.

    Examples
    --------
    .. code-block:: python

        from axisfuzzy import copy

        # Copy a Fuzznum
        original = fuzznum((0.6, 0.3))
        copied = copy(original)
        copied.md = 0.8  # original.md remains 0.6

        # Copy a Fuzzarray
        arr = fuzzarray([a, b, c])
        arr_copy = copy(arr)
        arr_copy[0].md = 0.9  # arr[0].md remains unchanged
    """
    return _copy_factory(obj)


@register_mixin(name='ravel', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _ravel_impl(self):
    """
    Return a contiguous flattened array.

    This method is similar to ``flatten`` but may return a view when possible,
    providing NumPy-like ``ravel`` semantics for fuzzy containers.

    Returns
    -------
    Fuzzarray
        A 1D array containing the same elements as the input.
        For Fuzznum input, returns a Fuzzarray with shape (1,).

    Notes
    -----
    In the current implementation, this method always returns a copy (same as flatten).
    Future versions may optimize to return views when the memory layout allows.

    Examples
    --------
    .. code-block:: python

        # Ravel a multi-dimensional array
        arr = fuzzarray([[a, b], [c, d]])  # shape (2, 2)
        raveled = arr.ravel()              # shape (4,)

        # Ravel a Fuzznum
        scalar = fuzznum((0.6, 0.3))
        raveled_scalar = scalar.ravel()    # shape (1,)
    """
    return _ravel_factory(self)


@register_mixin(name='transpose', injection_type='top_level_function')
def _transpose_impl(obj, *axes):
    """
    Return a view of the fuzzy object with axes transposed.

    This function provides NumPy-like ``transpose`` functionality, permuting
    the dimensions of a fuzzy array according to the specified axes order.

    Parameters
    ----------
    obj : Fuzzarray or Fuzznum
        The fuzzy object to transpose.
    *axes : int, optional
        If specified, must be a tuple or list of ints with the same number
        of dimensions as the input array. The i'th axis of the returned array
        will correspond to the axis numbered axes[i] of the input.
        If not specified, reverses the order of all axes.

    Returns
    -------
    Fuzzarray or Fuzznum
        Transposed view of the input. For Fuzznum input, returns a copy
        (since scalar arrays have no dimensions to transpose).

    Examples
    --------
    .. code-block:: python

        from axisfuzzy import transpose

        # Transpose all axes (reverse order)
        arr = fuzzarray([[a, b, c], [d, e, f]])  # shape (2, 3)
        transposed = transpose(arr)               # shape (3, 2)

        # Transpose with specific axes order
        arr3d = fuzzarray([[[a, b]], [[c, d]]])  # shape (2, 1, 2)
        transposed3d = transpose(arr3d, (2, 0, 1))  # shape (2, 2, 1)

        # Transpose Fuzznum (returns copy)
        scalar = fuzznum((0.6, 0.3))
        transposed_scalar = transpose(scalar)     # still a Fuzznum
    """
    return _transpose_factory(obj, *axes)


@register_mixin(name='T', target_classes=["Fuzzarray", "Fuzznum"], injection_type='instance_function')
@property
def _T_impl(self):
    """
    View of the fuzzy array with axes transposed.

    This property provides convenient access to the transpose operation,
    following NumPy's ``.T`` property convention.

    Returns
    -------
    Fuzzarray or Fuzznum
        Transposed view of the array. For arrays, returns a view with
        axes reversed. For Fuzznum, returns a copy (no dimensions to transpose).

    Examples
    --------
    .. code-block:: python

        # Transpose via property access
        arr = fuzzarray([[a, b], [c, d]])  # shape (2, 2)
        transposed = arr.T                 # shape (2, 2)

        # For 1D arrays, T returns a copy
        arr1d = fuzzarray([a, b, c])       # shape (3,)
        transposed1d = arr1d.T             # shape (3,)
    """
    return _transpose_factory(self)


@register_mixin(name='broadcast_to', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _broadcast_to_impl(self, *shape):
    """
    Broadcast the fuzzy object to a new shape.

    This method provides NumPy-like ``broadcast_to`` functionality, creating
    a view of the input array broadcasted to the specified shape according
    to NumPy's broadcasting rules.

    Parameters
    ----------
    *shape : int
        Target shape for broadcasting. The shape must be compatible with
        the input array's shape according to broadcasting rules.

    Returns
    -------
    Fuzzarray
        View of the input array broadcasted to the target shape.

    Raises
    ------
    ValueError
        If the input shape cannot be broadcasted to the target shape.

    Notes
    -----
    Broadcasting rules follow NumPy conventions: shapes are compatible when,
    for each dimension, the sizes are equal, one of them is 1, or one of
    them does not exist.

    Examples
    --------
    .. code-block:: python

        # Broadcast a scalar to an array
        scalar = fuzznum((0.6, 0.3))
        broadcasted = scalar.broadcast_to(3, 4)  # shape (3, 4)

        # Broadcast a 1D array to 2D
        arr1d = fuzzarray([a, b, c])       # shape (3,)
        broadcasted2d = arr1d.broadcast_to(2, 3)  # shape (2, 3)

        # Broadcast with compatible dimensions
        arr = fuzzarray([[a], [b]])        # shape (2, 1)
        broadcasted = arr.broadcast_to(2, 4)     # shape (2, 4)
    """
    return _broadcast_to_factory(self, *shape)


@register_mixin(name='item', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _item_impl(self, *args):
    """
    Return the scalar item of the fuzzy object.

    This method provides NumPy-like ``item`` functionality, extracting a
    single Fuzznum from a Fuzzarray or returning a copy of a Fuzznum.

    Parameters
    ----------
    *args : int, optional
        Index arguments for multi-dimensional arrays. If not provided,
        the array must contain exactly one element.

    Returns
    -------
    Fuzznum
        The scalar element at the specified location, or a copy of the
        input Fuzznum.

    Raises
    ------
    ValueError
        If the array has more than one element and no index is provided.

    Examples
    --------
    .. code-block:: python

        # Extract item from single-element array
        arr = fuzzarray([fuzznum((0.6, 0.3))])  # shape (1,)
        item = arr.item()                       # Fuzznum

        # Extract item with index
        arr2d = fuzzarray([[a, b], [c, d]])     # shape (2, 2)
        item = arr2d.item(1, 0)                 # Fuzznum at position [1, 0]

        # Item of Fuzznum returns copy
        scalar = fuzznum((0.6, 0.3))
        item = scalar.item()                    # copy of scalar
    """
    return _item_factory(self, *args)


@register_mixin(name='any', target_classes=["Fuzzarray"], injection_type='both')
def _any_impl(self):
    """
    Test whether any array element evaluates to True.

    For fuzzy arrays, this returns True if the array is not empty,
    since all valid Fuzznum instances are considered truthy.

    Returns
    -------
    bool
        True if the array contains any elements; False for empty arrays.

    Examples
    --------
    .. code-block:: python

        # Non-empty array
        arr = fuzzarray([a, b, c])
        result = arr.any()  # True

        # Empty array
        empty = fuzzarray([])
        result = empty.any()  # False
    """
    return _any_factory(self)


@register_mixin(name='all', target_classes=["Fuzzarray"], injection_type='both')
def _all_impl(self):
    """
    Test whether all array elements evaluate to True.

    For fuzzy arrays, this is always True for non-empty arrays,
    since all valid Fuzznum instances are considered truthy.

    Returns
    -------
    bool
        True if all elements are truthy (always True for valid Fuzzarrays
        with at least one element); False for empty arrays.

    Examples
    --------
    .. code-block:: python

        # Non-empty array with valid fuzzy numbers
        arr = fuzzarray([a, b, c])
        result = arr.all()  # True

        # Empty array
        empty = fuzzarray([])
        result = empty.all()  # True (vacuously true)
    """
    return _all_factory(self)


@register_mixin(name='concat', target_classes=['Fuzzarray'], injection_type='both')
def _concat_impl(self, *others, axis: int = 0):
    """
    Join a sequence of Fuzzarrays along an existing axis.

    This method provides NumPy-like ``concatenate`` functionality for fuzzy arrays,
    combining multiple arrays along a specified dimension.

    Parameters
    ----------
    *others : Fuzzarray
        Additional Fuzzarray objects to concatenate with this array.
    axis : int, optional
        Axis along which the arrays are joined. Default is 0.

    Returns
    -------
    Fuzzarray
        Concatenated array with combined data from all input arrays.

    Raises
    ------
    TypeError
        If any input is not a Fuzzarray.
    ValueError
        If arrays have incompatible mtypes, q values, or shapes for concatenation.

    Examples
    --------
    .. code-block:: python

        # Concatenate along first axis
        arr1 = fuzzarray([a, b])        # shape (2,)
        arr2 = fuzzarray([c, d])        # shape (2,)
        concatenated = arr1.concat(arr2)  # shape (4,)

        # Concatenate multiple arrays
        arr3 = fuzzarray([e, f])
        concatenated = arr1.concat(arr2, arr3)  # shape (6,)

        # Concatenate along specific axis
        arr2d1 = fuzzarray([[a, b]])    # shape (1, 2)
        arr2d2 = fuzzarray([[c, d]])    # shape (1, 2)
        concatenated = arr2d1.concat(arr2d2, axis=0)  # shape (2, 2)
    """
    return _concat_factory(self, *others, axis=axis)


@register_mixin(name='stack', target_classes=['Fuzzarray'], injection_type='both')
def _stack_impl(self, *others, axis: int = 0):
    """
    Stack Fuzzarrays along a new axis.

    This method provides NumPy-like ``stack`` functionality, joining arrays
    along a newly created dimension rather than an existing one.

    Parameters
    ----------
    *others : Fuzzarray
        Additional Fuzzarray objects to stack with this array. All arrays
        must have the same shape.
    axis : int, optional
        Axis position in the result array along which the input arrays are stacked.
        Default is 0.

    Returns
    -------
    Fuzzarray
        Stacked array with one additional dimension compared to the input arrays.

    Raises
    ------
    TypeError
        If any input is not a Fuzzarray.
    ValueError
        If arrays have incompatible mtypes, q values, or different shapes.

    Examples
    --------
    .. code-block:: python

        # Stack 1D arrays to create 2D array
        arr1 = fuzzarray([a, b])        # shape (2,)
        arr2 = fuzzarray([c, d])        # shape (2,)
        stacked = arr1.stack(arr2)      # shape (2, 2)

        # Stack along different axis
        stacked_axis1 = arr1.stack(arr2, axis=1)  # shape (2, 2)

        # Stack multiple arrays
        arr3 = fuzzarray([e, f])
        stacked = arr1.stack(arr2, arr3)  # shape (3, 2)
    """
    return _stack_factory(self, *others, axis=axis)


@register_mixin(name='append', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _append_impl(self, item, axis=None, inplace=False):
    """
    Append elements to the fuzzy object.

    This method provides NumPy-like ``append`` functionality with additional
    support for in-place modification and fuzzy-specific data types.

    Parameters
    ----------
    item : Fuzznum, Fuzzarray, or list
        Elements to append. Can be a single Fuzznum, another Fuzzarray,
        or a list of Fuzznum/Fuzzarray objects.
    axis : int or None, optional
        Axis along which to append. If None (default), both arrays are
        flattened before concatenation. Default is None.
    inplace : bool, optional
        If True and the object is a Fuzzarray, modify it in place and return None.
        If False, return a new object. Default is False.

    Returns
    -------
    Fuzzarray or None
        If inplace=False, returns a new Fuzzarray with appended elements.
        If inplace=True, modifies the original object and returns None.

    Raises
    ------
    TypeError
        If item contains unsupported types or if inplace=True for Fuzznum.
    ValueError
        If shapes are incompatible for concatenation.

    Examples
    --------
    .. code-block:: python

        # Append single element
        arr = fuzzarray([a, b])
        appended = arr.append(c)  # Fuzzarray([a, b, c])

        # Append array
        arr2 = fuzzarray([d, e])
        appended = arr.append(arr2)  # Fuzzarray([a, b, d, e])

        # In-place append
        arr.append(f, inplace=True)  # modifies arr directly

        # Append with axis specification
        arr2d = fuzzarray([[a, b]])  # shape (1, 2)
        item2d = fuzzarray([[c, d]]) # shape (1, 2)
        appended = arr2d.append(item2d, axis=0)  # shape (2, 2)
    """
    return _append_factory(self, item, axis, inplace)


@register_mixin(name='pop', target_classes=["Fuzznum", "Fuzzarray"], injection_type='both')
def _pop_impl(self, index=-1, inplace=False):
    """
    Remove and return an element from a one-dimensional array.

    This method provides list-like ``pop`` functionality for fuzzy arrays,
    removing and returning an element at a specified index.

    Parameters
    ----------
    index : int, optional
        Index of the element to remove and return. Default is -1 (last element).
    inplace : bool, optional
        If True and the object is a Fuzzarray, modify it in place and return
        only the popped element. If False, return both the popped element and
        the modified array. Default is False.

    Returns
    -------
    Fuzznum or tuple of (Fuzznum, Fuzzarray)
        If inplace=True, returns the popped Fuzznum.
        If inplace=False, returns a tuple of (popped_element, remaining_array).

    Raises
    ------
    TypeError
        If called on a Fuzznum (scalar objects don't support pop).
    ValueError
        If called on a multi-dimensional array (only 1D arrays supported).
    IndexError
        If the array is empty or the index is out of bounds.

    Examples
    --------
    .. code-block:: python

        # Pop last element (out-of-place)
        arr = fuzzarray([a, b, c])
        popped, remaining = arr.pop()  # popped=c, remaining=[a, b]

        # Pop specific index
        popped, remaining = arr.pop(0)  # popped=a, remaining=[b, c]

        # In-place pop
        arr = fuzzarray([a, b, c])
        popped = arr.pop(inplace=True)  # arr is now [a, b], returns c

        # Pop from empty array raises error
        empty = fuzzarray([])
        # empty.pop()  # raises IndexError
    """
    return _pop_factory(self, index, inplace)
