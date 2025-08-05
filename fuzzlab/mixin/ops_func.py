#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/5 17:14
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
This module defines common mathematical and aggregation operations
(e.g., sum, mean, max, min) for Fuzzarray and Fuzznum objects.

These operations are registered with the global mixin registry
to be dynamically injected as methods into Fuzzarray and Fuzznum classes,
and also as top-level functions.
"""
import copy
from typing import Union, Tuple

import numpy as np

from .registry import get_mixin_registry

from ..core.fuzzarray import Fuzzarray
from ..core.fuzznums import Fuzznum

# Get the global registry instance
registry = get_mixin_registry()


@registry.register(name='sum', target_classes=["Fuzzarray", "Fuzznum"])
def _sum_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the sum of all elements in the fuzzy array or fuzzy number.

    If `self` is a `Fuzznum`, it returns a copy of itself.
    If `self` is a `Fuzzarray`, it performs summation along specified axes.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to sum.
        axis (Union[int, Tuple[int, ...]], optional): The one or more axes along which
                                                       the summation is performed.
                                                       If None, sum all elements.
                                                       Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The sum result.
                                       If `axis` is None, returns a single `Fuzznum`.
                                       If an `axis` is specified, returns a `Fuzzarray`.

    Raises:
        ValueError: If an `axis` is out of bounds for the array's dimension.
        TypeError: If `np.sum` with `axis=None` does not return a `Fuzznum` object.
    """
    # If the object is a Fuzznum, return a copy of itself as sum of a single number is itself
    if isinstance(self, Fuzznum):
        return self.copy()

    # If the Fuzzarray is empty, return a new Fuzznum with default values
    if not self.size:
        return Fuzznum(self.mtype, self.q).create()

    # Validate the 'axis' argument for Fuzzarray dimensions
    if axis is not None:
        if isinstance(axis, int):
            # Check if the single axis is within valid bounds
            if not (-self.ndim <= axis < self.ndim):
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}")
        elif isinstance(axis, tuple):
            # Check if all axes in the tuple are within valid bounds
            for ax in axis:
                if not (-self.ndim <= ax < self.ndim):
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {self.ndim}")
        # numpy's sum function automatically handles negative axes

    # Perform the summation using numpy's sum function on the underlying data
    result = np.sum(self._data, axis=axis)

    # Determine the return type based on whether an axis was specified
    if axis is None:
        # If axis is None, the result should be a single Fuzznum
        if not isinstance(result, Fuzznum):
            raise TypeError(f"np.sum with axis=None did not return a Fuzznum object, got '{type(result)}'.")
        return result
    else:
        # If an axis was specified, the result is a Fuzzarray
        return Fuzzarray(result, copy=False)


@registry.register(name='mean', target_classes=["Fuzzarray", "Fuzznum"])
def _mean_impl(self: Union['Fuzzarray', 'Fuzznum'],
               axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the mean of all elements in the fuzzy array or fuzzy number.

    If `self` is a `Fuzznum`, it returns a copy of itself.
    If `self` is a `Fuzzarray`, it performs mean calculation along specified axes.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to calculate the mean.
        axis (Union[int, Tuple[int, ...]], optional): The one or more axes along which
                                                       the mean is performed.
                                                       If None, calculate mean of all elements.
                                                       Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The mean result.
                                       If `axis` is None, returns a single `Fuzznum`.
                                       If an `axis` is specified, returns a `Fuzzarray`.

    Raises:
        ValueError: If an `axis` is out of bounds for the array's dimension.
        TypeError: If `np.mean` with `axis=None` does not return a `Fuzznum` object.
    """
    # If the object is a Fuzznum, return a copy of itself as mean of a single number is itself
    if isinstance(self, Fuzznum):
        return self.copy()

    # If the Fuzzarray is empty, return a new Fuzznum with default values
    if not self.size:
        return Fuzznum(self.mtype, self.q).create()

    # Validate the 'axis' argument for Fuzzarray dimensions
    if axis is not None:
        if isinstance(axis, int):
            # Check if the single axis is within valid bounds
            if not (-self.ndim <= axis < self.ndim):
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}")
        elif isinstance(axis, tuple):
            # Check if all axes in the tuple are within valid bounds
            for ax in axis:
                if not (-self.ndim <= ax < self.ndim):
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {self.ndim}")
        # numpy's mean function automatically handles negative axes

    # Perform the mean calculation using numpy's mean function on the underlying data
    result = np.mean(self._data, axis=axis)

    # Determine the return type based on whether an axis was specified
    if axis is None:
        # If axis is None, the result should be a single Fuzznum
        if not isinstance(result, Fuzznum):
            raise TypeError(f"np.mean with axis=None did not return a Fuzznum object, got '{type(result)}'.")
        return result
    else:
        # If an axis was specified, the result is a Fuzzarray
        return Fuzzarray(result, copy=False)


@registry.register(name='max', target_classes=["Fuzzarray", "Fuzznum"])
def _max_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the maximum of all elements in the fuzzy array or fuzzy number.

    If `self` is a `Fuzznum`, it returns a copy of itself.
    If `self` is a `Fuzzarray`, it performs max calculation along specified axes.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to calculate the maximum.
        axis (Union[int, Tuple[int, ...]], optional): The one or more axes along which
                                                       the maximum is performed.
                                                       If None, calculate max of all elements.
                                                       Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The maximum result.
                                       If `axis` is None, returns a single `Fuzznum`.
                                       If an `axis` is specified, returns a `Fuzzarray`.

    Raises:
        ValueError: If an `axis` is out of bounds for the array's dimension.
        TypeError: If `np.max` with `axis=None` does not return a `Fuzznum` object.
    """
    # If the object is a Fuzznum, return a copy of itself as max of a single number is itself
    if isinstance(self, Fuzznum):
        return self.copy()

    # If the Fuzzarray is empty, return a new Fuzznum with default values
    if not self.size:
        return Fuzznum(self.mtype, self.q).create()

    # Validate the 'axis' argument for Fuzzarray dimensions
    if axis is not None:
        if isinstance(axis, int):
            # Check if the single axis is within valid bounds
            if not (-self.ndim <= axis < self.ndim):
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}")
        elif isinstance(axis, tuple):
            # Check if all axes in the tuple are within valid bounds
            for ax in axis:
                if not (-self.ndim <= ax < self.ndim):
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {self.ndim}")
        # numpy's max function automatically handles negative axes

    # Perform the max calculation using numpy's max function on the underlying data
    result = np.max(self._data, axis=axis)

    # Determine the return type based on whether an axis was specified
    if axis is None:
        # If axis is None, the result should be a single Fuzznum
        if not isinstance(result, Fuzznum):
            raise TypeError(f"np.max with axis=None did not return a Fuzznum object, got '{type(result)}'.")
        return result
    else:
        # If an axis was specified, the result is a Fuzzarray
        return Fuzzarray(result, copy=False)


@registry.register(name='min', target_classes=["Fuzzarray", "Fuzznum"])
def _min_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the minimum of all elements in the fuzzy array or fuzzy number.

    If `self` is a `Fuzznum`, it returns a copy of itself.
    If `self` is a `Fuzzarray`, it performs min calculation along specified axes.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to calculate the minimum.
        axis (Union[int, Tuple[int, ...]], optional): The one or more axes along which
                                                       the minimum is performed.
                                                       If None, calculate min of all elements.
                                                       Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The minimum result.
                                       If `axis` is None, returns a single `Fuzznum`.
                                       If an `axis` is specified, returns a `Fuzzarray`.

    Raises:
        ValueError: If an `axis` is out of bounds for the array's dimension.
        TypeError: If `np.min` with `axis=None` does not return a `Fuzznum` object.
    """
    # If the object is a Fuzznum, return a copy of itself as min of a single number is itself
    if isinstance(self, Fuzznum):
        return self.copy()

    # If the Fuzzarray is empty, return a new Fuzznum with default values
    if not self.size:
        return Fuzznum(self.mtype, self.q).create()

    # Validate the 'axis' argument for Fuzzarray dimensions
    if axis is not None:
        if isinstance(axis, int):
            # Check if the single axis is within valid bounds
            if not (-self.ndim <= axis < self.ndim):
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}")
        elif isinstance(axis, tuple):
            # Check if all axes in the tuple are within valid bounds
            for ax in axis:
                if not (-self.ndim <= ax < self.ndim):
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {self.ndim}")
        # numpy's min function automatically handles negative axes

    # Perform the min calculation using numpy's min function on the underlying data
    result = np.min(self._data, axis=axis)

    # Determine the return type based on whether an axis was specified
    if axis is None:
        # If axis is None, the result should be a single Fuzznum
        if not isinstance(result, Fuzznum):
            raise TypeError(f"np.min with axis=None did not return a Fuzznum object, got '{type(result)}'.")
        return result
    else:
        # If an axis was specified, the result is a Fuzzarray
        return Fuzzarray(result, copy=False)


@registry.register(name='reshape', target_classes=["Fuzzarray", "Fuzznum"])
def _reshape_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  *shape: int) -> 'Fuzzarray':
    """
    Gives a new shape to an array without changing its data.

    This method follows the behavior of `numpy.ndarray.reshape`.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to reshape.
        *shape (int or tuple of ints): The new shape should be compatible with the
                                       original shape. If an integer, then the
                                       result will be a 1-D array of that length.
                                       One shape dimension can be -1, in which case
                                       the value is inferred from the length of
                                       the array and remaining dimensions.

    Returns:
        Fuzzarray: A new Fuzzarray with the specified shape. It will be a view
                   of the original array if possible, otherwise a copy.

    Raises:
        ValueError: If the new shape is not compatible with the original shape.
    """
    # Handle the case where shape is passed as a single tuple, e.g., reshape((2, 3))
    # which is a common numpy pattern.
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    # If self is a Fuzznum, treat it as a 0-dim array for reshaping.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # Use numpy's reshape and wrap the result in a new Fuzzarray.
    # numpy.reshape returns a view if possible, otherwise a copy.
    reshaped_data = data_array.reshape(shape)
    return Fuzzarray(reshaped_data, copy=False)


@registry.register(name='flatten', target_classes=["Fuzzarray", "Fuzznum"])
def _flatten_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzzarray':
    """
    Return a copy of the array collapsed into one dimension.

    It will return a copy of the flattened one-dimensional `Fuzzarray`.
    For a `Fuzznum`, it will return a one-dimensional `Fuzzarray` containing that element.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to flatten.

    Returns:
        Fuzzarray: A new 1-D Fuzzarray containing a copy of the elements.
    """
    # If self is a Fuzznum, treat it as a 0-dim array.
    if isinstance(self, Fuzznum):
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # numpy.flatten always returns a new 1-D copy of the array.
    flattened_data = data_array.flatten()
    # Wrap the flattened copy in a new Fuzzarray.
    return Fuzzarray(flattened_data, copy=False)


@registry.register(name='squeeze', target_classes=["Fuzzarray", "Fuzznum"])
def _squeeze_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Remove single-dimensional entries from the shape of an array.

    Entries that remove one-dimensional elements from the array's shape.
    If the result is a 0-dimensional array, it will return a Fuzznum scalar;
    otherwise, it will return a new Fuzzarray.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to squeeze.
        axis (None or int or tuple of ints, optional): Selects a subset of the
            single-dimensional entries to remove. If an axis is selected with
            a shape entry greater than one, a ValueError is raised.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The squeezed fuzzy object. If the array
                                       becomes a 0-d array, a Fuzznum is returned.
                                       Otherwise, a Fuzzarray is returned (as a view).
    """
    # If self is a Fuzznum, it's already a scalar (squeezed). Return a copy.
    if isinstance(self, Fuzznum):
        return self.copy()

    # For a Fuzzarray, use the underlying numpy array's squeeze method.
    # This returns a view of the original data.
    squeezed_data = self._data.squeeze(axis=axis)

    # If the result is a 0-d array, it contains a single Fuzznum.
    # Extract and return the Fuzznum object itself.
    if squeezed_data.ndim == 0:
        return squeezed_data.item()
    else:
        # Otherwise, return a new Fuzzarray wrapping the squeezed data view.
        return Fuzzarray(squeezed_data, copy=False)


@registry.register_top_level_function(name='copy')
def _copy_top_level_impl(obj: Union['Fuzzarray', 'Fuzznum']) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Returns a deep copy of the fuzzy object.

    This function creates a new object that is a complete and independent copy
    of the original. Any changes to the new object will not affect the
    original object.

    Args:
        obj (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to copy.

    Returns:
        Union['Fuzzarray', 'Fuzznum']: A new object that is a deep copy of the original.

    Raises:
        TypeError: If the input object is not a Fuzzarray or Fuzznum.
    """
    if isinstance(obj, (Fuzzarray, Fuzznum)):
        # Delegate the actual copying to the object's own copy method.
        # This assumes Fuzzarray and Fuzznum both have a .copy() method
        # that performs a deep copy suitable for their internal structure.
        return obj.copy()
    else:
        raise TypeError(f"Unsupported type for copy: {type(obj)}. Expected Fuzzarray or Fuzznum.")


@registry.register(name='ravel', target_classes=["Fuzzarray", "Fuzznum"])
def _ravel_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzzarray':
    """
    Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned.
    A view is returned if possible; otherwise, a copy is made. This is
    the primary difference from `flatten`, which always returns a copy.

    For a `Fuzznum`, this returns a 1-D `Fuzzarray` of size 1 containing
    the number.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to ravel.

    Returns:
        Fuzzarray: A new 1-D Fuzzarray. It will be a view of the original
                   array if possible, otherwise a copy.
    """
    # If self is a Fuzznum, treat it as a 0-dim array.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    # Use numpy.ravel on the data array. This returns a view if possible.
    raveled_data = np.ravel(data_array)

    # Wrap the result in a new Fuzzarray, preserving the view semantics
    # by setting copy=False.
    return Fuzzarray(raveled_data, copy=False)


@registry.register_top_level_function(name='transpose')
def _transpose_impl(obj: Union['Fuzzarray', 'Fuzznum']) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Returns a view of the fuzzy object with axes transposed.

    For a Fuzzarray, this is equivalent to `obj.T`.
    For a Fuzznum, it returns the Fuzznum itself.

    Args:
        obj (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to transpose.

    Returns:
        Union['Fuzzarray', 'Fuzznum']: The transposed fuzzy object.
    """
    if isinstance(obj, Fuzznum):
        return copy.deepcopy(obj)
    elif isinstance(obj, Fuzzarray):
        return Fuzzarray(obj.data.T)
    else:
        raise TypeError(f"Unsupported type for transpose: {type(obj)}")



