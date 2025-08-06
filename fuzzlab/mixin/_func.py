#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 13:13
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import copy
from typing import Union, Tuple

import numpy as np

from ._registry import get_mixin_registry
from ..core import Fuzznum, Fuzzarray

registry = get_mixin_registry()


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


@registry.register(name='broadcast_to', target_classes=["Fuzzarray", "Fuzznum"])
def _broadcast_to_impl(self: Union['Fuzzarray', 'Fuzznum'],
                       *shape: int) -> 'Fuzzarray':
    """
    Broadcasts the fuzzy object to a new shape.

    This method mimics the behavior of `numpy.broadcast_to`.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to broadcast.
        *shape (int or tuple of ints): The shape to which the object should be broadcast.
                                       If a single tuple is passed, it should be unpacked.

    Returns:
        Fuzzarray: A new Fuzzarray that is a broadcasted view of the original object.

    Raises:
        ValueError: If the object cannot be broadcast to the specified shape.
    """
    # Handle the case where shape is passed as a single tuple, e.g., broadcast_to((2, 3))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)

    # If self is a Fuzznum, treat it as a 0-dim array for broadcasting.
    if isinstance(self, Fuzznum):
        # np.array(self) creates a 0-dim array containing the Fuzznum object.
        data_array = np.array(self)
    else:
        # For a Fuzzarray, use its underlying numpy data array.
        data_array = self._data

    try:
        # Use numpy's broadcast_to and wrap the result in a new Fuzzarray.
        # np.broadcast_to returns a read-only view.
        broadcasted_data = np.broadcast_to(data_array, target_shape)
        return Fuzzarray(broadcasted_data, copy=False)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast object with shape {self.shape} to shape {target_shape}: {e}")


@registry.register(name='item', target_classes=["Fuzzarray", "Fuzznum"])
def _item_impl(self: Union['Fuzzarray', 'Fuzznum']) -> 'Fuzznum':
    """
    Returns the scalar item of the fuzzy object.

    If the object is a Fuzznum, it returns a copy of itself.
    If the object is a Fuzzarray, it must contain exactly one element,
    which is then returned as a Fuzznum.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.

    Returns:
        Fuzznum: The single Fuzznum element.

    Raises:
        ValueError: If the Fuzzarray contains more than one element.
    """
    if isinstance(self, Fuzznum):
        # Fuzznum is already a scalar, return a copy of itself
        return self.copy()
    elif isinstance(self, Fuzzarray):
        if self.size == 1:
            # If Fuzzarray has only one element, return that Fuzznum
            # self._data is a numpy array, so self._data.item() will extract the single Fuzznum object
            return self._data.item()
        else:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
    else:
        # This case should ideally not be reached due to type hints,
        # but added for robustness.
        raise TypeError(f"Unsupported type for item() method: {type(self)}")


@registry.register(name='sort', target_classes=["Fuzzarray", "Fuzznum"])
def _sort_impl(self: Union['Fuzzarray', 'Fuzznum'],
               axis: int = -1) -> Union['Fuzzarray', 'Fuzznum']:
    """
    Return a sorted copy of a fuzzy array.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to sort.
        axis (int, optional): Axis along which to sort. If None, the array is
                              flattened before sorting. The default is -1,
                              which sorts along the last axis.

    Returns:
        Fuzzarray: A sorted copy of the array.
    """
    if isinstance(self, Fuzznum):
        return self.copy()

    sorted_data = np.sort(self._data, axis=axis)
    return Fuzzarray(sorted_data, copy=False)


@registry.register(name='argsort', target_classes=["Fuzzarray", "Fuzznum"])
def _argsort_impl(self: Union['Fuzzarray', 'Fuzznum'],
                  axis: int = -1) -> np.ndarray:
    """
    Returns the indices that would sort a fuzzy array.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (int, optional): Axis along which to sort. Default is -1.

    Returns:
        np.ndarray: An array of indices that sort the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0)  # Index of a scalar is 0

    return np.argsort(self._data, axis=axis)


@registry.register(name='argmax', target_classes=["Fuzzarray", "Fuzznum"])
def _argmax_impl(self: Union['Fuzzarray', 'Fuzznum'],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[np.integer, int, np.ndarray]:
    """
    Return indices of the maximum values along an axis.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which to operate.
                                                       By default, flattened input is used.

    Returns:
        np.ndarray: Array of indices into the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0).item()

    result = np.argmax(self._data, axis=axis)
    if isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result.item()


@registry.register(name='argmin', target_classes=["Fuzzarray", "Fuzznum"])
def _argmin_impl(self: Union['Fuzzarray', 'Fuzznum'],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[np.integer, int, np.ndarray]:
    """
    Return indices of the minimum values along an axis.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which to operate.
                                                       By default, flattened input is used.

    Returns:
        np.ndarray: Array of indices into the array.
    """
    if isinstance(self, Fuzznum):
        return np.array(0).item()

    result = np.argmin(self._data, axis=axis)
    if isinstance(result, np.ndarray):
        return result.tolist()
    else:
        return result.item()



