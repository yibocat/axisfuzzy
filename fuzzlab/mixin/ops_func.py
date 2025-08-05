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
