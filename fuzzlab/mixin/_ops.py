#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 17:02
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

# from .registry import get_mixin_registry

from .registry import get_mixin_registry

from ..core._fuzzarray import Fuzzarray
from ..core.fuzznums import Fuzznum

# Get the global registry instance
registry = get_mixin_registry()


@registry.register(name='sum', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
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


@registry.register(name='mean', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
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


@registry.register(name='max', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
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


@registry.register(name='min', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _min_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the minimum of all elements in the fuzzy array or fuzzy number.
    ...existing code...
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


@registry.register(name='prod', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _prod_impl(self: Union['Fuzzarray', 'Fuzznum'],
               axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray']:
    """
    Calculate the product of all elements in the fuzzy array or fuzzy number.

    If `self` is a `Fuzznum`, it returns a copy of itself.
    If `self` is a `Fuzzarray`, it performs product along specified axes.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object for product calculation.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which a product is performed.
                                                       If None, product of all elements. Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray']: The product result.
    """
    if isinstance(self, Fuzznum):
        return self.copy()

    if not self.size:
        # The product of an empty array is the multiplicative identity, 1.
        return Fuzznum(self.mtype, self.q).create()

    result = np.prod(self._data, axis=axis)

    if axis is None:
        if not isinstance(result, Fuzznum):
            raise TypeError(f"np.prod with axis=None did not return a Fuzznum object, got '{type(result)}'.")
        return result
    else:
        return Fuzzarray(result, copy=False)


@registry.register(name='var', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _var_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray', float]:
    """
    Compute the variance along the specified axis.

    For a `Fuzznum`, the variance is 0.
    For a `Fuzzarray`, it calculates the variance using fuzzy arithmetic.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to calculate variance.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which the variance is computed.
                                                       If None, variance of all elements. Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray', float]: The variance. Returns 0.0 for a single Fuzznum.
    """
    if isinstance(self, Fuzznum):
        return 0.0

    if not self.size:
        return Fuzznum(self.mtype, self.q).create(md=0.0, nmd=1.0) # Represents zero

    # keepdims=True is crucial for broadcasting the subtraction correctly
    mean_val = self.mean(axis=axis)
    deviations = self - mean_val
    squared_dev = deviations ** 2
    variance = squared_dev.mean(axis=axis)
    return variance


@registry.register(name='std', target_classes=["Fuzzarray", "Fuzznum"], injection_type='both')
def _std_impl(self: Union['Fuzzarray', 'Fuzznum'],
              axis: Union[int, Tuple[int, ...]] = None) -> Union['Fuzznum', 'Fuzzarray', float]:
    """
    Compute the standard deviation along the specified axis.

    For a `Fuzznum`, the standard deviation is 0.
    For a `Fuzzarray`, it is the positive square root of the variance.

    Args:
        self (Union['Fuzzarray', 'Fuzznum']): The fuzzy object to calculate std dev.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes along which the std dev is computed.
                                                       If None, computed for all elements. Defaults to None.

    Returns:
        Union['Fuzznum', 'Fuzzarray', float]: The standard deviation. Returns 0.0 for a single Fuzznum.
    """
    if isinstance(self, Fuzznum):
        return 0.0

    # Calculate variance first
    variance = self.var(axis=axis)

    # Std dev is the square root of variance
    return variance ** 0.5
