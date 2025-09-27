#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN Constructor Extension Methods.

This module implements constructor functions for creating IVQROFN objects with
various initialization patterns, including empty, positive, negative, and full
initialization modes.
"""

from typing import Tuple, Union, Optional
import numpy as np

from ....core import Fuzznum, Fuzzarray, get_registry_fuzztype


def _ivqrofn_empty(shape: Optional[Tuple[int, ...]] = None,
                   q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create an empty (uninitialized) IVQROFN Fuzzarray or Fuzznum.
    
    Parameters:
        shape: Optional shape for Fuzzarray creation. If None, returns Fuzznum
        q: Q-rung parameter for constraint validation
        
    Returns:
        Union[Fuzzarray, Fuzznum]: Empty IVQROFN with zero intervals
    """
    q = 1 if q is None else q
    if shape is None:
        # Create single Fuzznum with zero intervals
        return Fuzznum(mtype='ivqrofn', q=q).create(md=[0.0, 0.0], nmd=[0.0, 0.0])

    # Create Fuzzarray with specified shape
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    backend = backend_cls(shape=shape, q=q)
    return Fuzzarray(backend=backend)


def _ivqrofn_positive(shape: Optional[Tuple[int, ...]] = None,
                      q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create a positive IVQROFN (maximum membership, zero non-membership).
    
    Parameters:
        shape: Optional shape for Fuzzarray creation. If None, returns Fuzznum
        q: Q-rung parameter for constraint validation
        
    Returns:
        Union[Fuzzarray, Fuzznum]: IVQROFN with md=[1.0, 1.0], nmd=[0.0, 0.0]
    """
    q = 1 if q is None else q
    if shape is None:
        # Create single positive Fuzznum
        return Fuzznum(mtype='ivqrofn', q=q).create(md=[1.0, 1.0], nmd=[0.0, 0.0])

    # Create Fuzzarray filled with positive values
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values([1.0, 1.0], [0.0, 0.0])
    return Fuzzarray(backend=backend)


def _ivqrofn_negative(shape: Optional[Tuple[int, ...]] = None,
                      q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create a negative IVQROFN (zero membership, maximum non-membership).
    
    Parameters:
        shape: Optional shape for Fuzzarray creation. If None, returns Fuzznum
        q: Q-rung parameter for constraint validation
        
    Returns:
        Union[Fuzzarray, Fuzznum]: IVQROFN with md=[0.0, 0.0], nmd=[1.0, 1.0]
    """
    q = 1 if q is None else q
    if shape is None:
        # Create single negative Fuzznum
        return Fuzznum(mtype='ivqrofn', q=q).create(md=[0.0, 0.0], nmd=[1.0, 1.0])

    # Create Fuzzarray filled with negative values
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values([0.0, 0.0], [1.0, 1.0])
    return Fuzzarray(backend=backend)


def _ivqrofn_full(fill_value: Fuzznum,
                  shape: Tuple[int, ...],
                  q: Optional[int] = None) -> Fuzzarray:
    """
    Create an IVQROFN Fuzzarray filled with a specific Fuzznum value.
    
    Parameters:
        fill_value: IVQROFN Fuzznum to fill the array with
        shape: Shape of the resulting Fuzzarray
        q: Q-rung parameter for constraint validation
        
    Returns:
        Fuzzarray: IVQROFN Fuzzarray filled with the specified value
        
    Raises:
        ValueError: If shape is empty or q parameters don't match
        TypeError: If fill_value is not an IVQROFN Fuzznum
    """
    q = 1 if q is None else q

    if not shape or shape is None:
        raise ValueError("shape must be a non-empty tuple of integers.")
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'ivqrofn':
        raise TypeError("fill_value must be an IVQROFN Fuzznum.")
    if fill_value.q != q:
        raise ValueError(f"Q-rung mismatch: array q is {q}, but fill_value q is {fill_value.q}")

    # Create backend and fill with the specified value
    backend_cls = get_registry_fuzztype().get_backend('ivqrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(fill_value.md, fill_value.nmd)
    return Fuzzarray(backend=backend)


def _ivqrofn_empty_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create an empty IVQROFN with the same shape and q as the input object.
    
    Parameters:
        obj: Input Fuzznum or Fuzzarray to copy shape and q from
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Empty IVQROFN matching input characteristics
        
    Raises:
        TypeError: If obj is not a Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = None  # For Fuzznum, use None to create another Fuzznum
        q = obj.q

    return _ivqrofn_empty(shape=shape, q=q)


def _ivqrofn_positive_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create a positive IVQROFN with the same shape and q as the input object.
    
    Parameters:
        obj: Input Fuzznum or Fuzzarray to copy shape and q from
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Positive IVQROFN matching input characteristics
        
    Raises:
        TypeError: If obj is not a Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = None
        q = obj.q

    return _ivqrofn_positive(shape=shape, q=q)


def _ivqrofn_negative_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create a negative IVQROFN with the same shape and q as the input object.
    
    Parameters:
        obj: Input Fuzznum or Fuzzarray to copy shape and q from
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Negative IVQROFN matching input characteristics
        
    Raises:
        TypeError: If obj is not a Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = None
        q = obj.q

    return _ivqrofn_negative(shape=shape, q=q)


def _ivqrofn_full_like(fill_value: Fuzznum, obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    """
    Create an IVQROFN Fuzzarray filled with a specific value and matching input shape.
    
    Parameters:
        fill_value: IVQROFN Fuzznum to fill the array with
        obj: Input Fuzznum or Fuzzarray to copy shape and q from
        
    Returns:
        Fuzzarray: IVQROFN Fuzzarray filled with the specified value
        
    Raises:
        TypeError: If obj is not a Fuzznum or Fuzzarray, or fill_value is not IVQROFN
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'ivqrofn':
        raise TypeError("fill_value must be an IVQROFN Fuzznum.")

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = (1,)  # Create 1-element array for Fuzznum input
        q = obj.q

    return _ivqrofn_full(fill_value=fill_value, shape=shape, q=q)