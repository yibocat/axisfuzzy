#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Constructor Extension Methods.

This module implements constructor methods for classical fuzzy sets (FS),
providing factory functions for creating FS objects with common initialization
patterns. All methods respect the simple mathematical structure of FS with
only membership degrees.

Mathematical Foundation:
    Classical fuzzy sets are characterized by a single membership function
    μ_A: X → [0, 1], making them the simplest and most fundamental fuzzy type.
"""

from typing import Tuple, Union, Optional

from ....core import Fuzznum, Fuzzarray, get_registry_fuzztype


def _fs_empty(shape: Optional[Tuple[int, ...]] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create empty (uninitialized) FS objects with membership degree 0.0.
    
    For classical fuzzy sets, "empty" means no membership (md=0.0),
    representing the empty set in classical fuzzy set theory.
    
    Parameters:
        shape (Optional[Tuple[int, ...]]): Target shape. None creates Fuzznum.
        
    Returns:
        Union[Fuzzarray, Fuzznum]: Empty FS object(s)
    """
    if shape is None:
        return Fuzznum(mtype='fs').create(md=0.0)

    backend_cls = get_registry_fuzztype().get_backend('fs')
    backend = backend_cls(shape=shape)
    backend.fill_from_values(0.0)  # Fill with empty membership
    return Fuzzarray(backend=backend)


def _fs_positive(shape: Optional[Tuple[int, ...]] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create positive FS objects with maximum membership degree 1.0.
    
    For classical fuzzy sets, "positive" means full membership (md=1.0),
    representing complete membership in the fuzzy set.
    
    Parameters:
        shape (Optional[Tuple[int, ...]]): Target shape. None creates Fuzznum.
        
    Returns:
        Union[Fuzzarray, Fuzznum]: Positive FS object(s)
    """
    if shape is None:
        return Fuzznum(mtype='fs').create(md=1.0)

    backend_cls = get_registry_fuzztype().get_backend('fs')
    backend = backend_cls(shape=shape)
    backend.fill_from_values(1.0)  # Fill with full membership
    return Fuzzarray(backend=backend)


def _fs_negative(shape: Optional[Tuple[int, ...]] = None) -> Union[Fuzzarray, Fuzznum]:
    """
    Create negative FS objects with minimum membership degree 0.0.
    
    For classical fuzzy sets, "negative" is equivalent to empty (md=0.0),
    representing no membership in the fuzzy set. This maintains consistency
    with the extension framework while respecting FS mathematical properties.
    
    Parameters:
        shape (Optional[Tuple[int, ...]]): Target shape. None creates Fuzznum.
        
    Returns:
        Union[Fuzzarray, Fuzznum]: Negative FS object(s)
    """
    if shape is None:
        return Fuzznum(mtype='fs').create(md=0.0)

    backend_cls = get_registry_fuzztype().get_backend('fs')
    backend = backend_cls(shape=shape)
    backend.fill_from_values(0.0)  # Fill with no membership
    return Fuzzarray(backend=backend)


def _fs_full(fill_value: Fuzznum, shape: Tuple[int, ...]) -> Fuzzarray:
    """
    Create FS Fuzzarray filled with a specific Fuzznum value.
    
    This factory method creates a new FS Fuzzarray where all elements
    are initialized to the same membership degree from the provided Fuzznum.
    
    Parameters:
        fill_value (Fuzznum): FS Fuzznum to use as fill value
        shape (Tuple[int, ...]): Target shape for the array
        
    Returns:
        Fuzzarray: FS Fuzzarray filled with the specified value
        
    Raises:
        ValueError: If shape is empty or None
        TypeError: If fill_value is not an FS Fuzznum
    """
    if not shape or shape is None:
        raise ValueError("shape must be a non-empty tuple of integers.")
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'fs':
        raise TypeError("fill_value must be an FS Fuzznum.")

    backend_cls = get_registry_fuzztype().get_backend('fs')
    backend = backend_cls(shape=shape)
    backend.fill_from_values(fill_value.md)
    return Fuzzarray(backend=backend)


def _fs_empty_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create empty FS object with the same shape as the input.
    
    Parameters:
        obj (Union[Fuzznum, Fuzzarray]): Template object for shape
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Empty FS object with matching shape
        
    Raises:
        TypeError: If obj is not Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
    else:
        shape = None  # Scalar shape for Fuzznum

    return _fs_empty(shape=shape)


def _fs_positive_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create positive FS object with the same shape as the input.
    
    Parameters:
        obj (Union[Fuzznum, Fuzzarray]): Template object for shape
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Positive FS object with matching shape
        
    Raises:
        TypeError: If obj is not Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
    else:
        shape = None  # Scalar shape for Fuzznum

    return _fs_positive(shape=shape)


def _fs_negative_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create negative FS object with the same shape as the input.
    
    Parameters:
        obj (Union[Fuzznum, Fuzzarray]): Template object for shape
        
    Returns:
        Union[Fuzznum, Fuzzarray]: Negative FS object with matching shape
        
    Raises:
        TypeError: If obj is not Fuzznum or Fuzzarray
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
    else:
        shape = None  # Scalar shape for Fuzznum

    return _fs_negative(shape=shape)


def _fs_full_like(fill_value: Fuzznum, obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    """
    Create FS object filled with specific value, matching input shape.
    
    Parameters:
        fill_value (Fuzznum): FS Fuzznum to use as fill value
        obj (Union[Fuzznum, Fuzzarray]): Template object for shape
        
    Returns:
        Union[Fuzznum, Fuzzarray]: FS object filled with value, matching shape
        
    Raises:
        TypeError: If fill_value is not FS Fuzznum or obj is invalid
    """
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'fs':
        raise TypeError("fill_value must be an FS Fuzznum.")

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        return _fs_full(fill_value=fill_value, shape=shape)
    else:
        # For Fuzznum, return a copy of fill_value
        return Fuzznum(mtype='fs').create(md=fill_value.md)