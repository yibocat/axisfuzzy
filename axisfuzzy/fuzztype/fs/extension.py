#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
Fuzzy Sets (FS) Extension Registration.

This module registers all FS extension methods with the AxisFuzzy extension system,
making them available as instance methods, instance properties, and top-level functions
based on Zadeh's classical fuzzy set theory.

All extensions are automatically registered when this module is imported,
following the Register-Dispatch-Inject architecture pattern.
"""

from typing import Union

import numpy as np

from . import ext

from ...extension import extension
from ...core import Fuzznum, Fuzzarray


# ========================= Constructor Extensions =========================

@extension(name='empty', mtype='fs', injection_type='top_level_function')
def fs_empty_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty (uninitialized) FS Fuzzarray or Fuzznum."""
    return ext._fs_empty(*args, **kwargs)


@extension(name='positive', mtype='fs', injection_type='top_level_function')
def fs_positive_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray or Fuzznum filled with maximum membership (md=1.0)."""
    return ext._fs_positive(*args, **kwargs)


@extension(name='negative', mtype='fs', injection_type='top_level_function')
def fs_negative_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray or Fuzznum filled with minimum membership (md=0.0)."""
    return ext._fs_negative(*args, **kwargs)


@extension(name='full', mtype='fs', injection_type='top_level_function')
def fs_full_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray filled with a specific Fuzznum value."""
    return ext._fs_full(*args, **kwargs)


@extension(name='empty_like', mtype='fs', injection_type='top_level_function')
def fs_empty_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty FS Fuzzarray or Fuzznum with the same shape as the input."""
    return ext._fs_empty_like(*args, **kwargs)


@extension(name='positive_like', mtype='fs', injection_type='top_level_function')
def fs_positive_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray or Fuzznum filled with maximum membership 
    with the same shape as the input."""
    return ext._fs_positive_like(*args, **kwargs)


@extension(name='negative_like', mtype='fs', injection_type='top_level_function')
def fs_negative_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray or Fuzznum filled with minimum membership 
    with the same shape as the input."""
    return ext._fs_negative_like(*args, **kwargs)


@extension(name='full_like', mtype='fs', injection_type='top_level_function')
def fs_full_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a FS Fuzzarray or Fuzznum filled with a specific value
    with the same shape as the input."""
    return ext._fs_full_like(*args, **kwargs)


# ========================= I/O Operation Extensions =========================

@extension(name='to_csv', mtype='fs', target_classes=['Fuzzarray'])
def fs_to_csv_ext(fuzz, *args, **kwargs):
    """Save a FS Fuzzarray to a CSV file."""
    return ext._fs_to_csv(fuzz, *args, **kwargs)


@extension(
    name='read_csv',
    mtype='fs',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def fs_from_csv_ext(*args, **kwargs) -> Fuzzarray:
    """Load a FS Fuzzarray from a CSV file.
    
    This serves as the default implementation for reading CSV files when no mtype
    is specified, assuming the simplest fuzzy set format (FS).
    """
    return ext._fs_from_csv(*args, **kwargs)


@extension(name='to_json', mtype='fs', target_classes=['Fuzzarray'])
def fs_to_json_ext(fuzz, *args, **kwargs):
    """Save a FS Fuzzarray to a JSON file."""
    return ext._fs_to_json(fuzz, *args, **kwargs)


@extension(
    name='read_json',
    mtype='fs',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def fs_from_json_ext(*args, **kwargs) -> Fuzzarray:
    """Load a FS Fuzzarray from a JSON file.
    
    This serves as the default implementation for reading JSON files when no mtype
    is specified, assuming the simplest fuzzy set format (FS).
    """
    return ext._fs_from_json(*args, **kwargs)


@extension(name='to_npy', mtype='fs', target_classes=['Fuzzarray'])
def fs_to_npy_ext(fuzz, *args, **kwargs):
    """Save a FS Fuzzarray to a NumPy binary file."""
    return ext._fs_to_npy(fuzz, *args, **kwargs)


@extension(
    name='read_npy',
    mtype='fs',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def fs_from_npy_ext(*args, **kwargs) -> Fuzzarray:
    """Load a FS Fuzzarray from a NumPy binary file.
    
    This serves as the default implementation for reading NPY files when no mtype
    is specified, assuming the simplest fuzzy set format (FS).
    """
    return ext._fs_from_npy(*args, **kwargs)


# ========================= Measurement Operation Extensions =========================

@extension(name='distance', mtype='fs')
def fs_distance_ext(fuzz_1, fuzz_2, p_l=2) -> Union[np.ndarray, float]:
    """Calculate the distance between two FS Fuzzarrays or Fuzznums."""
    return ext._fs_distance(fuzz_1, fuzz_2, p_l=p_l)


# ========================= String Conversion Extensions =========================

@extension(
    name='str2fuzznum',
    mtype='fs',
    target_classes=['Fuzznum'],
    injection_type='top_level_function')
def fs_from_str_ext(fuzznum_str: str) -> Fuzznum:
    """Convert a string representation of a FS Fuzznum to an actual Fuzznum.
    
    This serves as the default implementation for string parsing when no mtype
    is specified, attempting to parse the simplest fuzzy set format (FS).
    """
    return ext._fs_from_str(fuzznum_str)


# ========================= Mathematical Operation Extensions =========================

@extension(
    name='sum',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_sum_ext(fuzz: Union[Fuzzarray, Fuzznum],
               axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the sum of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_sum(fuzz, axis=axis)


@extension(
    name='mean',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_mean_ext(fuzz: Union[Fuzzarray, Fuzznum],
                axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the mean of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_mean(fuzz, axis=axis)


@extension(
    name='max',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_max_ext(fuzz: Union[Fuzzarray, Fuzznum],
               axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the maximum of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_max(fuzz, axis=axis)


@extension(
    name='min',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_min_ext(fuzz: Union[Fuzzarray, Fuzznum],
               axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the minimum of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_min(fuzz, axis=axis)


@extension(
    name='prod',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_prod_ext(fuzz: Union[Fuzzarray, Fuzznum],
                axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the product of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_prod(fuzz, axis=axis)


@extension(
    name='var',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_var_ext(fuzz: Union[Fuzzarray, Fuzznum],
               axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the variance of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_var(fuzz, axis=axis)


@extension(
    name='std',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def fs_std_ext(fuzz: Union[Fuzzarray, Fuzznum],
               axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the standard deviation of a FS Fuzzarray or Fuzznum along a specified axis."""
    return ext._fs_std(fuzz, axis=axis)


# ========================= Property Extensions =========================

@extension(
    name='score',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def fs_score_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the score of a FS Fuzzarray or Fuzznum (equivalent to membership degree)."""
    return ext._fs_score(fuzz)


@extension(
    name='acc',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def fs_acc_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the accuracy of a FS Fuzzarray or Fuzznum (equivalent to membership degree)."""
    return ext._fs_acc(fuzz)


@extension(
    name='ind',
    mtype='fs',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def fs_ind_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the indeterminacy of a FS Fuzzarray or Fuzznum (1 - membership degree)."""
    return ext._fs_ind(fuzz)