#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/1/16 16:00
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

"""
IVQROFN Extension Registration Module.

This module registers all IVQROFN extension methods with the AxisFuzzy extension
system, providing seamless integration of type-specific functionality for
Interval-Valued Q-Rung Orthopair Fuzzy Numbers.
"""

from typing import Union
import numpy as np

from . import ext
from ...extension import extension
from ...core import Fuzznum, Fuzzarray

# ========================= Constructor Extensions =========================

@extension(name='empty', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_empty_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty (uninitialized) IVQROFN Fuzzarray or Fuzznum."""
    return ext._ivqrofn_empty(*args, **kwargs)


@extension(name='positive', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_positive_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an IVQROFN with maximum membership intervals (md=[1,1], nmd=[0,0])."""
    return ext._ivqrofn_positive(*args, **kwargs)


@extension(name='negative', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_negative_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an IVQROFN with maximum non-membership intervals (md=[0,0], nmd=[1,1])."""
    return ext._ivqrofn_negative(*args, **kwargs)


@extension(name='full', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_full_ext(*args, **kwargs) -> Fuzzarray:
    """Create an IVQROFN Fuzzarray filled with a specific Fuzznum."""
    return ext._ivqrofn_full(*args, **kwargs)


@extension(name='empty_like', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_empty_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty IVQROFN with the same shape as the input object."""
    return ext._ivqrofn_empty_like(*args, **kwargs)


@extension(name='positive_like', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_positive_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a positive IVQROFN with the same shape as the input object."""
    return ext._ivqrofn_positive_like(*args, **kwargs)


@extension(name='negative_like', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_negative_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a negative IVQROFN with the same shape as the input object."""
    return ext._ivqrofn_negative_like(*args, **kwargs)


@extension(name='full_like', mtype='ivqrofn', injection_type='top_level_function')
def ivqrofn_full_like_ext(*args, **kwargs) -> Fuzzarray:
    """Create an IVQROFN Fuzzarray filled with a specific value and matching input shape."""
    return ext._ivqrofn_full_like(*args, **kwargs)


# ========================= I/O Operation Extensions =========================

@extension(name='to_csv', mtype='ivqrofn', target_classes=['Fuzzarray'])
def ivqrofn_to_csv_ext(fuzz, *args, **kwargs) -> None:
    """Export IVQROFN Fuzzarray to CSV file."""
    return ext._ivqrofn_to_csv(fuzz, *args, **kwargs)


@extension(
    name='from_csv',
    mtype='ivqrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def ivqrofn_from_csv_ext(*args, **kwargs) -> Fuzzarray:
    """Read IVQROFN Fuzzarray from CSV file."""
    return ext._ivqrofn_from_csv(*args, **kwargs)


@extension(name='to_json', mtype='ivqrofn', target_classes=['Fuzzarray'])
def ivqrofn_to_json_ext(fuzz, *args, **kwargs) -> None:
    """Export IVQROFN Fuzzarray to JSON file."""
    return ext._ivqrofn_to_json(fuzz, *args, **kwargs)


@extension(
    name='from_json',
    mtype='ivqrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def ivqrofn_from_json_ext(*args, **kwargs) -> Fuzzarray:
    """Read IVQROFN Fuzzarray from JSON file."""
    return ext._ivqrofn_from_json(*args, **kwargs)


@extension(name='to_npy', mtype='ivqrofn', target_classes=['Fuzzarray'])
def ivqrofn_to_npy_ext(fuzz, *args, **kwargs) -> None:
    """Export IVQROFN Fuzzarray to NumPy .npy file."""
    return ext._ivqrofn_to_npy(fuzz, *args, **kwargs)


@extension(
    name='from_npy',
    mtype='ivqrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def ivqrofn_from_npy_ext(*args, **kwargs) -> Fuzzarray:
    """Read IVQROFN Fuzzarray from NumPy .npy file."""
    return ext._ivqrofn_from_npy(*args, **kwargs)


# ========================= Measurement Operation Extensions =========================

@extension(name='distance', mtype='ivqrofn')
def ivqrofn_distance_ext(fuzz_1, fuzz_2, p_l=2, indeterminacy=True) -> Union[float, np.ndarray]:
    """Calculate distance between two IVQROFN objects."""
    return ext._ivqrofn_distance(fuzz_1, fuzz_2, p_l=p_l, indeterminacy=indeterminacy)


# ========================= String Conversion Extensions =========================

@extension(
    name='str2fuzznum',
    mtype='ivqrofn',
    target_classes=['Fuzznum'],
    injection_type='top_level_function')
def ivqrofn_from_str_ext(fuzznum_str: str, q: int = 1) -> Fuzznum:
    """Convert string to IVQROFN Fuzznum."""
    return ext._ivqrofn_from_str(fuzznum_str, q)


# ========================= Aggregation Operation Extensions =========================

@extension(
    name='sum',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_sum_ext(fuzz: Union[Fuzznum, Fuzzarray],
                    axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the sum of IVQROFN along a specified axis."""
    return ext._ivqrofn_sum(fuzz, axis=axis)


@extension(
    name='mean',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_mean_ext(fuzz: Union[Fuzznum, Fuzzarray],
                     axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the mean of IVQROFN along a specified axis."""
    return ext._ivqrofn_mean(fuzz, axis=axis)


@extension(
    name='max',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_max_ext(fuzz: Union[Fuzznum, Fuzzarray],
                    axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the maximum of IVQROFN along a specified axis."""
    return ext._ivqrofn_max(fuzz, axis=axis)


@extension(
    name='min',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_min_ext(fuzz: Union[Fuzznum, Fuzzarray],
                    axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the minimum of IVQROFN along a specified axis."""
    return ext._ivqrofn_min(fuzz, axis=axis)


@extension(
    name='prod',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_prod_ext(fuzz: Union[Fuzznum, Fuzzarray],
                     axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the product of IVQROFN along a specified axis."""
    return ext._ivqrofn_prod(fuzz, axis=axis)


@extension(
    name='var',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_var_ext(fuzz: Union[Fuzznum, Fuzzarray],
                    axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the variance of IVQROFN along a specified axis."""
    return ext._ivqrofn_var(fuzz, axis=axis)


@extension(
    name='std',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def ivqrofn_std_ext(fuzz: Union[Fuzznum, Fuzzarray],
                    axis: int = None) -> Union[Fuzznum, Fuzzarray]:
    """Calculate the standard deviation of IVQROFN along a specified axis."""
    return ext._ivqrofn_std(fuzz, axis=axis)


# ========================= Property Access Extensions =========================

@extension(
    name='score',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def ivqrofn_score_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the score of IVQROFN using upper bounds."""
    return ext._ivqrofn_score(fuzz)


@extension(
    name='acc',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def ivqrofn_acc_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the accuracy of IVQROFN using upper bounds."""
    return ext._ivqrofn_acc(fuzz)


@extension(
    name='ind',
    mtype='ivqrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def ivqrofn_ind_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the indeterminacy of IVQROFN using upper bounds."""
    return ext._ivqrofn_ind(fuzz)