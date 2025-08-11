#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 10:40
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union

import numpy as np

from . import factory

from ...extension import extension

from ...core.fuzznums import Fuzznum
from ...core.t_fuzzarray import Fuzzarray


# ========================= constructor =========================

@extension(name='empty', mtype='qrofn', injection_type='top_level_function')
def qrofn_empty_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty (uninitialized) QROFN Fuzzarray."""
    return factory._qrofn_empty(*args, **kwargs)


@extension(name='positive', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with ones (md=1, nmd=0)."""
    return factory._qrofn_poss(*args, **kwargs)


@extension(name='negative', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with ones (md=0, nmd=1)."""
    return factory._qrofn_negs(*args, **kwargs)


@extension(name='full', mtype='qrofn', injection_type='top_level_function')
def qrofn_full_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with a specific Fuzznum."""
    return factory._qrofn_full(*args, **kwargs)


@extension(name='empty_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_empty_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create an empty QROFN Fuzzarray with the same shape as the input Fuzznum or Fuzzarray."""
    return factory._qrofn_empty_like(*args, **kwargs)


@extension(name='positive_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with ones (md=1, nmd=0)
    with the same shape as the input Fuzznum or Fuzzarray."""
    return factory._qrofn_poss_like(*args, **kwargs)


@extension(name='negative_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with ones (md=0, nmd=1)
    with the same shape as the input Fuzznum or Fuzzarray."""
    return factory._qrofn_negs_like(*args, **kwargs)


@extension(name='full_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_full_like_ext(*args, **kwargs) -> Union[Fuzzarray, Fuzznum]:
    """Create a QROFN Fuzzarray filled with a specific Fuzznum
    with the same shape as the input Fuzznum or Fuzzarray."""
    return factory._qrofn_full_like(*args, **kwargs)


# ========================= IO Operation =========================

@extension(name='to_csv', mtype='qrofn', target_classes=['Fuzzarray'])
def qrofn_to_csv_ext(fuzz, *args, **kwargs):
    """Save a QROFN Fuzzarray to a CSV file."""
    return factory._qrofn_to_csv(fuzz, *args, **kwargs)


@extension(
    name='read_csv',
    mtype='qrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrofn_from_csv_ext(*args, **kwargs) -> Fuzzarray:
    """Load a QROFN Fuzzarray from a CSV file."""
    return factory._qrofn_from_csv(*args, **kwargs)


@extension(name='to_json', mtype='qrofn', target_classes=['Fuzzarray'])
def qrofn_to_json_ext(fuzz, *args, **kwargs):
    """Save a QROFN Fuzzarray to a JSON file."""
    return factory._qrofn_to_json(fuzz, *args, **kwargs)


@extension(
    name='read_json',
    mtype='qrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrofn_from_json_ext(*args, **kwargs) -> Fuzzarray:
    """Load a QROFN Fuzzarray from a JSON file."""
    return factory._qrofn_from_json(*args, **kwargs)


@extension(name='to_npy', mtype='qrofn', target_classes=['Fuzzarray'])
def qrofn_to_npy_ext(fuzz, *args, **kwargs):
    """Save a QROFN Fuzzarray to a NumPy binary file."""
    return factory._qrofn_to_npy(fuzz, *args, **kwargs)


@extension(
    name='read_npy',
    mtype='qrofn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrofn_from_npy_ext(*args, **kwargs) -> Fuzzarray:
    """Load a QROFN Fuzzarray from a NumPy binary file."""
    return factory._qrofn_from_npy(*args, **kwargs)


# ========================= Measurement Operations =========================

@extension(name='distance', mtype='qrofn')
def qrofn_distance_ext(fuzz_1, fuzz_2,
                       p_l=2, indeterminacy=True) -> Union[np.ndarray, float]:
    """Calculate the distance between two QROFN Fuzzarrays."""
    return factory._qrofn_distance(fuzz_1, fuzz_2, p_l=p_l, indeterminacy=indeterminacy)


# ========================= String Conversion =========================

@extension(
    name='str2fuzznum',
    mtype='qrofn',
    target_classes=['Fuzznum'],
    injection_type='top_level_function')
def qrofn_from_str_ext(fuzznum_str: str, q: int = 1) -> Fuzznum:
    """Convert a string representation of a QROFN Fuzznum to an actual Fuzznum."""
    return factory._qrofn_from_str(fuzznum_str, q)


# ========================= Aggregation Operations =========================


@extension(
    name='sum',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_sum_ext(fuzz: Union[Fuzzarray, Fuzznum],
                  axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the sum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_sum(fuzz, axis=axis)


@extension(
    name='mean',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_mean_ext(fuzz: Union[Fuzzarray, Fuzznum],
                   axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the mean of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_mean(fuzz, axis=axis)


@extension(
    name='max',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_max_ext(fuzz: Union[Fuzzarray, Fuzznum],
                  axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the maximum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_max(fuzz, axis=axis)


@extension(
    name='min',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_min_ext(fuzz: Union[Fuzzarray, Fuzznum],
                  axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the minimum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_min(fuzz, axis=axis)


@extension(
    name='prod',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_prod_ext(fuzz: Union[Fuzzarray, Fuzznum],
                   axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the product of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_prod(fuzz, axis=axis)


@extension(
    name='var',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_var_ext(fuzz: Union[Fuzzarray, Fuzznum],
                  axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the variance of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_var(fuzz, axis=axis)


@extension(
    name='std',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrofn_std_ext(fuzz: Union[Fuzzarray, Fuzznum],
                  axis: int = None) -> Union[Fuzzarray, Fuzznum]:
    """Calculate the standard deviation of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return factory._qrofn_std(fuzz, axis=axis)


@extension(
    name='score',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
@property
def qrofn_score_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the score of a QROFN Fuzzarray or Fuzznum."""
    return factory._qrofn_score(fuzz)


@extension(
    name='acc',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
@property
def qrofn_acc_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the accuracy of a QROFN Fuzzarray or Fuzznum."""
    return factory._qrofn_acc(fuzz)


@extension(
    name='ind',
    mtype='qrofn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
@property
def qrofn_ind_ext(fuzz: Union[Fuzzarray, Fuzznum]) -> Union[float, np.ndarray]:
    """Calculate the indeterminacy of a QROFN Fuzzarray or Fuzznum."""
    return factory._qrofn_ind(fuzz)
