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
    """
    Parse strings of the form "<md,nmd>" into q-rung orthopair fuzzy numbers (QROFNs).

    :param fuzznum_str: The string representation of the QROFN.
    :param q: The q-rung level (default is 1).
    :return: A Fuzznum object representing the QROFN.
    """
    return factory._qrofn_from_str(fuzznum_str, q)
