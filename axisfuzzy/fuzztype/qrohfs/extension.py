#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/20 13:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import numpy as np

from . import ext
from ...extension import extension
from ...core import Fuzznum, Fuzzarray

# ========================= constructor =========================
@extension(name='empty', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_empty_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create an empty (uninitialized) QROHFN Fuzzarray."""
    return ext._qrohfn_empty(*args, **kwargs)


@extension(name='positive', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_poss_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create a QROHFN Fuzzarray filled with ones (md=1, nmd=0)."""
    return ext._qrohfn_poss(*args, **kwargs)


@extension(name='negative', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_negs_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create a QROHFN Fuzzarray filled with ones (md=0, nmd=1)."""
    return ext._qrohfn_negs(*args, **kwargs)


@extension(name='full', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_full_ext(*args, **kwargs) -> Fuzzarray:
    """Create a QROHFN Fuzzarray filled with a specific Fuzznum."""
    return ext._qrohfn_full(*args, **kwargs)


@extension(name='empty_like', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_empty_like_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create an empty QROHFN Fuzzarray with the same shape as the input Fuzznum or Fuzzarray."""
    return ext._qrohfn_empty_like(*args, **kwargs)


@extension(name='positive_like', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_poss_like_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create a QROHFN Fuzzarray filled with ones (md=1, nmd=0)
    with the same shape as the input Fuzznum or Fuzzarray."""
    return ext._qrohfn_poss_like(*args, **kwargs)


@extension(name='negative_like', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_negs_like_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Create a QROHFN Fuzzarray filled with ones (md=0, nmd=1)
    with the same shape as the input Fuzznum or Fuzzarray."""
    return ext._qrohfn_negs_like(*args, **kwargs)


@extension(name='full_like', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_full_like_ext(*args, **kwargs) -> Fuzzarray:
    """Create a QROHFN Fuzzarray filled with a specific Fuzznum
    with the same shape as the input Fuzznum or Fuzzarray."""
    return ext._qrohfn_full_like(*args, **kwargs)


# ========================= IO Operation =========================
@extension(name='to_csv', mtype='qrohfn', target_classes=['Fuzzarray'])
def qrohfn_to_csv_ext(fuzz, *args, **kwargs) -> None:
    """Export QROHFN Fuzzarray to CSV file."""
    return ext._qrohfn_to_csv(fuzz, *args, **kwargs)


@extension(
    name='read_csv',
    mtype='qrohfn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrohfn_from_csv_ext(*args, **kwargs) -> Fuzzarray:
    """Read QROHFN Fuzzarray from CSV file."""
    return ext._qrohfn_from_csv(*args, **kwargs)


@extension(name='to_json', mtype='qrohfn', target_classes=['Fuzzarray'])
def qrohfn_to_json_ext(fuzz, *args, **kwargs) -> None:
    """Export QROHFN Fuzzarray to JSON file."""
    return ext._qrohfn_to_json(fuzz, *args, **kwargs)


@extension(
    name='read_json',
    mtype='qrohfn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrohfn_from_json_ext(*args, **kwargs) -> Fuzzarray:
    """Read QROHFN Fuzzarray from JSON file."""
    return ext._qrohfn_from_json(*args, **kwargs)


@extension(name='to_npy', mtype='qrohfn', target_classes=['Fuzzarray'])
def qrohfn_to_npy_ext(fuzz, *args, **kwargs) -> None:
    """Export QROHFN Fuzzarray to NumPy .npy file."""
    return ext._qrohfn_to_npy(fuzz, *args, **kwargs)


@extension(
    name='read_npy',
    mtype='qrohfn',
    target_classes=['Fuzzarray'],
    injection_type='top_level_function')
def qrohfn_from_npy_ext(*args, **kwargs) -> Fuzzarray:
    """Read QROHFN Fuzzarray from NumPy .npy file."""
    return ext._qrohfn_from_npy(*args, **kwargs)


# ========================= Measurement Operations =========================

@extension(name='distance', mtype='qrohfn')
def qrohfn_distance_ext(fuzz_1, fuzz_2, p_l=2, tao=0.5, indeterminacy=True) -> float | np.ndarray:
    """Calculate distance between two QROHFN Fuzznum or Fuzzarray."""
    return ext._qrohfn_distance(fuzz_1, fuzz_2, gamma=p_l, tao=tao, indeterminacy=indeterminacy)


@extension(name='normalize', mtype='qrohfn', injection_type='top_level_function')
def qrohfn_normalize_ext(*args, **kwargs) -> Fuzznum | Fuzzarray:
    """Normalize QROHFN Fuzznum or Fuzzarray."""
    return ext._normalize_qrohfn(*args, **kwargs)


# ========================= String Conversion =========================

@extension(
    name='str2fuzznum',
    mtype='qrohfn',
    target_classes=['Fuzznum'],
    injection_type='top_level_function')
def qrohfn_from_str_ext(fuzznum_str: str, q: int = 1) -> Fuzznum:
    """Convert string to QROHFN Fuzznum."""
    return ext._qrohfn_from_str(fuzznum_str, q)


# ========================= Aggregation Operations =========================

@extension(
    name='sum',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_sum_ext(fuzz: Fuzznum | Fuzzarray,
                  axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the sum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_sum(fuzz, axis=axis)


@extension(
    name='mean',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_mean_ext(fuzz: Fuzznum | Fuzzarray,
                   axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the mean of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_mean(fuzz, axis=axis)


@extension(
    name='max',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_max_ext(fuzz: Fuzznum | Fuzzarray,
                  axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the maximum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_max(fuzz, axis=axis)


@extension(
    name='min',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_min_ext(fuzz: Fuzznum | Fuzzarray,
                  axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the minimum of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_min(fuzz, axis=axis)


@extension(
    name='prod',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_prod_ext(fuzz: Fuzznum | Fuzzarray,
                   axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the product of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_prod(fuzz, axis=axis)


@extension(
    name='var',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_var_ext(fuzz: Fuzznum | Fuzzarray,
                  axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the variance of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_var(fuzz, axis=axis)


@extension(
    name='std',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
)
def qrohfn_std_ext(fuzz: Fuzznum | Fuzzarray,
                  axis: int = None) -> Fuzznum | Fuzzarray:
    """Calculate the standard deviation of a QROFN Fuzzarray or Fuzznum along a specified axis."""
    return ext._qrohfn_std(fuzz, axis=axis)


@extension(
    name='score',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def qrohfn_score_ext(fuzz: Fuzznum | Fuzzarray) -> float | np.ndarray:
    """Calculate the score of a QROFN Fuzzarray or Fuzznum."""
    return ext._qrohfn_score(fuzz)


@extension(
    name='acc',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def qrohfn_acc_ext(fuzz: Fuzznum | Fuzzarray) -> float | np.ndarray:
    """Calculate the accuracy of a QROFN Fuzzarray or Fuzznum."""
    return ext._qrohfn_acc(fuzz)


@extension(
    name='ind',
    mtype='qrohfn',
    target_classes=['Fuzzarray', 'Fuzznum'],
    injection_type='instance_property')
def qrohfn_ind_ext(fuzz: Fuzznum | Fuzzarray) -> float | np.ndarray:
    """Calculate the indeterminacy of a QROFN Fuzzarray or Fuzznum."""
    return ext._qrohfn_ind(fuzz)
