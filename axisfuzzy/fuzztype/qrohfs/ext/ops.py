#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 21:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Callable, Union, Tuple

import numpy as np

from ....core import (
    OperationTNorm,
    get_registry_operation,
    Fuzznum,
    Fuzzarray,
    get_fuzztype_backend
)

from ..utils import _pairwise_combinations


def _reduce_sets_along_axis(sets: np.ndarray, reduce_func) -> np.ndarray:
    """
    Reduce object-array of hesitant sets along axis=0 using given binary reduction function.
    Each element on axis=0 is an array (hesitant set).
    """
    # sets shape: (n, ...) where n > 0, dtype=object,
    # each element is a 1D numpy array (hesitant set)
    n = sets.shape[0]
    out_shape = sets.shape[1:]
    result = np.empty(out_shape, dtype=object)

    for idx in np.ndindex(out_shape):
        acc = sets[0][idx]
        for i in range(1, n):
            acc = _pairwise_combinations(acc, sets[i][idx], reduce_func)
        result[idx] = acc
    return result


def _qrohfn_sum(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Sum for QROHFN arrays along axis"""
    if arr.size == 0:
        raise ValueError('Cannot sum empty array')
    if isinstance(arr, Fuzznum):
        return arr

    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    mds, nmds = arr.backend.get_component_arrays()

    if axis is None:
        # flatten
        md_res = mds.flat[0]
        nmd_res = nmds.flat[0]
        for i in range(1, arr.size):
            md_res = _pairwise_combinations(md_res, mds.flat[i], tnorm.t_conorm)
            nmd_res = _pairwise_combinations(nmd_res, nmds.flat[i], tnorm.t_norm)
        return Fuzznum('qrohfn', q=arr.q).create(md=md_res, nmd=nmd_res)

    # axis逻辑
    axis = int(axis)
    md_moved = np.moveaxis(mds, axis, 0)   # shape (n, ...)
    nmd_moved = np.moveaxis(nmds, axis, 0)

    md_out = _reduce_sets_along_axis(md_moved, tnorm.t_conorm)
    nmd_out = _reduce_sets_along_axis(nmd_moved, tnorm.t_norm)

    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls.from_arrays(md_out, nmd_out, q=arr.q)
    return Fuzzarray(backend=new_backend)


def _qrohfn_mean(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Mean for QROHFN array"""
    if arr.size == 0:
        return Fuzznum('qrohfn', q=arr.q).create()
    if isinstance(arr, Fuzznum):
        return arr

    if axis is None:
        n = arr.size
    else:
        if isinstance(axis, tuple):
            n = np.prod([arr.shape[a] for a in axis])
        else:
            n = arr.shape[axis]

    total = _qrohfn_sum(arr, axis=axis)
    return total / n


def _qrohfn_max(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Max for QROHFN array based on score = mean(md)^q - mean(nmd)^q."""
    if arr.size == 0:
        raise ValueError("max() of empty array")

    if isinstance(arr, Fuzznum):
        return arr

    mds, nmds = arr.backend.get_component_arrays()

    def score(md, nmd, q):
        return np.mean(md)**q - np.mean(nmd)**q

    if axis is None:
        scores = [score(mds.flat[i], nmds.flat[i], arr.q) for i in range(arr.size)]
        idx = np.argmax(scores)
        return arr.backend.get_fuzznum_view(np.unravel_index(idx, arr.shape))

    # axis case
    axis = int(axis)
    moved_mds = np.moveaxis(mds, axis, 0)  # shape (n, …)
    moved_nmds = np.moveaxis(nmds, axis, 0)

    out_shape = moved_mds.shape[1:]
    md_out = np.empty(out_shape, dtype=object)
    nmd_out = np.empty(out_shape, dtype=object)

    for idx in np.ndindex(out_shape):
        scores = [score(moved_mds[i][idx], moved_nmds[i][idx], arr.q) for i in range(moved_mds.shape[0])]
        j = int(np.argmax(scores))
        md_out[idx] = moved_mds[j][idx]
        nmd_out[idx] = moved_nmds[j][idx]

    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls.from_arrays(md_out, nmd_out, q=arr.q)
    return Fuzzarray(backend=new_backend)


def _qrohfn_min(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Min for QROHFN array based on score = mean(md)^q - mean(nmd)^q."""
    if arr.size == 0:
        raise ValueError("min() of empty array")

    if isinstance(arr, Fuzznum):
        return arr

    mds, nmds = arr.backend.get_component_arrays()

    def score(md, nmd, q):
        return np.mean(md)**q - np.mean(nmd)**q

    if axis is None:
        scores = [score(mds.flat[i], nmds.flat[i], arr.q) for i in range(arr.size)]
        idx = np.argmin(scores)
        return arr.backend.get_fuzznum_view(np.unravel_index(idx, arr.shape))

    # axis case
    axis = int(axis)
    moved_mds = np.moveaxis(mds, axis, 0)
    moved_nmds = np.moveaxis(nmds, axis, 0)

    out_shape = moved_mds.shape[1:]
    md_out = np.empty(out_shape, dtype=object)
    nmd_out = np.empty(out_shape, dtype=object)

    for idx in np.ndindex(out_shape):
        scores = [score(moved_mds[i][idx], moved_nmds[i][idx], arr.q) for i in range(moved_mds.shape[0])]
        j = int(np.argmin(scores))
        md_out[idx] = moved_mds[j][idx]
        nmd_out[idx] = moved_nmds[j][idx]

    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls.from_arrays(md_out, nmd_out, q=arr.q)
    return Fuzzarray(backend=new_backend)


def _qrohfn_prod(arr: Union[Fuzznum, Fuzzarray],
                 axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Product for QROHFN arrays along axis"""
    if arr.size == 0:
        raise ValueError("Cannot product empty array")
    if isinstance(arr, Fuzznum):
        return arr

    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    mds, nmds = arr.backend.get_component_arrays()

    if axis is None:
        md_res = mds.flat[0]
        nmd_res = nmds.flat[0]
        for i in range(1, arr.size):
            md_res = _pairwise_combinations(md_res, mds.flat[i], tnorm.t_norm)
            nmd_res = _pairwise_combinations(nmd_res, nmds.flat[i], tnorm.t_conorm)
        return Fuzznum('qrohfn', q=arr.q).create(md=md_res, nmd=nmd_res)

    axis = int(axis)
    md_moved = np.moveaxis(mds, axis, 0)
    nmd_moved = np.moveaxis(nmds, axis, 0)

    md_out = _reduce_sets_along_axis(md_moved, tnorm.t_norm)
    nmd_out = _reduce_sets_along_axis(nmd_moved, tnorm.t_conorm)

    backend_cls = get_fuzztype_backend('qrohfn')
    new_backend = backend_cls.from_arrays(md_out, nmd_out, q=arr.q)
    return Fuzzarray(backend=new_backend)


def _qrohfn_var(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Variance for QROHFN array along axis."""
    if arr.size == 0:
        raise ValueError("var() arg is empty")

    mean_val = _qrohfn_mean(arr, axis=axis)
    diff = arr - mean_val
    squared_diff = diff * diff
    return _qrohfn_mean(squared_diff, axis=axis)


def _qrohfn_std(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """Std deviation for QROHFN array along axis."""
    variance = _qrohfn_var(arr, axis=axis)
    return variance ** 0.5


def _qrohfn_score(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """Score for QROHFN: (mean(md))^q - (mean(nmd))^q"""
    if isinstance(arr, Fuzznum):
        return (np.mean(arr.md) ** arr.q - np.mean(arr.nmd) ** arr.q).item()

    mds, nmds = arr.backend.get_component_arrays()
    # 使用 frompyfunc 消除 Python-level 循环
    ufunc = np.frompyfunc(lambda md, nmd: np.mean(md)**arr.q - np.mean(nmd)**arr.q, 2, 1)
    return ufunc(mds, nmds).astype(np.float64)


def _qrohfn_acc(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """Accuracy for QROHFN: (mean(md))^q + (mean(nmd))^q"""
    if isinstance(arr, Fuzznum):
        return (np.mean(arr.md) ** arr.q + np.mean(arr.nmd) ** arr.q).item()

    mds, nmds = arr.backend.get_component_arrays()
    ufunc = np.frompyfunc(lambda md, nmd: np.mean(md)**arr.q + np.mean(nmd)**arr.q, 2, 1)
    return ufunc(mds, nmds).astype(np.float64)


def _qrohfn_ind(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """Indeterminacy for QROHFN: 1 - (mean(md)^q + mean(nmd)^q)"""
    if isinstance(arr, Fuzznum):
        return (1.0 - (np.mean(arr.md) ** arr.q + np.mean(arr.nmd) ** arr.q)).item()

    mds, nmds = arr.backend.get_component_arrays()
    ufunc = np.frompyfunc(lambda md, nmd: 1.0 - (np.mean(md)**arr.q + np.mean(nmd)**arr.q), 2, 1)
    return ufunc(mds, nmds).astype(np.float64)
