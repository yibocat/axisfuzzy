#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 18:01
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union, Tuple

import numpy as np

from ....core import OperationTNorm, get_registry_operation, Fuzznum, Fuzzarray


def _qrofn_sum(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance sum for QROFN Fuzzarray using t-norm/t-conorm reduction.
    """
    if arr.size == 0:
        raise ValueError('Cannot sum empty array')

    if arr.size == 1:
        # If the array has only one element, return it directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    mds, nmds = arr.backend.get_component_arrays()

    md_sum = tnorm.t_conorm_reduce(mds, axis=axis)
    nmd_sum = tnorm.t_norm_reduce(nmds, axis=axis)

    if axis is None:
        return Fuzznum('qrofn', q=arr.q).create(md=md_sum, nmd=nmd_sum)
    else:
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(md_sum, nmd_sum, q=arr.q)
        return Fuzzarray(backend=new_backend)


def _qrofn_mean(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance mean for QROFN Fuzzarray using t-norm/t-conorm reduction.
    """
    if arr.size == 0:
        return Fuzznum('qrofn', q=arr.q).create()
    if arr.size == 1:
        # If the array has only one element, return it directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    if axis is None:
        n = arr.size
    else:
        n = np.prod([arr.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

    total = _qrofn_sum(arr, axis=axis)
    return total / n


def _qrofn_max(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance max for QROFN Fuzzarray based on score.
    """
    if arr.size == 0:
        raise ValueError("max() arg is an empty sequence")

    if arr.size == 1:
        # If the array has only one element, return it directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    mds, nmds = arr.backend.get_component_arrays()
    scores = mds ** arr.q - nmds ** arr.q
    indices = np.argmax(scores, axis=axis)

    if axis is None:
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]

    else:
        # This is a simplified implementation for single-axis max.
        # A fully robust multi-axis implementation is more complex.
        if isinstance(axis, tuple):
            raise NotImplementedError("max with tuple axis is not yet supported.")

        # Create a grid of indices to select from the original array
        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _qrofn_min(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance min for QROFN Fuzzarray based on score.
    """
    if arr.size == 0:
        raise ValueError("min() arg is an empty sequence")

    if arr.size == 1:
        # If the array has only one element, return it directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    mds, nmds = arr.backend.get_component_arrays()
    scores = mds ** arr.q - nmds ** arr.q
    indices = np.argmin(scores, axis=axis)

    if axis is None:
        from ....mixin.factory import _flatten_factory
        return _flatten_factory(arr)[indices]
    else:
        if isinstance(axis, tuple):
            raise NotImplementedError("min with tuple axis is not yet supported.")

        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


def _qrofn_prod(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance product for QROFN Fuzzarray using t-norm/t-conorm reduction.
    """
    if arr.size == 0:
        raise ValueError('Cannot product empty array')

    if arr.size == 1:
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    op_registry = get_registry_operation()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    mds, nmds = arr.backend.get_component_arrays()

    # 对于乘积运算，使用 t-norm 和 t-conorm
    md_prod = tnorm.t_norm_reduce(mds, axis=axis)
    nmd_prod = tnorm.t_conorm_reduce(nmds, axis=axis)

    if axis is None:
        return Fuzznum('qrofn', q=arr.q).create(md=md_prod, nmd=nmd_prod)
    else:
        backend_cls = arr.backend.__class__
        new_backend = backend_cls.from_arrays(md_prod, nmd_prod, q=arr.q)
        return Fuzzarray(backend=new_backend)


def _qrofn_var(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance variance for QROFN Fuzzarray.
    计算方差：Var(X) = E[(X - E[X])²]
    """
    if arr.size == 0:
        raise ValueError("var() arg is an empty sequence")

    if arr.size == 1:
        return Fuzznum('qrofn', q=arr.q).create(md=0.0, nmd=1.0)

    # 先计算均值
    mean_val = _qrofn_mean(arr, axis=axis)

    # 计算每个元素与均值的差
    diff = arr - mean_val

    # 计算差的平方（使用乘法）
    squared_diff = diff * diff

    # 计算平方差的均值
    return _qrofn_mean(squared_diff, axis=axis)


def _qrofn_std(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance standard deviation for QROFN Fuzzarray.
    标准差是方差的平方根
    """
    if arr.size == 0:
        raise ValueError("std() arg is an empty sequence")

    if arr.size == 1:
        return Fuzznum('qrofn', q=arr.q).create(md=0.0, nmd=1.0)

    variance = _qrofn_var(arr, axis=axis)

    # 对方差开平方根
    return variance ** 0.5


def _qrofn_score(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance score calculation for QROFN: md^q - nmd^q
    """
    if isinstance(arr, Fuzznum):
        # 单个 Fuzznum 的得分值
        return arr.md ** arr.q - arr.nmd ** arr.q

    # Fuzzarray 的向量化计算
    mds, nmds = arr.backend.get_component_arrays()
    scores = mds ** arr.q - nmds ** arr.q
    return scores


def _qrofn_acc(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance accuracy calculation for QROFN: md^q + nmd^q
    """
    if isinstance(arr, Fuzznum):
        # 单个 Fuzznum 的精确度
        return arr.md ** arr.q + arr.nmd ** arr.q

    # Fuzzarray 的向量化计算
    mds, nmds = arr.backend.get_component_arrays()
    accuracy = mds ** arr.q + nmds ** arr.q
    return accuracy


def _qrofn_ind(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]:
    """
    High-performance indeterminacy calculation for QROFN: 1 - acc = 1 - (md^q + nmd^q)
    """
    if isinstance(arr, Fuzznum):
        # 单个 Fuzznum 的不确定度
        acc = arr.md ** arr.q + arr.nmd ** arr.q
        return 1.0 - acc

    # Fuzzarray 的向量化计算
    mds, nmds = arr.backend.get_component_arrays()
    accuracy = mds ** arr.q + nmds ** arr.q
    indeterminacy = 1.0 - accuracy
    return indeterminacy
