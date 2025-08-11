#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 18:01
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union, Tuple

import numpy as np

from ....core import OperationTNorm
from ....core import get_operation_registry
from ....core import Fuzznum
from ....core.t_fuzzarray import Fuzzarray


def _qrofn_sum(arr: Union[Fuzznum, Fuzzarray],
              axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]:
    """
    High-performance sum for QROFN Fuzzarray using t-norm/t-conorm reduction.
    """
    if arr.size == 0:
        # Additive identity for QROFN is <0, 1>
        return Fuzznum('qrofn', q=arr.q).create(md=0.0, nmd=1.0)

    if arr.size == 1:
        # If the array has only one element, return it directly
        if isinstance(arr, Fuzznum):
            return arr
        else:
            return arr.backend.get_fuzznum_view(0)

    op_registry = get_operation_registry()
    norm_type, params = op_registry.get_default_t_norm_config()
    tnorm = OperationTNorm(norm_type=norm_type, q=arr.q, **params)

    mds, nmds = arr.backend.get_component_arrays()

    # TODO: t_conorm.reduce 和 t_norm.reduce 需要实现
    md_sum = tnorm.t_conorm.reduce(mds, axis=axis)
    nmd_sum = tnorm.t_norm.reduce(nmds, axis=axis)

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
        # TODO: 这里的 .flatten 方法可能有问题, Fuzzarray 并没实现该方法
        #  在 Mixin 中实现该方法,通过内部调用该方法
        return arr.flatten()[indices]
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
        # TODO: 这里的 .flatten 方法可能有问题, Fuzzarray 并没实现该方法
        #  在 Mixin 中实现该方法,通过内部调用该方法
        return arr.flatten()[indices]
    else:
        if isinstance(axis, tuple):
            raise NotImplementedError("min with tuple axis is not yet supported.")

        grid = np.indices(indices.shape)
        idx_list = list(grid)
        idx_list.insert(axis, indices)

        return arr[tuple(idx_list)]


# TODO: 以下方法待实现
def _qrofn_prod(arr: Union[Fuzznum, Fuzzarray],
                axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]: ...


def _qrofn_var(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]: ...


def _qrofn_std(arr: Union[Fuzznum, Fuzzarray],
               axis: Union[int, Tuple[int, ...]] = None) -> Union[Fuzznum, Fuzzarray]: ...


def _qrofn_score(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]: ...


def _qrofn_acc(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]: ...


def _qrofn_ind(arr: Union[Fuzznum, Fuzzarray]) -> Union[float, np.ndarray]: ...



















