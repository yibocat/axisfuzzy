#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 10:37
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Tuple

from ....core.registry import get_backend
from ....core.fuzznums import Fuzznum
from ....core.t_fuzzarray import Fuzzarray


def qrofn_empty(shape: Tuple[int, ...], q: int) -> Fuzzarray:
    """
    Create an empty (uninitialized) QROFN Fuzzarray of a given shape.

    Args:
        shape: The shape of the array.
        q: The q-rung parameter.

    Returns:
        An uninitialized Fuzzarray of the given shape.
    """
    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    return Fuzzarray(backend=backend)


def qrofn_poss(shape: Tuple[int, ...], q: int) -> Fuzzarray:
    """
    Create a QROFN Fuzzarray of a given shape, filled with ones (md=1, nmd=0).

    Args:
        shape: The shape of the array.
        q: The q-rung parameter.

    Returns:
        A Fuzzarray filled with the identity element.
    """
    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(1.0, 0.0)
    return Fuzzarray(backend=backend)


def qrofn_negs(shape: Tuple[int, ...], q: int) -> Fuzzarray:
    """
    Create a QROFN Fuzzarray of a given shape, filled with ones (md=1, nmd=0).

    Args:
        shape: The shape of the array.
        q: The q-rung parameter.

    Returns:
        A Fuzzarray filled with the identity element.
    """
    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(0.0, 1.0)
    return Fuzzarray(backend=backend)


def qrofn_full(shape: Tuple[int, ...], fill_value: Fuzznum, q: int) -> Fuzzarray:
    """
    Create a QROFN Fuzzarray of a given shape, filled with a specific Fuzznum value.

    Args:
        shape: The shape of the array.
        fill_value: The Fuzznum to fill the array with.
        q: The q-rung parameter.

    Returns:
        A Fuzzarray filled with the specified value.
    """
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'qrofn':
        raise TypeError("fill_value must be a QROFN Fuzznum.")
    if fill_value.q != q:
        raise ValueError(f"Q-rung mismatch: array q is {q}, but fill_value q is {fill_value.q}")

    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(fill_value.md, fill_value.nmd)
    return Fuzzarray(backend=backend)
