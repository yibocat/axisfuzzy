#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/11 10:37
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Tuple, Union

from ....core.registry import get_backend
from ....core.fuzznums import Fuzznum
from ....core.t_fuzzarray import Fuzzarray


def _qrofn_empty(shape: Tuple[int, ...], q: int) -> Union[Fuzzarray, Fuzznum]:

    if not shape:
        return Fuzznum(mtype='qrofn', q=q).create(md=0., nmd=0.)

    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    return Fuzzarray(backend=backend)


def _qrofn_poss(shape: Tuple[int, ...], q: int) -> Union[Fuzzarray, Fuzznum]:
    if not shape:
        return Fuzznum(mtype='qrofn', q=q).create(md=1.0, nmd=0.0)

    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(1.0, 0.0)
    return Fuzzarray(backend=backend)


def _qrofn_negs(shape: Tuple[int, ...], q: int) -> Union[Fuzzarray, Fuzznum]:
    if not shape:
        return Fuzznum(mtype='qrofn', q=q).create(md=0.0, nmd=1.0)

    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(0.0, 1.0)
    return Fuzzarray(backend=backend)


def _qrofn_full(shape: Tuple[int, ...], fill_value: Fuzznum, q: int) -> Fuzzarray:
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'qrofn':
        raise TypeError("fill_value must be a QROFN Fuzznum.")
    if fill_value.q != q:
        raise ValueError(f"Q-rung mismatch: array q is {q}, but fill_value q is {fill_value.q}")

    backend_cls = get_backend('qrofn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(fill_value.md, fill_value.nmd)
    return Fuzzarray(backend=backend)


def _qrofn_empty_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    if isinstance(fuzznum, Fuzzarray):
        shape = fuzznum.shape
        q = fuzznum.q
    else:
        shape = ()
        q = fuzznum.q

    return _qrofn_empty(shape=shape, q=q)


def _qrofn_poss_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    if isinstance(fuzznum, Fuzzarray):
        shape = fuzznum.shape
        q = fuzznum.q
    else:
        shape = ()
        q = fuzznum.q

    return _qrofn_poss(shape=shape, q=q)


def _qrofn_negs_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    if isinstance(fuzznum, Fuzzarray):
        shape = fuzznum.shape
        q = fuzznum.q
    else:
        shape = ()
        q = fuzznum.q

    return _qrofn_negs(shape=shape, q=q)


def _qrofn_full_like(fuzznum: Union[Fuzznum, Fuzzarray], fill_value: Fuzznum) -> Fuzzarray:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'qrofn':
        raise TypeError("fill_value must be a QROFN Fuzznum.")

    if isinstance(fuzznum, Fuzzarray):
        shape = fuzznum.shape
        q = fuzznum.q
    else:
        shape = ()
        q = fuzznum.q

    return _qrofn_full(shape=shape, fill_value=fill_value, q=q)
