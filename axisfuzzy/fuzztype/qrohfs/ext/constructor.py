#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 20:16
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Optional, Union, Tuple
from ....core import Fuzznum, Fuzzarray, get_registry_fuzztype


def _qrohfn_empty(shape: Optional[Tuple[int, ...]] = None,
                  q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    q = 1 if q is None else q
    if shape is None:
        return Fuzznum(mtype='qrohfn', q=q).create(md=[0.], nmd=[0.])

    backend_cls = get_registry_fuzztype().get_backend('qrohfn')
    backend = backend_cls(shape=shape, q=q)
    return Fuzzarray(backend=backend)


def _qrohfn_poss(shape: Optional[Tuple[int, ...]] = None,
                 q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    q = 1 if q is None else q
    if shape is None:
        return Fuzznum(mtype='qrohfn', q=q).create(md=[1.0], nmd=[0.0])

    backend_cls = get_registry_fuzztype().get_backend('qrohfn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(1.0, 0.0)
    return Fuzzarray(backend=backend)


def _qrohfn_negs(shape: Optional[Tuple[int, ...]] = None,
                 q: Optional[int] = None) -> Union[Fuzzarray, Fuzznum]:
    q = 1 if q is None else q
    if shape is None:
        return Fuzznum(mtype='qrohfn', q=q).create(md=[0.0], nmd=[1.0])

    backend_cls = get_registry_fuzztype().get_backend('qrohfn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(0.0, 1.0)
    return Fuzzarray(backend=backend)


def _qrohfn_full(fill_value: Fuzznum,
                 shape: Tuple[int, ...],
                 q: Optional[int] = None) -> Fuzzarray:
    q = 1 if q is None else q

    if not shape or shape is None:
        raise ValueError("shape must be a non-empty tuple of integers.")
    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'qrohfn':
        raise TypeError("fill_value must be a QROHFN Fuzznum.")
    if fill_value.q != q:
        raise ValueError(f"Q-rung mismatch: array q is {q}, but fill_value q is {fill_value.q}")

    backend_cls = get_registry_fuzztype().get_backend('qrohfn')
    backend = backend_cls(shape=shape, q=q)
    backend.fill_from_values(fill_value.md, fill_value.nmd)
    return Fuzzarray(backend=backend)


def _qrohfn_empty_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = ()
        q = obj.q

    return _qrohfn_empty(shape=shape, q=q)


def _qrohfn_poss_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = ()
        q = obj.q

    return _qrohfn_poss(shape=shape, q=q)


def _qrohfn_negs_like(obj: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = ()
        q = obj.q

    return _qrohfn_negs(shape=shape, q=q)


def _qrohfn_full_like(fill_value: Fuzznum, obj: Union[Fuzznum, Fuzzarray]) -> Fuzzarray:
    if not isinstance(obj, (Fuzznum, Fuzzarray)):
        raise TypeError('obj must be an instance of Fuzznum or Fuzzarray')

    if not isinstance(fill_value, Fuzznum) or fill_value.mtype != 'qrohfn':
        raise TypeError("fill_value must be a QROHFN Fuzznum.")

    if isinstance(obj, Fuzzarray):
        shape = obj.shape
        q = obj.q
    else:
        shape = ()
        q = obj.q

    return _qrohfn_full(shape=shape, fill_value=fill_value, q=q)
