#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 19:08
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union

from ....core import Fuzznum, Fuzzarray, fuzzarray
from ....extension import extension
from ....utils import experimental


@extension(name='zeros', mtype='qrofn', injection_type='top_level_function')
def qrofn_zeros(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', q=q).create(md=0., nmd=0.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)


@extension(name='negs', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', q=q).create(md=0., nmd=1.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)


@extension(name='poss', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', q=q).create(md=1., nmd=0.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)


@extension(name='full', mtype='qrofn', injection_type='top_level_function')
def qrofn_full(fuzznum: Fuzznum, *shape) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, Fuzznum):
        raise TypeError('fuzznum must be an instance of Fuzznum')

    if not shape:
        return fuzznum
    else:
        return fuzzarray(fuzznum, shape=shape, copy=False)


@extension(name='zeros_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_zeros_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    fuzz_num = Fuzznum('qrofn', q=fuzznum.q).create(md=0., nmd=0.)

    return fuzzarray(fuzz_num, shape=fuzznum.shape, copy=False) if isinstance(fuzznum, Fuzzarray) else fuzz_num


@extension(name='poss_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    fuzz_num = Fuzznum('qrofn', q=fuzznum.q).create(md=1., nmd=0.)

    return fuzzarray(fuzz_num, shape=fuzznum.shape, copy=False) if isinstance(fuzznum, Fuzzarray) else fuzz_num


@extension(name='negs_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs_like(fuzznum: Union[Fuzznum, Fuzzarray]) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')

    fuzz_num = Fuzznum('qrofn', q=fuzznum.q).create(md=0., nmd=1.)

    return fuzzarray(fuzz_num, shape=fuzznum.shape, copy=False) if isinstance(fuzznum, Fuzzarray) else fuzz_num


@extension(name='full_like', mtype='qrofn', injection_type='top_level_function')
def qrofn_full_like(fuzznum: Union[Fuzznum, Fuzzarray], y: Fuzznum) -> Union[Fuzznum, Fuzzarray]:
    if not isinstance(fuzznum, (Fuzznum, Fuzzarray)):
        raise TypeError('fuzznum must be an instance of Fuzznum or Fuzzarray')
    if not isinstance(y, Fuzznum):
        raise TypeError('y must be an instance of Fuzznum')

    return fuzzarray(y, shape=fuzznum.shape, copy=False) if isinstance(fuzznum, Fuzzarray) else fuzznum


@experimental
@extension(name='fuzznum', mtype='qrofn', injection_type='top_level_function')
def qrofn_fuzznum(md: float, nmd: float, q: int = 1) -> Fuzznum:
    return Fuzznum('qrofn', q=q).create(md=md, nmd=nmd)
