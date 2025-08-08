#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 19:08
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Union

from ....core import Fuzznum, Fuzzarray, fuzzarray
from ....extension import extension


@extension(name='zeros', mtype='qrofn', injection_type='top_level_function')
def qrofn_zeros(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', qrung=q).create(md=0., nmd=0.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)


@extension(name='negs', mtype='qrofn', injection_type='top_level_function')
def qrofn_negs(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', qrung=q).create(md=0., nmd=1.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)


@extension(name='poss', mtype='qrofn', injection_type='top_level_function')
def qrofn_poss(*shape, q: int = 1) -> Union[Fuzznum, Fuzzarray]:
    fuzz_num = Fuzznum('qrofn', qrung=q).create(md=1., nmd=0.)

    if not shape:
        return fuzz_num
    else:
        return fuzzarray(fuzz_num, shape=shape, copy=False)
