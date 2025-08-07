#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 16:44
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import random
from typing import Union

import numpy as np

from ...core import Fuzznum, Fuzzarray
from ...extension import extension


@extension(
    name='distance',
    mtype='qrofn',
    target_classes=['Fuzznum', 'Fuzzarray'],
    injection_type='both'
)
def qrofn_distance(fuzz1: Fuzznum, fuzz2: Fuzznum, p: int = 2) -> float:
    q = fuzz1.q

    md_diff = abs(fuzz1.md ** q - fuzz2.md ** q) ** p
    nmd_diff = abs(fuzz1.nmd ** q - fuzz2.nmd ** q) ** p
    return ((md_diff + nmd_diff) / 2) ** (1 / p)

