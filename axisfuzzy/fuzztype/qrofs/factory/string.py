#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import re

from ....core import Fuzznum

# 匹配数字，包括整数、小数、科学计数法
_NUMBER = r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

# 匹配形如 <md, nmd> 的字符串，支持前后空格
_PATTERN = re.compile(
    rf'^\s*<\s*({_NUMBER})\s*,\s*({_NUMBER})\s*>\s*$'
)


def _qrofn_from_str(fuzznum_str: str, q: int = 1) -> Fuzznum:
    if not isinstance(fuzznum_str, str):
        raise TypeError(f"fuzznum_str must be a str. got '{type(fuzznum_str).__name__}'")
    if not isinstance(q, int) or q < 0:
        raise ValueError(f"'q' must be a non-negative integer. got '{q!r}'")

    m = _PATTERN.match(fuzznum_str)
    if not m:
        raise ValueError(f"Format error: "
                         f"Cannot parse string: {fuzznum_str!r}.")

    md = float(m.group(1))
    nmd = float(m.group(2))

    if not (0.0 <= md <= 1.0 and 0.0 <= nmd <= 1.0):
        raise ValueError(f"Value out of bounds: md={md}, nmd={nmd} must be within [0,1]")
    if md ** q + nmd ** q > 1 + 1e-12:
        raise ValueError(f"Violation of q-rung constraint: md^q + nmd^q = {md**q + nmd**q:.6f} > 1 (q={q})")

    return Fuzznum('qrofn', q=q).create(md=md, nmd=nmd)
