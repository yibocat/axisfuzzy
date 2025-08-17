#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


from typing import Optional, Union, Any

import numpy as np

from ..core import Fuzzarray, Fuzznum
from ..membership import MembershipFunction
from .base import FuzzificationStrategy

class Fuzzifier:
    """
    一个可配置、可复用的模糊化引擎。
    """
    # --- Public Instance Attributes ---
    mtype: str
    method: str
    strategy: FuzzificationStrategy
    mf: MembershipFunction

    def __init__(self,
                 mf: Union[MembershipFunction, str],
                 mtype: Optional[str] = ...,
                 method: Optional[str] = ...,
                 **kwargs: Any) -> None: ...

    def __call__(self, x: Union[float, int, list, np.ndarray]) -> Union[Fuzznum, Fuzzarray]: ...


def fuzzify(
        x: Union[float, int, list, np.ndarray],
        mf: Union[MembershipFunction, str],
        mtype: Optional[str] = ...,
        method: Optional[str] = ...,
        **kwargs: Any
) -> Union[Fuzznum, Fuzzarray]: ...