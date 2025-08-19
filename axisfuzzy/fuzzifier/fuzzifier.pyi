#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 19:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Union, Optional, Any, List

import numpy as np

from ..membership import MembershipFunction
from ..core import Fuzznum, Fuzzarray

from .strategy import FuzzificationStrategy


class Fuzzifier:

    mtype: str
    method: str
    strategy: FuzzificationStrategy
    mf_cls: type[MembershipFunction]
    mf_params_list: List

    def __init__(self,
                 mf: Union[MembershipFunction, str],
                 mtype: Optional[str] = ...,
                 method: Optional[str] = ...,
                 **kwargs: Any): ...

    def __call__(self,
                 x: Union[float, int, list, np.ndarray]) -> Fuzznum | Fuzzarray: ...
    def __repr__(self): ...
    def plot(self,
             x_range: tuple = ...,
             num_points: int = ...,
             show: bool = True): ...