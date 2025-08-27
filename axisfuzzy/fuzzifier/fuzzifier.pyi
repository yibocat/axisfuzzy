#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 19:30
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Union, Optional, Any, List, Dict

import numpy as np

from ..membership import MembershipFunction
from ..core import Fuzznum, Fuzzarray

from .strategy import FuzzificationStrategy


class Fuzzifier:

    _init_mf: Union[MembershipFunction, str]
    _init_mtype: Optional[str]
    _init_method: Optional[str]
    _init_kwargs: Dict[str, Any]

    mtype: str
    method: str
    strategy: FuzzificationStrategy
    mf_cls: type[MembershipFunction]
    provided_mf_instance: Optional[MembershipFunction]
    mf_params_list: List[Dict[str, Any]]

    def __init__(self,
                 mf: Union[MembershipFunction, str],
                 mtype: Optional[str] = ...,
                 method: Optional[str] = ...,
                 **kwargs: Any): ...

    def __call__(self,
                 x: Union[float, int, list, np.ndarray]) -> Fuzznum | Fuzzarray: ...
    def __repr__(self): ...
    def get_config(self) -> Dict[str, Any]: ...
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Fuzzifier': ...
    def plot(self,
             x_range: tuple = ...,
             num_points: int = ...,
             show: bool = True): ...