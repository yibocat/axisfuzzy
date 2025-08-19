#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 23:33
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union

import numpy as np

from ..config import get_config
from ..core import Fuzznum, Fuzzarray
from ..membership import MembershipFunction


class FuzzificationStrategy(ABC):
    """
    模糊化策略基类
    - 每个具体策略实现 fuzzify(x, mf_cls, mf_params_list)
    - 输入 x 可以是 scalar/list/ndarray
    """

    mtype: Optional[str] = None
    method: Optional[str] = None

    def __init__(self, q: Optional[int] = None, **kwargs: Any):
        self.q = q if q is not None else get_config().DEFAULT_Q
        self.kwargs = kwargs


    @abstractmethod
    def fuzzify(self,
                x: Union[float, int, list, np.ndarray],
                mf_cls: type,
                mf_params_list: List[Dict]) -> Union[Fuzznum, Fuzzarray]:
        """
        模糊化输入数据

        Args:
            x: 输入 (float/int, list, ndarray)
            mf_cls: 隶属函数类
            mf_params_list: list of dict (隶属函数参数集)

        Returns:
            单个 Fuzznum 或 Fuzzarray
        """
        pass

    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'mtype': self.mtype,
            'method': self.method,
            'q': self.q,
            'kwargs': self.kwargs,
            'class_name': self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(mtype='{self.mtype}', method='{self.method}')")
