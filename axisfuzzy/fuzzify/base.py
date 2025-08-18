#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

import numpy as np

from ..config import get_config
from ..core import Fuzznum, Fuzzarray
from ..membership import MembershipFunction


class FuzzificationStrategy(ABC):

    mtype: Optional[str] = None
    method: Optional[str] = None

    def __init__(self, q: Optional[int] = None, **kwargs: Any):
        """
        初始化模糊化策略

        Args:
            q: 可选的参数，通常用于q-rung模糊数的策略
            **kwargs: 策略特定的参数
        """
        self.q = q if q is not None else get_config().DEFAULT_Q
        self.kwargs = kwargs

    @abstractmethod
    def fuzzify_scalar(self,
                       x: Optional[float],
                       mf: Optional[MembershipFunction] = None) -> 'Fuzznum':
        """
        对单个标量值进行模糊化

        Args:
            x: 输入的精确值（某些策略可能不需要此参数，如expert策略）
            mf: 隶属函数对象（某些策略可能不需要此参数）

        Returns:
            模糊化后的Fuzznum对象
        """
        pass

    @abstractmethod
    def fuzzify_array(self,
                      x: Optional[np.ndarray],
                      mf: Optional[MembershipFunction] = None) -> 'Fuzzarray':
        """
        对数组进行批量模糊化（高性能版本）

        Args:
            x: 输入的精确值数组
            mf: 隶属函数对象

        Returns:
            模糊化后的Fuzzarray对象
        """
        pass

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息

        Returns:
            包含策略元信息的字典
        """
        return {
            'mtype': self.mtype,
            'method': self.method,
            'q': self.q,
            'kwargs': self.kwargs,
            'class_name': self.__class__.__name__
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mtype='{self.mtype}', method='{self.method}')"
