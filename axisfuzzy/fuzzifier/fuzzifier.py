#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 12:17
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# fuzzifier.py
from typing import Union, Optional, Any, List, Dict
import numpy as np
from matplotlib import pyplot as plt

from .registry import get_registry_fuzzify
from ..config import get_config
from ..membership import MembershipFunction, get_mf_class
from ..core import Fuzznum, Fuzzarray


class Fuzzifier:
    """
    模糊化引擎
    - 强制参数格式: mf_params: List[Dict]
    - 分离策略参数与隶属函数参数
    - 调用策略决策生成结果
    """

    def __init__(self,
                 mf: Union[MembershipFunction, str],
                 mtype: Optional[str] = None,
                 method: Optional[str] = None,
                 **kwargs: Any):
        """
        Parameters
        ----------
        mf : 隶属函数类名或实例
        mtype : 模糊数类型 (e.g. qrofn, qrohfn)
        method : 策略方法名
        kwargs : 包含 'mf_params' 与策略参数
        """
        # 1. 确定策略类
        self.mtype = mtype or get_config().DEFAULT_MTYPE
        registry = get_registry_fuzzify()
        self.method = method or registry.get_default_method(self.mtype)

        if self.method is None:
            raise ValueError(f"No default method for mtype '{self.mtype}'")

        strategy_cls = registry.get_strategy(self.mtype, self.method)
        if strategy_cls is None:
            raise ValueError(f"No strategy found for mtype {self.mtype}, method {self.method}")

        # 2. 确定隶属函数类
        if isinstance(mf, MembershipFunction):
            self.mf_cls = mf.__class__
            self.provided_mf_instance = mf
        else:
            self.mf_cls = get_mf_class(mf)
            self.provided_mf_instance = None

        # 3. 提取并标准化 mf_params
        if "mf_params" not in kwargs:
            raise ValueError("Fuzzifier requires 'mf_params' argument (dict or list of dicts).")

        mf_params = kwargs.pop("mf_params")
        if isinstance(mf_params, dict):
            mf_params_list = [mf_params]
        elif isinstance(mf_params, list) and all(isinstance(d, dict) for d in mf_params):
            mf_params_list = mf_params
        else:
            raise TypeError("mf_params must be either a dict, or a list of dicts.")

        self.mf_params_list: List[Dict] = mf_params_list

        # 4. 实例化策略（其余 kwargs 转给策略）
        self.strategy = strategy_cls(**kwargs)

    def __call__(self, x: Union[float, int, list, np.ndarray]) -> Fuzznum | Fuzzarray:
        """调用策略进行模糊化"""
        return self.strategy.fuzzify(x, self.mf_cls, self.mf_params_list)

    def __repr__(self):
        return (f"Fuzzifier(method='{self.method}', "
                f"mtype='{self.mtype}', "
                f"mf='{self.mf_cls.__name__}', "
                f"params={self.mf_params_list})")

    def plot(self,
             x_range: tuple = (0, 1),
             num_points: int = 100,
             show: bool = True):
        """
        绘制此 Fuzzifier 对应的隶属函数集合曲线.

        Args:
            x_range (tuple): 横坐标区间 (min, max)
            num_points (int): 采样点个数
            show (bool): 是否立即显示图像
        """
        x = np.linspace(x_range[0], x_range[1], num_points)

        plt.figure(figsize=(8, 5))
        for idx, params in enumerate(self.mf_params_list):
            # 创建对应的隶属函数实例
            mf = self.mf_cls(**params)
            y = mf.compute(x)
            label = f"{self.mf_cls.__name__} {params}"
            plt.plot(x, y, label=label)

        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.title(f"Fuzzifier: {self.mf_cls.__name__} [{self.mtype}:{self.method}]")
        plt.legend()
        plt.grid(True)

        if show:
            plt.show()
