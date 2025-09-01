#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/19 12:17
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

# fuzzifier.py
from typing import Union, Optional, Any, List, Dict, Type
import numpy as np
import inspect

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
                 mf: Union[MembershipFunction, Type[MembershipFunction], str],
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
        # --- 保存原始构造参数，这对于序列化至关重要 ---
        self._init_mf = mf
        self._init_mtype = mtype
        self._init_method = method
        self._init_kwargs = kwargs.copy()   # 必须复制
        # --- 序列化所需信息结束 ---

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
            # 传入的是隶属函数实例
            self.mf_cls = mf.__class__
            self.provided_mf_instance = mf
        elif inspect.isclass(mf) and issubclass(mf, MembershipFunction):
            # 传入的是隶属函数类
            self.mf_cls = mf
            self.provided_mf_instance = None
        else:
            # 传入的是字符串名称或别名
            self.mf_cls = get_mf_class(mf)
            self.provided_mf_instance = None

        # 3. 提取并标准化 mf_params
        if "mf_params" not in kwargs:
            # 如果传入的是隶属函数实例且未提供 mf_params，则自动从实例中提取参数
            if isinstance(mf, MembershipFunction):
                mf_params = mf.get_parameters()
            else:
                raise ValueError("Fuzzifier requires 'mf_params' argument (dict or list of dicts) when not using a MembershipFunction instance.")
        else:
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

    def get_config(self) -> Dict[str, Any]:
        """
        返回一个可序列化的配置字典，用于重建 Fuzzifier。

        此方法返回重建实例所需的所有构造参数。
        """
        # 隶属函数总是以字符串形式保存，以确保可移植性
        if isinstance(self._init_mf, str):
            mf_name = self._init_mf
        else:
            mf_name = self._init_mf.__class__.__name__

        config = {
            'mf': mf_name,
            'mtype': self._init_mtype,
            'method': self._init_method,
        }
        # 将 mf_params 和其他策略参数合并
        config.update(self._init_kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Fuzzifier':
        """
        从配置字典重建 Fuzzifier 实例。

        Args:
            config (dict): 由 get_config 方法生成的配置字典。

        Returns:
            Fuzzifier: 重建的 Fuzzifier 实例。
        """
        return cls(**config)

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
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib or uv add --optional analysis matplotlib"
            )
        
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
