#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import inspect
from typing import Optional, Union, Any

import numpy as np

from .registry import get_registry_fuzzify
from ..core import Fuzzarray, Fuzznum
from ..config import get_config
from ..membership import MembershipFunction, get_mf_class


class Fuzzifier:
    """
    一个可配置、可复用的模糊化引擎。

    此类将模糊化的“配置”阶段（选择隶属函数、策略等）与“执行”阶段分离。
    首先创建一个 Fuzzifier 实例并配置好它，然后可以重复调用该实例来
    模糊化不同的数据集。
    """

    def __init__(self,
                 mf: Union[MembershipFunction, str],
                 mtype: Optional[str] = None,
                 method: Optional[str] = None,
                 **kwargs: Any):
        """
        初始化并配置模糊化引擎。

        Args:
            mf: 隶属函数实例或名称字符串。
            mtype: 目标模糊数类型。如果为 None，使用全局默认值。
            method: 模糊化策略名称。如果为 None，使用 mtype 的默认策略。
            **kwargs: 包含创建隶属函数和初始化策略所需的所有参数。
                      (e.g., a=0, b=1, c=2, q=2, pi=0.1)
        """
        # 1. 确定 mtype 和 method
        self.mtype = mtype or get_config().DEFAULT_MTYPE
        registry = get_registry_fuzzify()
        self.method = method or registry.get_default_method(self.mtype)

        if self.method is None:
            available = registry.get_available_mtypes()
            raise ValueError(f"No default method for mtype '{self.mtype}'. Available mtypes: {available}")

        # 2. 获取并实例化策略
        strategy_cls = registry.get_strategy(self.mtype, self.method)
        if strategy_cls is None:
            available = registry.get_available_methods(self.mtype)
            raise ValueError(f"Strategy not found for mtype='{self.mtype}', method='{self.method}'. "
                             f"Available methods for {self.mtype}: {available}")

        is_mf_instance = isinstance(mf, MembershipFunction)
        mf_cls = mf.__class__ if is_mf_instance else get_mf_class(mf)

        # 3. 使用自省机制分离参数
        strategy_params = inspect.signature(strategy_cls.__init__).parameters.keys()
        mf_params = inspect.signature(mf_cls.__init__).parameters.keys()

        strategy_init_kwargs = {}
        mf_init_kwargs = {}

        for key, value in kwargs.items():
            if key in strategy_params:
                strategy_init_kwargs[key] = value
            elif key in mf_params:
                if is_mf_instance:
                    # 如果 mf 已经是实例，不应该再提供它的参数
                    raise ValueError(f"Parameter '{key}' is for the membership function, "
                                     "but an already initialized instance was provided.")
                mf_init_kwargs[key] = value
            else:
                raise ValueError(f"Unknown parameter '{key}' for strategy '{strategy_cls.__name__}' "
                                 f"or membership function '{mf_cls.__name__}'.")

        # 4. 实例化策略和隶属函数
        self.strategy = strategy_cls(**strategy_init_kwargs)
        self.mf = mf if is_mf_instance else mf_cls(**mf_init_kwargs)

    def __call__(self, x: Union[float, int, list, np.ndarray]) -> Union[Fuzznum, Fuzzarray]:
        """
        执行模糊化。

        Args:
            x: 需要被模糊化的精确值或数组。

        Returns:
            生成的 Fuzznum 或 Fuzzarray。
        """
        # 统一处理 numpy 标量与 0 维 ndarray
        if isinstance(x, (int, float, np.number)):
            return self.strategy.fuzzify_scalar(float(x), self.mf)

        elif isinstance(x, (list, tuple, np.ndarray)):
            x_array = np.asarray(x)
            if x_array.ndim == 0:
                return self.strategy.fuzzify_scalar(float(x_array), self.mf)
            return self.strategy.fuzzify_array(x_array, self.mf)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}. Must be float, int, list, tuple, or np.ndarray.")


def fuzzify(
        x: Union[float, int, list, np.ndarray],
        mf: Union[MembershipFunction, str],
        mtype: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs: Any
) -> Union[Fuzznum, Fuzzarray]:
    """
    将精确值或数组模糊化为 Fuzznum 或 Fuzzarray (便捷函数)。

    这是一个无状态的便捷函数，用于一次性的模糊化任务。它在内部创建
    一个临时的 Fuzzifier 实例并立即调用它。

    对于需要使用相同配置重复进行模糊化的场景，建议直接创建和使用
    Fuzzifier 类的实例以获得更好的性能和代码清晰度。

    Args:
        x:  输入的精确值。
        mf: 隶属函数实例或名称字符串。
        mtype:  目标模糊数的类型。
        method: 要使用的模糊化策略名称。
        **kwargs:   创建隶属函数和初始化策略所需的所有参数。

    Returns:
        生成的 Fuzznum 或 Fuzzarray。
    """
    return Fuzzifier(mf=mf, mtype=mtype, method=method, **kwargs)(x)
