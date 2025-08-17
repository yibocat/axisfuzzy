#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import inspect
from typing import Dict, Type, Any, Tuple

from .base import MembershipFunction
from . import function as mfs


# 自动构建一个从类名字符串到类对象的映射
# e.g., {'TriangularMF': <class '...TriangularMF'>, ...}
_mf_class_map: Dict[str, Type[MembershipFunction]] = {
    name: obj for name, obj in inspect.getmembers(mfs, inspect.isclass)
    if issubclass(obj, MembershipFunction) and obj is not MembershipFunction
}

# 手动维护一个别名映射，提供更友好的名称
# e.g., {'triangular': TriangularMF, 'trimf': TriangularMF}
_mf_alias_map: Dict[str, Type[MembershipFunction]] = {
    'sigmoid': mfs.SigmoidMF,
    'trimf': mfs.TriangularMF,
    'trapmf': mfs.TrapezoidalMF,
    'gaussmf': mfs.GaussianMF,
    'smf': mfs.SMF,
    'zmf': mfs.ZMF,
    'gbellmf': mfs.GeneralizedBellMF,
    'pimf': mfs.PiMF,
    'gauss2mf': mfs.DoubleGaussianMF
}

# 将类名的小写版本也加入别名映射
_mf_alias_map.update({k.lower(): v for k, v in _mf_class_map.items()})


def get_mf_class(name: str) -> Type[MembershipFunction]:
    """
    根据名称获取隶属函数类。

    Args:
        name: 隶属函数的名称或别名 (大小写不敏感)。

    Returns:
        对应的 MembershipFunction 子类。

    Raises:
        ValueError: 如果找不到指定的隶属函数。
    """
    mf_cls = _mf_alias_map.get(name.lower())
    if mf_cls is None:
        available = ", ".join(sorted(_mf_alias_map.keys()))
        raise ValueError(f"Unknown membership function '{name}'. Available functions are: {available}")
    return mf_cls


def create_mf(name: str, **mf_kwargs: Any) -> Tuple[MembershipFunction, Dict[str, Any]]:
    """
    隶属函数工厂函数。

    根据名称和参数创建隶属函数实例，并返回未被使用的参数。

    Args:
        name: 隶属函数的名称或别名。
        **mf_kwargs: 用于创建实例的参数和可能传递给其他组件的额外参数。

    Returns:
        一个元组，包含：
        - 创建的 MembershipFunction 实例。
        - 未在创建过程中使用的 `kwargs` 字典。
    """
    mf_cls = get_mf_class(name)

    # 获取构造函数需要的所有参数名称
    constructor_signature = inspect.signature(mf_cls.__init__)
    mf_params = constructor_signature.parameters.keys()

    # 从 mf_kwargs 中分离出隶属函数的参数
    mf_init_kwargs = {}
    remaining_kwargs = {}
    for key, value in mf_kwargs.items():
        if key in mf_params:
            mf_init_kwargs[key] = value
        else:
            remaining_kwargs[key] = value

    # 创建实例
    instance = mf_cls(**mf_init_kwargs)

    return instance, remaining_kwargs   # type: ignore
