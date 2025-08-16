#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/14 17:08
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import warnings
from typing import Tuple, Type, Dict, Optional, List

from .base import FuzzificationStrategy


class FuzzificationRegistry:
    """
    管理和存储所有可用的模糊化策略。

    该注册表维护一个从 (mtype, method) 到 FuzzificationStrategy 子类的映射。
    """
    def __init__(self):
        # 策略注册表: {(mtype, method): strategy_class}
        self._strategies: Dict[Tuple[str, str], Type[FuzzificationStrategy]] = {}
        # 默认策略: {mtype: default_method}
        self._default_methods: Dict[str, str] = {}

    def register(self,
                 mtype: str,
                 method: str,
                 strategy_cls: Type[FuzzificationStrategy],
                 is_default: bool = False) -> None:
        """
        注册一个新的模糊化策略。

        Args:
            mtype: 策略适用的模糊数类型 (e.g., 'qrofn')。
            method: 策略的名称 (e.g., 'expert', 'hesitation')。
            strategy_cls: 实现该策略的 FuzzificationStrategy 子类。
            is_default: 是否设为该mtype的默认方法
        """
        if not issubclass(strategy_cls, FuzzificationStrategy):
            raise TypeError(f"Registered strategy class must be a subclass of FuzzificationStrategy, "
                            f"got {strategy_cls}")

        key = (mtype, method)
        if key in self._strategies:
            warnings.warn(f"Overriding existing strategy for {key}")
            pass

        self._strategies[key] = strategy_cls

        if is_default or mtype not in self._default_methods:
            self._default_methods[mtype] = method

    def get_strategy(self,
                     mtype: str,
                     method: Optional[str] = None) -> Optional[Type[FuzzificationStrategy]]:
        """
        根据 mtype 和 method 获取一个模糊化策略类。

        Args:
            mtype: 目标模糊数类型。
            method: 目标策略名称。如果为 None，则尝试获取默认方法。

        Returns:
            对应的 FuzzificationStrategy 子类，如果未找到则返回 None。
        """
        if method is None:
            method = self.get_default_method(mtype)
            if method is None:
                return None

        return self._strategies.get((mtype, method))

    def list_strategies(self,
                        mtype: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        列出所有注册的策略

        Args:
            mtype: 如果指定，只返回该类型的策略

        Returns:
            策略列表 [(mtype, method), ...]
        """
        if mtype is None:
            return list(self._strategies.keys())
        else:
            return [(mt, method) for mt, method in self._strategies.keys() if mt == mtype]

    def get_available_mtypes(self) -> List[str]:
        """获取所有支持的模糊数类型"""
        return list(set(mtype for mtype, _ in self._strategies.keys()))

    def get_available_methods(self, mtype: str) -> List[str]:
        """获取指定模糊数类型的所有可用方法"""
        return [method for mt, method in self._strategies.keys() if mt == mtype]

    def get_default_method(self, mtype: str) -> Optional[str]:
        """获取指定模糊数类型的默认方法"""
        return self._default_methods.get(mtype)

    def get_registry_info(self) -> Dict[str, any]:
        """获取注册表的详细信息"""
        return {
            'total_strategies': len(self._strategies),
            'supported_mtypes': self.get_available_mtypes(),
            'default_methods': self._default_methods.copy(),
            'strategies': {
                f"{mtype}.{method}": strategy_class.__name__
                for (mtype, method), strategy_class in self._strategies.items()
            }
        }


_registry = FuzzificationRegistry()


def get_registry_fuzzify() -> FuzzificationRegistry:
    """获取全局模糊化注册表实例"""
    return _registry


def register_fuzzify(mtype: str,
                     method: str,
                     is_default: bool = False):
    """
    装饰器：注册模糊化策略

    Args:
        mtype: 模糊数类型
        method: 方法名称
        is_default: 是否设为默认方法
    """

    def decorator(strategy_cls: Type[FuzzificationStrategy]):
        get_registry_fuzzify().register(
            mtype=mtype,
            method=method,
            strategy_cls=strategy_cls,
            is_default=is_default
        )
        return strategy_cls

    return decorator
