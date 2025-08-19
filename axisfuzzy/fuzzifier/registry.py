#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 23:34
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
from typing import Dict, Tuple, Type, Optional, List

from .strategy import FuzzificationStrategy


class FuzzificationStrategyRegistry:

    def __init__(self):
        # Strategy Registry: {(mtype, method): strategy_class}
        self._strategies: Dict[Tuple[str, str], Type[FuzzificationStrategy]] = {}
        self._default_methods: Dict[str, str] = {}

    def register(self,
                 mtype: str,
                 method: str,
                 strategy_cls: Type[FuzzificationStrategy],
                 is_default: bool = False) -> None:
        """
        Register a new fuzzification strategy.

        Args:
            mtype: The type of fuzzy number (e.g., 'qrofn').
            method: The name of the strategy (e.g., 'expert', 'hesitation').
            strategy_cls: The class implementing the strategy.
            is_default: Whether this should be the default method for the mtype.
        """
        if not issubclass(strategy_cls, FuzzificationStrategy):
            raise TypeError(f"Registered strategy class must be a subclass of FuzzificationStrategy, "
                            f"got {strategy_cls}")

        key = (mtype, method)
        if key in self._strategies:
            raise ValueError(f"Strategy for {key} is already registered.")

        self._strategies[key] = strategy_cls

        if is_default or mtype not in self._default_methods:
            self._default_methods[mtype] = method

    def get_strategy(self,
                     mtype: str,
                     method: Optional[str] = None) -> Optional[Type[FuzzificationStrategy]]:
        if method is None:
            method = self.get_default_method(mtype)
            if method is None:
                return None

        return self._strategies.get((mtype, method))

    def list_strategies(self,
                        mtype: Optional[str] = None) -> List[Tuple[str, str]]:
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


_registry = FuzzificationStrategyRegistry()


def get_registry_fuzzify() -> FuzzificationStrategyRegistry:
    """获取全局模糊化注册表实例"""
    return _registry


def register_fuzzifier(is_default: bool = False):
    def decorator(strategy_cls: Type[FuzzificationStrategy]):
        get_registry_fuzzify().register(
            mtype=strategy_cls.mtype,
            method=strategy_cls.method,
            strategy_cls=strategy_cls,
            is_default=is_default
        )
        return strategy_cls

    return decorator
