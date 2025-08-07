#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Callable, Union

from .registry import get_extension_registry
from ..core import Fuzznum, Fuzzarray


class ExtensionDispatcher:
    """外部功能分发器"""

    def __init__(self):
        self.registry = get_extension_registry()

    def create_instance_method(self, func_name: str) -> Callable:
        """创建实例方法分发器"""
        def method_dispatcher(obj, *args, **kwargs):
            """获取对象的 mtype"""
            mtype = getattr(obj, 'mtype', None)
            if mtype is None:
                raise AttributeError(f"Object {type(obj).__name__} has no 'mtype' attribute")

            # 获取对应的实现
            implementation = self.registry.get_function(func_name, mtype)
            if implementation is None:
                available_mtypes = list(self.registry._functions.get(func_name, {}).keys())
                has_default = func_name in self.registry._defaults

                error_msg = f"Function '{func_name}' not implemented for mtype '{mtype}'"
                if available_mtypes:
                    error_msg += f". Available for: {available_mtypes}"
                if has_default:
                    error_msg += ". Default implementation available but failed to load"

                raise NotImplementedError(error_msg)

            # 调用实现
            return implementation(obj, *args, **kwargs)

        method_dispatcher.__name__ = func_name
        method_dispatcher.__doc__ = f"Dispatched method for {func_name}"
        return method_dispatcher

    def create_top_level_function(self, func_name: str) -> Callable:
        """创建顶级函数分发器"""
        def function_dispatcher(obj: Union[Fuzznum, Fuzzarray], *args, **kwargs):
            # 检查对象类型
            if not isinstance(obj, (Fuzznum, Fuzzarray)):
                raise TypeError(f"First argument must be Fuzznum or Fuzzarray, got {type(obj).__name__}")

            # 获取对象的mtype
            mtype = getattr(obj, 'mtype', None)
            if mtype is None:
                raise AttributeError(f"Object {type(obj).__name__} has no 'mtype' attribute")

            # 获取对应的实现
            implementation = self.registry.get_function(func_name, mtype)
            if implementation is None:
                raise NotImplementedError(f"Function '{func_name}' not implemented for mtype '{mtype}'")

            # 调用实现
            return implementation(obj, *args, **kwargs)

        function_dispatcher.__name__ = func_name
        function_dispatcher.__doc__ = f"Dispatched top-level function for {func_name}"
        return function_dispatcher


# 全局分发器实例
_dispatcher = ExtensionDispatcher()


def get_extension_dispatcher() -> ExtensionDispatcher:
    """获取全局分发器实例"""
    return _dispatcher
