#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:39
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Dict, Type, Any

from .registry import get_extension_registry
from .dispatcher import get_extension_dispatcher


class ExtensionInjector:
    """外部功能注入器"""

    def __init__(self):
        self.registry = get_extension_registry()
        self.dispatcher = get_extension_dispatcher()

    def inject_all(self, class_map: Dict[str, Type], module_namespace: Dict[str, Any]):
        """注入所有注册的功能"""
        functions = self.registry.list_functions()

        for func_name, func_info in functions.items():
            self._inject_function(func_name, func_info, class_map, module_namespace)

    def _inject_function(self,
                         func_name: str,
                         func_info: Dict[str, Any],
                         class_map: Dict[str, Type],
                         module_namespace: Dict[str, Any]):
        """注入单个功能"""
        # 收集所有目标类
        target_classes = set()
        injection_types = set()

        # 从特化实现中收集
        for mtype_info in func_info['implementations'].values():
            target_classes.update(mtype_info['target_classes'])
            injection_types.add(mtype_info['injection_type'])

        # 从默认实现中收集
        if func_info['default']:
            target_classes.update(func_info['default']['target_classes'])
            injection_types.add(func_info['default']['injection_type'])

        # 注入实例方法
        if any(it in ['instance_method', 'both'] for it in injection_types):
            method_dispatcher = self.dispatcher.create_instance_method(func_name)

            for class_name in target_classes:
                if class_name in class_map:
                    cls = class_map[class_name]
                    if not hasattr(cls, func_name):
                        setattr(cls, func_name, method_dispatcher)

        # 注入顶级函数
        if any(it in ['top_level_function', 'both'] for it in injection_types):
            function_dispatcher = self.dispatcher.create_top_level_function(func_name)

            if func_name not in module_namespace:
                module_namespace[func_name] = function_dispatcher


# 全局注入器实例
_injector = ExtensionInjector()


def get_extension_injector() -> ExtensionInjector:
    return _injector
