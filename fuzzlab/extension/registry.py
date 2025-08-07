#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 14:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Tuple, Callable, Any, Union
import threading


@dataclass
class FunctionMetadata:
    """功能元数据"""
    name: str
    mtype: Optional[str]
    target_classes: List[str]
    injection_type: Literal['instance_method', 'top_level_function', 'both']
    is_default: bool = False
    priority: int = 0
    description: str = ""


class ExtensionRegistry:
    """
    基于mtype的外部功能扩展注册表

    支持：
    1. 基于mtype的功能特化
    2. 默认实现回退机制
    3. 优先级排序
    4. 线程安全操作
    """
    def __init__(self):
        self._lock = threading.RLock()

        # {function_name: {mtype: (implementation, metadata)}}
        self._functions: Dict[str, Dict[str, Tuple[Callable, FunctionMetadata]]] = {}

        # {function_name: (default_implementation, metadata)}
        self._defaults: Dict[str, Tuple[Callable, FunctionMetadata]] = {}

        # Registration History
        self._registration_history: List[Dict[str, Any]] = []

    def register(self,
                 name: str,
                 mtype: Optional[str] = None,
                 target_classes: Union[str, List[str]] = None,
                 injection_type: Literal['instance_method', 'top_level_function', 'both'] = 'both',
                 is_default: bool = False,
                 priority: int = 0,
                 **kwargs) -> Callable:
        """
        注册外部功能装饰器

        Args:
            name: 功能名称
            mtype: 目标模糊数类型，None表示通用实现
            target_classes: 目标类列表
            injection_type: 注入类型
            is_default: 是否为默认实现
            priority: 优先级

        Returns:
        """
        if isinstance(target_classes, str):
            target_classes = [target_classes]
        elif target_classes is None:
            target_classes = ['Fuzznum', 'Fuzzarray']

        def decorator(func: Callable) -> Callable:
            metadata = FunctionMetadata(
                name=name,
                mtype=mtype,
                target_classes=target_classes,
                injection_type=injection_type,
                is_default=is_default,
                priority=priority,
                **kwargs
            )

            with self._lock:
                if is_default:
                    # 注册默认实现
                    if name in self._defaults:
                        existing_priority = self._defaults[name][1].priority
                        if priority <= existing_priority:
                            raise ValueError(f"Default implementation for '{name}' already exists with higher priority")
                    self._defaults[name] = (func, metadata)
                else:
                    # 注册特化实现
                    if name not in self._functions:
                        self._functions[name] = {}

                    if mtype in self._functions[name]:
                        existing_priority = self._functions[name][mtype][1].priority
                        if priority <= existing_priority:
                            raise ValueError(f"Implementation for '{name}' with mtype '{mtype}' already exists with higher priority")

                    self._functions[name][mtype] = (func, metadata)

                # 记录注册历史
                self._registration_history.append({
                    'name': name,
                    'mtype': mtype,
                    'is_default': is_default,
                    'priority': priority,
                    'timestamp': self._get_timestamp()
                })

            return func

        return decorator

    def get_function(self, name: str, mtype: str) -> Optional[Callable]:
        """
        获取指定 mtype 的功能实现
        """
        with self._lock:
            # 首先查找特化实现
            if name in self._functions and mtype in self._functions[name]:
                return self._functions[name][mtype][0]

            # 回退到默认实现
            if name in self._defaults:
                return self._defaults[name][0]

            return None

    def get_metadata(self, name: str, mtype: Optional[str] = None) -> Optional[FunctionMetadata]:
        """获取功能元数据"""
        with self._lock:
            if mtype and name in self._functions and mtype in self._functions[name]:
                return self._functions[name][mtype][1]

            if name in self._defaults:
                return self._defaults[name][1]

            return None

    def list_functions(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册功能"""
        with self._lock:
            result = {}

            # 添加特化实现
            for func_name, implementations in self._functions.items():
                if func_name not in result:
                    result[func_name] = {'implementations': {}, 'default': None}

                for mtype, (func, metadata) in implementations.items():
                    result[func_name]['implementations'][mtype] = {
                        'priority': metadata.priority,
                        'target_classes': metadata.target_classes,
                        'injection_type': metadata.injection_type
                    }

            # 添加默认实现
            for func_name, (func, metadata) in self._defaults.items():
                if func_name not in result:
                    result[func_name] = {'implementations': {}, 'default': None}

                result[func_name]['default'] = {
                    'priority': metadata.priority,
                    'target_classes': metadata.target_classes,
                    'injection_type': metadata.injection_type
                }

            return result

    @staticmethod
    def _get_timestamp():
        import datetime
        return datetime.datetime.now().isoformat()


# 全局单例
_extension_registry = None
_extension_registry_lock = threading.RLock()


def get_extension_registry() -> ExtensionRegistry:
    """获取全局扩展注册表实例"""
    global _extension_registry
    if _extension_registry is None:
        with _extension_registry_lock:
            if _extension_registry is None:
                _extension_registry = ExtensionRegistry()
    return _extension_registry
