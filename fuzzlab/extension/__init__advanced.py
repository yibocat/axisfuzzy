#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 21:11
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .registry import get_extension_registry
from .dispatcher import get_extension_dispatcher
from .injector import get_extension_injector
from .decorators import extension, batch_extension
from .utils import call_extension

from .conditional_ import conditional_extension
from .performance_ import monitored_extension, get_extension_performance_monitor
from .plugins_ import get_extension_plugin_manager
from .validation_ import get_extension_validator


def apply_extensions():
    """应用所有扩展功能"""

    # 验证注册
    validator = get_extension_validator()
    issues = validator.validate_all_registrations()
    if issues:
        import warnings
        for func_name, func_issues in issues.items():
            for issue in func_issues:
                warnings.warn(f"Extension validation issue in {func_name}: {issue}")

    # 准备类映射
    from ..core import Fuzznum, Fuzzarray
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray
    }

    # 注入到fuzzlab命名空间
    import fuzzlab
    module_namespace = fuzzlab.__dict__

    # 执行注入
    injector = get_extension_injector()
    injector.inject_all(class_map, module_namespace)


def get_extension_stats():
    """获取扩展统计信息"""
    registry = get_extension_registry()
    performance = get_extension_performance_monitor()
    validator = get_extension_validator()

    return {
        'registered_functions': registry.list_functions(),
        'performance_stats': performance.get_stats(),
        'validation_issues': validator.validate_all_registrations()
    }


__all__ = [
    'get_extension_registry',
    'get_extension_dispatcher',
    'get_extension_injector',
    'get_extension_plugin_manager',
    'get_extension_validator',
    'get_extension_stats',
    'extension',
    'conditional_extension',
    'batch_extension',
    'monitored_extension',
    'apply_extensions',
    'call_extension'
]
