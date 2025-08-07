#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 15:35
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
from typing import Optional, Union, List, Literal

from .registry import get_extension_registry


def extension(name: str,
              mtype: Optional[str] = None,
              target_classes: Union[str, List[str]] = None,
              injection_type: Literal['instance_method', 'top_level_function', 'both'] = 'both',
              is_default: bool = False,
              priority: int = 0,
              **kwargs):
    """
    外部功能扩展装饰器

    Examples:
        # 为qrofn注册距离计算
        @extension('distance', mtype='qrofn', target_classes=['Fuzznum'])
        def qrofn_distance(fuzz1, fuzz2, method='euclidean', **kwargs):
            # QROFN特定的距离计算
            pass

        # 通用距离计算（默认实现）
        @extension('distance', is_default=True, target_classes=['Fuzznum'])
        def default_distance(fuzz1, fuzz2, **kwargs):
            # 通用距离计算
            pass

        # 构造函数
        @extension('random', mtype='qrofn', target_classes=['Fuzznum'])
        def create_random_qrofn(**kwargs):
            # 创建随机QROFN
            pass
    """
    registry = get_extension_registry()
    return registry.register(
        name=name,
        mtype=mtype,
        target_classes=target_classes,
        injection_type=injection_type,
        is_default=is_default,
        priority=priority,
        **kwargs
    )


def batch_extension(registrations: List[dict]):
    """批量注册装饰器"""
    def decorator(func):
        registry = get_extension_registry()
        for reg in registrations:
            registry.register(**reg)(func)
        return func
    return decorator
