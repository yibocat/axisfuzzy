#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/6 16:07
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

import warnings
from typing import Type, Any, Dict

from .registry import get_extend_registry
from .dispatcher import create_method_dispatcher, create_top_level_dispatcher
from .. import Fuzznum, Fuzzarray

__all__ = []


def apply_extensions():
    """
    扫描并加载所有外部功能，然后将其注入到 Fuzznum、Fuzzarray 类和 fuzzlab 命名空间中。
    """
    extend_registry = get_extend_registry()

    # 2. 准备目标类映射
    class_map: Dict[str, Type[Any]] = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray
    }

    # 3. 注入功能
    # for func_name in list(extend_registry._injection_types.keys()):
    for func_name in list(extend_registry.get_injection().keys()):
        injection_type = extend_registry.get_injection_type(func_name)

        metadata = extend_registry.get_metadata().get(func_name, {})
        target_classes = metadata.get('target_classes', [])

        if injection_type in ['instance_method', 'both']:
            method_dispatcher = create_method_dispatcher(func_name)
            for class_name in target_classes:
                if class_name in class_map:
                    if not hasattr(class_map[class_name], func_name):
                        setattr(class_map[class_name], func_name, method_dispatcher)
                else:
                    # 可以选择警告或抛出错误
                    warnings.warn(f"Warning: Method '{func_name}' already exists on '{class_name}'. "
                                  f"Skipping injection.")

        # 注入为顶级函数
        if injection_type in ['top_level_function', 'both']:
            top_level_dispatcher = create_top_level_dispatcher(func_name)
            # 注入到 fuzzlab 命名空间
            # 假设 fuzzlab 是顶层包
            import fuzzlab as _fuzzlab_root
            if not hasattr(_fuzzlab_root, func_name):
                setattr(_fuzzlab_root, func_name, top_level_dispatcher)
                if func_name not in __all__:
                    __all__.append(func_name)
            else:
                warnings.warn(f"Warning: Top-level function '{func_name}' "
                              f"already exists in 'fuzzlab' namespace. Skipping injection.")


__all__ += [
    'get_extend_registry',
    'apply_extensions'
]
