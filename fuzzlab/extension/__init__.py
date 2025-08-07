#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 14:36
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .registry import get_extension_registry
from .dispatcher import get_extension_dispatcher
from .injector import get_extension_injector
from .decorators import extension, batch_extension
from .utils import call_extension


def apply_extensions():
    """
    应用所有扩展功能。
    这个函数负责将所有已注册的扩展（包括内部和外部）注入到 Fuzznum、Fuzzarray 类
    以及 fuzzlab 的顶级命名空间中。
    """
    # 准备类映射
    from ..core import Fuzznum, Fuzzarray
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray,
    }

    # 注入到全局命名空间
    import fuzzlab
    module_namespace = fuzzlab.__dict__

    # 执行注入
    apply_injector = get_extension_injector()
    apply_injector.inject_all(class_map, module_namespace)


__all__ = [
    'get_extension_registry',
    'get_extension_dispatcher',
    'get_extension_injector',
    'extension',
    'batch_extension',
    'apply_extensions',
    'call_extension',
]
