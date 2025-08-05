#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/5 16:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .registry import get_mixin_registry
from .ops_func import *

from ..core import Fuzznum, Fuzzarray


# 3. 执行动态注入
def _apply_functions():
    """
    将注册的功能动态注入到 Fuzznum 和 Fuzzarray 类中，
    并将顶层函数注入到 fuzzlab 模块的命名空间。
    """
    # 构建一个从类名字符串到实际类对象的映射
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray
    }

    # 调用 FunctionRegistry 的 build_and_inject 方法。
    # globals() 传递的是当前模块（即 fuzzlab/__init__.py）的命名空间，
    # 这样顶层函数就可以被直接添加到 fuzzlab 模块下。
    get_mixin_registry().build_and_inject(class_map, globals())


_apply_functions()

__all__ = [
    'get_mixin_registry'
] + get_mixin_registry().get_top_level_function_names()
