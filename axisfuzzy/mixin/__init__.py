#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/5 16:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
import warnings
from typing import Dict, Any


from .registry import get_registry_mixin
from ..core import Fuzznum, Fuzzarray

# 关键修复：导入 register 模块以触发所有 @register 装饰器的执行。
# 这个导入本身没有用到任何变量，但它的副作用是填充了 mixin 注册表。
from . import register

_applied = False


def _apply_functions(target_module_globals: Dict[str, Any] | None = None) -> bool:
    """
    将注册的功能动态注入到目标模块的命名空间以及 Fuzznum/Fuzzarray 类中。

    如果 target_module_globals 为 None，则默认注入到 axisfuzzy 顶级模块（axisfuzzy.__dict__）。
    返回 True 表示注入成功或已应用；False 表示注入失败。
    """
    global _applied
    if _applied:
        return True

    # prepare class map
    class_map = {
        'Fuzznum': Fuzznum,
        'Fuzzarray': Fuzzarray
    }

    # determine where to inject top-level functions: prefer axisfuzzy package
    if target_module_globals is None:
        try:
            import axisfuzzy
            target_module_globals = axisfuzzy.__dict__
        except Exception as e:

            # 这里可以添加日志记录或其他处理方式
            # _logger.exception("Failed to import axisfuzzy for mixin top-level injection: %s", e)

            warnings.warn(f"Failed to import axisfuzzy for mixin top-level injection: {e}")
            # fallback to local mixin module globals to avoid complete failure
            target_module_globals = globals()

    try:
        get_registry_mixin().build_and_inject(class_map, target_module_globals)
        _applied = True
        return True
    except Exception as e:
        # 这里可以添加日志记录或其他处理方式
        # _logger.exception("Failed to apply mixin functions: %s", e)

        warnings.warn(f"Failed to injection mixin functions: {e}")
        return False


# 自动注入（保留以兼容现有行为），但经过幂等与异常保护
# _apply_functions()

apply_mixins = _apply_functions

__all__ = ['get_registry_mixin', 'apply_mixins'] + get_registry_mixin().get_top_level_function_names()
