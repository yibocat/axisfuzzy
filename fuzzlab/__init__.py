#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 21:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
FuzzLab: A Python library for fuzzy number and fuzzy array computations.

This is the main entry point of the FuzzLab library. It aggregates the most
commonly used components from subpackages into a single, convenient namespace.
"""

# 1. 导入子模块以触发其内部的注册和动态加载机制。
#    - `fuzztype` 会自动发现并加载所有 mtype 的实现。
#    - 其他模块的导入确保了它们的功能在注入前是可用的。
from . import core
from . import config
from . import extension
from . import fuzzify
from . import membership
from . import mixin
from . import random
from . import fuzztype         # 必须导入以触发 mtype 的自动注册

# 2. 从子模块中显式导入需要提升到顶层命名空间的公共 API。
from .config import (
    get_config,
    set_config,
    Config
)
from .core import (
    Fuzznum,
    Fuzzarray,
    fuzzarray,
    operate
)
from .extension import (
    extension,
    batch_extension,
    apply_extensions,
    get_extension_registry
)
from .fuzzify import fuzzify, Fuzzifier
from .membership import create_mf, MembershipFunction
from .mixin import apply_mixins, get_mixin_registry

# 3. 准备 __all__ 列表，明确声明本模块的公共 API。
#    - 首先包含所有静态导入的名称。
#    - 然后动态地从 mixin 系统获取将被注入的函数名。
_static_api = [
    # config
    'get_config', 'set_config', 'Config',
    # core
    'Fuzznum', 'Fuzzarray', 'fuzzarray', 'operate',
    # extension decorators
    'extension', 'batch_extension',
    # fuzzify
    'fuzzify', 'Fuzzifier',
    # membership
    'create_mf', 'MembershipFunction',
    # random (as a module)
    'random',
]

# 从 mixin 系统获取动态函数名
_mixin_funcs = get_mixin_registry().get_top_level_function_names()

# 从 extension 系统获取动态函数名
_extension_funcs = get_extension_registry().get_top_level_function_names()

__all__ = sorted(list(set(_static_api + _mixin_funcs + _extension_funcs)))

# 4. 在所有模块都已加载、所有 API 都已声明后，执行动态注入。
#    这会将 extension 和 mixin 函数附加到核心类和本模块的命名空间中。
apply_extensions(globals())
apply_mixins(globals())

# 5. 清理命名空间
del _static_api
del _mixin_funcs
del _extension_funcs
del apply_extensions
del apply_mixins
del get_extension_registry
del get_mixin_registry

