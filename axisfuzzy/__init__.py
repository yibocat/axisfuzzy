#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
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
from . import fuzzifier
from . import membership
from . import mixin
from . import random
from . import fuzztype  # 确保所有 mtype 实现被加载

# analysis 模块使用延迟导入策略，避免在核心包导入时因缺少可选依赖而导致错误
# from . import analysis  # 移除直接导入

from .version import __version__

# 2. 从子模块中显式导入需要提升到顶层命名空间的公共 API。API。
from .config import (
    get_config_manager,
    get_config,
    set_config,
    load_config_file,
    save_config_file,
    reset_config
)
from .core import (
    FuzznumStrategy,
    FuzzarrayBackend,
    Fuzznum,
    Fuzzarray,
    fuzzyarray,
    fuzzynum,
    operate,
    OperationTNorm,
    get_registry_fuzztype,
    get_registry_operation,
    register_strategy,
    register_backend,
    register_operation,
    register_fuzztype,
)
from .extension import (
    get_registry_extension,
    extension,
    batch_extension,
    apply_extensions,
    external_extension
)

from .fuzzifier import (
    FuzzificationStrategy,
    Fuzzifier,
    get_registry_fuzzify,
    register_fuzzifier
)

from .membership import create_mf, get_mf_class
from .mixin import apply_mixins, get_registry_mixin

# 3. 准备 __all__ 列表，明确声明本模块的公共 API。
#    - 首先包含所有静态导入的名称。
#    - 然后动态地从 mixin 系统获取将被注入的函数名。
_static_api = [
    # config
    'get_config_manager',
    'get_config',
    'set_config',
    'load_config_file',
    'save_config_file',
    'reset_config',
    '__version__',
    # core
    'FuzznumStrategy',
    'FuzzarrayBackend',
    'Fuzznum',
    'Fuzzarray',
    'fuzzynum',
    'fuzzyarray',
    'operate',
    'OperationTNorm',
    'get_registry_fuzztype',
    'get_registry_operation',
    'get_fuzztype_mtypes',
    'register_strategy',
    'register_backend',
    'register_operation',
    'register_fuzztype',
    # extension
    'get_registry_extension',
    'extension',
    'batch_extension',
    'external_extension',
    # fuzzifier
    'FuzzificationStrategy',
    'Fuzzifier',
    'get_registry_fuzzify',
    'register_fuzzifier',
    # membership
    'create_mf',
    'get_mf_class',
    # mixin
    'get_registry_mixin'
    # random (as a module)
    'random',
    # analysis (as a module)
    # 'analysis'  # 移除直接导入，改为延迟导入
]

# 从 mixin 系统获取动态函数名
_mixin_funcs = get_registry_mixin().get_top_level_function_names()

# 从 extension 系统获取动态函数名
_extension_funcs = get_registry_extension().get_top_level_function_names()

__all__ = sorted(list(set(_static_api + _mixin_funcs + _extension_funcs)))

# 4. 在所有模块都已加载、所有 API 都已声明后，执行动态注入。
#    这会将 extension 和 mixin 函数附加到核心类和本模块的命名空间中。
apply_extensions()
apply_mixins(globals())


# 5. 清理命名空间
# 延迟导入 analysis 模块的实现
def __getattr__(name: str):
    """延迟导入机制，用于处理可选依赖模块。"""
    if name == "analysis":
        try:
            # 使用 importlib 避免递归导入
            import importlib
            analysis_module = importlib.import_module('.analysis', package='axisfuzzy')
            # 将模块缓存到全局命名空间中，避免重复导入
            globals()[name] = analysis_module
            return analysis_module
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' module. This module requires optional dependencies. "
                f"Please install with: pip install 'axisfuzzy[analysis]'. "
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


del _static_api
del _mixin_funcs
del _extension_funcs
del apply_extensions
del apply_mixins
