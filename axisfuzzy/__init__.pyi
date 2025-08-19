#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy


"""
Type stubs for the top-level `axisfuzzy` package.

This file provides type hints for all public APIs exposed in the `axisfuzzy`
namespace, including dynamically injected functions from the mixin system.
"""
from typing import List, Any

# 1. 显式地从子模块重导出所有静态 API 的类型。
#    使用 `as` 语法是向类型检查器声明重导出的最明确方式。
from .config import (
    get_config_manager as get_config_manager,
    get_config as get_config,
    set_config as set_config,
    load_config_file as load_config_file,
    save_config_file as save_config_file,
    reset_config as reset_config,
)

from .core import (
    FuzznumStrategy as FuzznumStrategy,
    FuzzarrayBackend as FuzzarrayBackend,
    Fuzznum as Fuzznum, fuzznum as fuzznum,
    Fuzzarray as Fuzzarray, fuzzarray as fuzzarray,
    operate as operate,
    OperationTNorm as OperationTNorm,

    get_registry_fuzztype as get_registry_fuzztype,
    get_registry_operation as get_registry_operation,

    get_fuzztype_mtypes as get_fuzztype_mtypes,
    get_fuzztype_strategy as get_fuzztype_strategy,
    get_fuzztype_backend as get_fuzztype_backend,

    register_strategy as register_strategy,
    register_backend as register_backend,
    register_operation as register_operation,
    register_fuzztype as register_fuzztype,
    # register_batch_fuzztypes as register_batch_fuzztypes,
    unregister_fuzztype as unregister_fuzztype
)
from .extension import (
    get_registry_extension as get_registry_extension,
    extension as extension,
    batch_extension as batch_extension
)

from .fuzzifier import (
    FuzzificationStrategy as FuzzificationStrategy,
    Fuzzifier as Fuzzifier,
    get_registry_fuzzify as get_registry_fuzzify,
    register_fuzzifier as register_fuzzifier,
)

from .membership import (
    create_mf as create_mf,
    get_mf_class as get_mf_class
)

# 2. 将 `random` 声明为一个可导入的模块。
from . import random as random

# 3. 从 `mixin/__init__.pyi` 中导入所有将被注入的顶层函数的签名。
from .mixin import (
    get_registry_mixin as get_registry_mixin,
    reshape as reshape,
    flatten as flatten,
    squeeze as squeeze,
    copy as copy,
    ravel as ravel,
    transpose as transpose,
    broadcast_to as broadcast_to,
    item as item,
    any as any,
    all as all,
    concat as concat,
    stack as stack,
    append as append,
    pop as pop,
)

# 4. 关键修复：从 qrofs 的扩展中导入所有将被注入的顶层函数的签名。
#    注意：未来有新 mtype 时，需要从其 extension.pyi 中导入。
from .extension import (
    empty as empty,
    positive as positive,
    negative as negative,
    full as full,
    empty_like as empty_like,
    positive_like as positive_like,
    negative_like as negative_like,
    full_like as full_like,
    read_csv as read_csv,
    read_json as read_json,
    read_npy as read_npy,
    distance as distance,
    str2fuzznum as str2fuzznum,
    sum as sum,
    mean as mean,
    max as max,
    min as min,
    prod as prod,
    var as var,
    std as std,
)

# 4. 声明 __all__ 列表的类型。
__all__: List[str]
