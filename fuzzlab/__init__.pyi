#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:10
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab


"""
Type stubs for the top-level `fuzzlab` package.

This file provides type hints for all public APIs exposed in the `fuzzlab`
namespace, including dynamically injected functions from the mixin system.
"""
from typing import List, Any

# 1. 显式地从子模块重导出所有静态 API 的类型。
#    使用 `as` 语法是向类型检查器声明重导出的最明确方式。
from .config import (
    get_config as get_config,
    set_config as set_config,
    Config as Config
)
from .core import (
    Fuzznum as Fuzznum,
    Fuzzarray as Fuzzarray,
    fuzzarray as fuzzarray,
    operate as operate
)
from .extension import (
    extension as extension,
    batch_extension as batch_extension
)
from .fuzzify import (
    fuzzify as fuzzify,
    Fuzzifier as Fuzzifier
)
from .membership import (
    create_mf as create_mf,
    MembershipFunction as MembershipFunction
)

# 2. 将 `random` 声明为一个可导入的模块。
from . import random as random

# 3. 从 `mixin/__init__.pyi` 中导入所有将被注入的顶层函数的签名。
from .mixin import (
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
