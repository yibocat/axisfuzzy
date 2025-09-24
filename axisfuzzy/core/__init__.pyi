#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Any, Dict, Tuple, Optional

# 从具体子模块显式重导出，以便类型检查器识别
from .base import FuzznumStrategy as FuzznumStrategy

from .backend import FuzzarrayBackend as FuzzarrayBackend

from .fuzznums import Fuzznum as Fuzznum

from .fuzzarray import Fuzzarray as Fuzzarray

from .dispatcher import (
    operate as operate)

from .operation import (
    OperationMixin as OperationMixin,
    get_registry_operation as get_registry_operation,
    register_operation as register_operation
)

from .triangular import OperationTNorm as OperationTNorm

from .registry import (
    get_registry_fuzztype as get_registry_fuzztype,
    register_strategy as register_strategy,
    register_backend as register_backend,
    register_fuzztype as register_fuzztype,
)

from .factory import (
    fuzzynum as fuzzynum,
    fuzzyarray as fuzzyarray
)

__all__ = [
    'FuzznumStrategy',
    'FuzzarrayBackend',
    'Fuzznum',
    'Fuzzarray',
    'fuzzyarray',
    'fuzzynum',
    'operate',
    'OperationMixin',
    'OperationTNorm',
    'get_registry_operation',
    'register_operation',

    'get_registry_fuzztype',
    'register_strategy',
    'register_backend',
    'register_fuzztype',
]