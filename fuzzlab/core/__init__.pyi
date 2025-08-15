#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 22:04
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Any, Dict, Tuple, Optional

# 从具体子模块显式重导出，以便类型检查器识别
from .base import FuzznumStrategy as FuzznumStrategy
from .dispatcher import operate as operate
from .fuzznums import Fuzznum as Fuzznum
from .fuzzarray import Fuzzarray as Fuzzarray, fuzzarray as fuzzarray
from .backend import FuzzarrayBackend as FuzzarrayBackend
from .operation import (
    OperationMixin as OperationMixin,
    get_operation_registry as get_operation_registry,
    register_operation as register_operation
)
from .registry import (
    get_fuzznum_registry as get_fuzznum_registry,
    get_backend as get_backend,
    register_fuzznum as register_fuzznum,
    batch_register_fuzz as batch_register_fuzz,
    unregister_fuzznum as unregister_fuzznum,
    get_fuzznum_registered_mtypes as get_fuzznum_registered_mtypes
)
from .triangular import OperationTNorm as OperationTNorm

__all__ = [
    'FuzznumStrategy',
    'Fuzzarray',
    'Fuzznum',
    'FuzzarrayBackend',
    'OperationMixin',
    'OperationTNorm',
    'operate',
    'fuzzarray',
    'get_operation_registry',
    'register_operation',
    'get_fuzznum_registry',
    'get_backend',
    'register_fuzznum',
    'batch_register_fuzz',
    'unregister_fuzznum',
    'get_fuzznum_registered_mtypes'
]