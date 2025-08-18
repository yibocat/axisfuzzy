#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from .base import FuzznumStrategy

from .dispatcher import operate

from .fuzznums import Fuzznum, fuzznum
from .fuzzarray import Fuzzarray, fuzzarray
from .backend import FuzzarrayBackend

from .operation import OperationMixin, get_registry_operation, register_operation

from .registry import (
    get_registry_fuzztype,
    register_strategy,
    register_backend,
    register_fuzztype,
    register_batch_fuzztypes,

    unregister_fuzztype,
    get_fuzztype_strategy,
    get_fuzztype_backend,
    get_fuzztype_mtypes
)

from .triangular import OperationTNorm

__all__ = [
    'FuzznumStrategy',
    'Fuzzarray',
    'Fuzznum',
    'FuzzarrayBackend',
    'OperationMixin',
    'OperationTNorm',
    'operate',
    'fuzznum',
    'fuzzarray',
    'get_registry_operation',
    'register_operation',
    'get_registry_fuzztype',
    'get_fuzztype_strategy',
    'get_fuzztype_backend',
    'register_strategy',
    'register_backend',
    'register_fuzztype',
    'register_batch_fuzztypes',
    'unregister_fuzztype',
    'get_fuzztype_mtypes'
]
