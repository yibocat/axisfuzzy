#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 22:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .base import FuzznumStrategy, FuzznumTemplate

from .dispatcher import operate

from .fuzzarray import Fuzzarray, fuzzarray
from .fuzznums import Fuzznum

from .operation import OperationMixin, get_operation_registry

from .registry import (
    get_fuzznum_registry,
    register_fuzznum,
    batch_register_fuzznums,
    unregister_fuzznum,
    get_fuzznum_registered_mtypes
)

from .triangular import OperationTNorm

__all__ = [
    'FuzznumStrategy',
    'FuzznumTemplate',
    'Fuzzarray',
    'Fuzznum',
    'OperationMixin',
    'OperationTNorm',
    'operate',
    'fuzzarray',
    'get_operation_registry',
    'get_fuzznum_registry',
    'register_fuzznum',
    'batch_register_fuzznums',
    'unregister_fuzznum',
    'get_fuzznum_registered_mtypes'
]
