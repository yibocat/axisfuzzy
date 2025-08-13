#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/7/30 22:21
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from .base import FuzznumStrategy

from .dispatcher import operate

from .fuzznums import Fuzznum
from .fuzzarray import Fuzzarray, fuzzarray
from .backend import FuzzarrayBackend

from .operation import OperationMixin, get_operation_registry

from .registry import (
    get_fuzznum_registry,
    get_backend,
    register_fuzznum,
    batch_register_fuzz,
    unregister_fuzznum,
    get_fuzznum_registered_mtypes
)

# from .triangular import OperationTNorm

from .triangular import OperationTNorm

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
    'get_fuzznum_registry',
    'get_backend',
    'register_fuzznum',
    'batch_register_fuzz',
    'unregister_fuzznum',
    'get_fuzznum_registered_mtypes'
]
