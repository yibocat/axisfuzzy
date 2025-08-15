#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 00:52
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from dataclasses import dataclass, field
from typing import Any, Dict, Callable

@dataclass
class Config:
    DEFAULT_MTYPE: str
    DEFAULT_T_NORM: str
    DEFAULT_PRECISION: int
    DEFAULT_EPSILON: float
    STRICT_ATTRIBUTE_MODE: bool
    ENABLE_CACHE: bool
    CACHE_SIZE: int
    ENABLE_FUZZNUM_CACHE: bool
    ENABLE_PERFORMANCE_MONITORING: bool
    ENABLE_LOGGING: bool
    DEBUG_MODE: bool
    TNORM_VERIFY: bool

    def __init__(self,
                 DEFAULT_MTYPE: str = ...,
                 DEFAULT_T_NORM: str = ...,
                 DEFAULT_PRECISION: int = ...,
                 DEFAULT_EPSILON: float = ...,
                 STRICT_ATTRIBUTE_MODE: bool = ...,
                 ENABLE_CACHE: bool = ...,
                 CACHE_SIZE: int = ...,
                 ENABLE_FUZZNUM_CACHE: bool = ...,
                 ENABLE_PERFORMANCE_MONITORING: bool = ...,
                 ENABLE_LOGGING: bool = ...,
                 DEBUG_MODE: bool = ...,
                 TNORM_VERIFY: bool = ...) -> None: ...
