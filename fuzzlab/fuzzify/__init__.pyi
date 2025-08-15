#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:26
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Type, Callable

from .base import FuzzificationStrategy as FuzzificationStrategy
from .fuzzifier import Fuzzifier as Fuzzifier, fuzzify as fuzzify

from .registry import (
    get_fuzzification_registry as get_fuzzification_registry,
    register_fuzzification_strategy as register_fuzzification_strategy,
)

__all__ = [
    'FuzzificationStrategy',
    'Fuzzifier',
    'fuzzify',
    'get_fuzzification_registry',
    'register_fuzzification_strategy',
]
