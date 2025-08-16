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
    get_registry_fuzzify as get_registry_fuzzify,
    register_fuzzify as register_fuzzify,
)

__all__ = [
    'FuzzificationStrategy',
    'Fuzzifier',
    'fuzzify',
    'get_registry_fuzzify',
    'register_fuzzify',
]
