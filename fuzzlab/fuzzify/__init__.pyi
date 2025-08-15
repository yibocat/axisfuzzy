#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:26
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import Type, Callable

from .base import FuzzificationStrategy
from .registry import FuzzificationRegistry
from .fuzzifier import Fuzzifier, fuzzify

def get_fuzzification_registry() -> FuzzificationRegistry: ...

def register_fuzzification_strategy(
    mtype: str,
    method: str,
    is_default: bool = ...
) -> Callable[[Type[FuzzificationStrategy]], Type[FuzzificationStrategy]]: ...