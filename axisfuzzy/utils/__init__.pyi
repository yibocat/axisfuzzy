#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/15 17:22
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab

from typing import TypeVar, Callable, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def experimental(func: Callable[P, T]) -> Callable[P, T]: ...
def deprecated(func: Callable[P, T]) -> Callable[P, T]: ...