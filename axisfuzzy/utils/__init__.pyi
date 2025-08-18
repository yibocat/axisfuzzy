#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import TypeVar, Callable, ParamSpec, Type, overload, Any

P = ParamSpec("P")
T = TypeVar("T")

@overload
def experimental(func: Callable[P, T]) -> Callable[P, T]: ...
@overload
def experimental(cls: Type[T]) -> Type[T]: ...
def experimental(obj: Any) -> Any: ...

def deprecated(func: Callable[P, T]) -> Callable[P, T]: ...