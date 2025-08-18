#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/18 18:23
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Callable, Any

from .registry import ExtensionRegistry

class ExtensionDispatcher:

    registry: ExtensionRegistry

    def __init__(self) -> None: ...
    def create_instance_method(self, func_name: str) -> Callable[..., Any]: ...
    def create_top_level_function(self, func_name: str) -> Callable[..., Any]: ...
    def create_instance_property(self, func_name: str) -> property: ...

# module-level singleton
_dispatcher: ExtensionDispatcher

def get_extension_dispatcher() -> ExtensionDispatcher: ...
