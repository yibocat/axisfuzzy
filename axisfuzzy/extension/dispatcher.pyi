#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/17 22:38
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

from typing import Callable, Any, Dict, Optional

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
